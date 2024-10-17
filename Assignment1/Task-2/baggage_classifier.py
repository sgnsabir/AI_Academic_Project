# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:15:43 2024

@author: Sabir
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import copy
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
from pathlib import Path

dataset_path = Path.home() / "Baggage"

def parse_args():
    """Parse command-line arguments.
    python main.py --data_dir /path/to/dataset --batch_size 16 --model_save_path /path/to/save/model.pth --num_epochs 30 --num_workers 8
    """
    parser = argparse.ArgumentParser(description="Baggage Classifier")
    parser.add_argument('--data_dir', type=str, default='Baggage', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--model_save_path', type=str, default='baggage_classifier_model.pth', help='Model save path')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    return parser.parse_args()

class MultiTaskImageFolder(datasets.ImageFolder):
    """Custom dataset for multi-task learning."""
    def __init__(self, root, transform=None, task1_mapping=None):
        super().__init__(root, transform)
        self.task1_mapping = task1_mapping or {}
        
    def __getitem__(self, index):
        image, label_bag_material = super().__getitem__(index)
        path, _ = self.samples[index]
        class_name = os.path.basename(os.path.dirname(path))        
        label_bag_type = self.task1_mapping.get(class_name, 0)
        return image, label_bag_type, label_bag_material

class BaggageClassifier:
    """Baggage classifier with multi-task learning."""
    def __init__(self, model_name="efficientnet_b0", num_classes_bag_type=2, num_classes_bag_material=3, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_classes_bag_type = num_classes_bag_type
        self.num_classes_bag_material = num_classes_bag_material
        self.args = args
        self.model = self.create_model()
        self.classes_bag_type = ['Regular', 'Irregular']
        self.classes_bag_material = []
    
    def set_seeds(self, seed=42):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def prepare_data(self):
        """Prepare data loaders for training and validation."""
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        task1_mapping = {
            'SoftBaggage': 0,             
            'HardPlastic Baggage': 0,     
            'MetalBaggage': 1,            
        }

        dataset = MultiTaskImageFolder(root=self.args.data_dir, transform=transform, task1_mapping=task1_mapping)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )

        self.classes_bag_material = dataset.classes

        return train_loader, val_loader

    def create_model(self):
        """Create the model with separate classifiers for each task."""
        if self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
            self.task1_classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.num_classes_bag_type)
            )

            self.task2_classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.num_classes_bag_material)
            )
        else:
            raise ValueError("Only 'efficientnet_b0' is supported.")

        for param in model.parameters():
            param.requires_grad = False
            
        for param in self.task1_classifier.parameters():
            param.requires_grad = True
        for param in self.task2_classifier.parameters():
            param.requires_grad = True

        model.to(self.device)
        self.task1_classifier.to(self.device)
        self.task2_classifier.to(self.device)
        return model

    def create_optimizer_scheduler(self):
        """Create optimizer and scheduler."""
        optimizer = optim.AdamW(
            list(self.task1_classifier.parameters()) + list(self.task2_classifier.parameters()),
            lr=0.001,
            weight_decay=1e-4
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return optimizer, scheduler

    def train_model(self, train_loader, val_loader):
        """Train the model."""
        criterion_bag_type = nn.CrossEntropyLoss()
        criterion_bag_material = nn.CrossEntropyLoss()
        optimizer, scheduler = self.create_optimizer_scheduler()

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_bag_type_wts = copy.deepcopy(self.task1_classifier.state_dict())
        best_bag_material_wts = copy.deepcopy(self.task2_classifier.state_dict())
        
        train_losses = []
        val_losses = []
        val_accuracies_bag_type = []
        val_accuracies_bag_material = []

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(self.args.num_epochs):
            print(f'Epoch {epoch+1}/{self.args.num_epochs}')

            self.model.train()
            self.task1_classifier.train()
            self.task2_classifier.train()
            running_loss = 0.0

            for inputs, labels_bag_type, labels_bag_material in train_loader:
                inputs = inputs.to(self.device)
                labels_bag_type = labels_bag_type.to(self.device)
                labels_bag_material = labels_bag_material.to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device_type):
                    features = self.model(inputs)
                    outputs_bag_type = self.task1_classifier(features)
                    outputs_bag_material = self.task2_classifier(features)
                    loss_bag_type = criterion_bag_type(outputs_bag_type, labels_bag_type)
                    loss_bag_material = criterion_bag_material(outputs_bag_material, labels_bag_material)
                    loss = loss_bag_type + loss_bag_material

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            val_loss, val_acc_bag_type, val_acc_bag_material = self.validate_model(val_loader, criterion_bag_type, criterion_bag_material)
            val_losses.append(val_loss)
            val_accuracies_bag_type.append(val_acc_bag_type)
            val_accuracies_bag_material.append(val_acc_bag_material)
            print(f'Epoch [{epoch+1}/{self.args.num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy Task 1: {val_acc_bag_type:.4f}, Task 2: {val_acc_bag_material:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_bag_type_wts = copy.deepcopy(self.task1_classifier.state_dict())
                best_bag_material_wts = copy.deepcopy(self.task2_classifier.state_dict())

            scheduler.step(val_loss)
            
        self.model.load_state_dict(best_model_wts)
        self.task1_classifier.load_state_dict(best_bag_type_wts)
        self.task2_classifier.load_state_dict(best_bag_material_wts)
        self.save_model()
        self.plot_metrics(train_losses, val_losses, val_accuracies_bag_type, val_accuracies_bag_material)
        self.evaluate_model(val_loader)

    def plot_metrics(self, train_losses, val_losses, val_accuracies_bag_type, val_accuracies_bag_material):
        """Plot training and validation metrics."""
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b', label='Training Loss')
        plt.plot(epochs, val_losses, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies_bag_type, 'b', label='Validation Accuracy Task 1')
        plt.plot(epochs, val_accuracies_bag_material, 'g', label='Validation Accuracy Task 2')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def validate_model(self, val_loader, criterion_bag_type, criterion_bag_material):
        """Validate the model."""
        self.model.eval()
        self.task1_classifier.eval()
        self.task2_classifier.eval()
        val_loss = 0.0
        all_labels_bag_type, all_preds_bag_type = [], []
        all_labels_bag_material, all_preds_bag_material = [], []

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.no_grad():
            for inputs, labels_bag_type, labels_bag_material in val_loader:
                inputs = inputs.to(self.device)
                labels_bag_type = labels_bag_type.to(self.device)
                labels_bag_material = labels_bag_material.to(self.device)
                with torch.amp.autocast(device_type=device_type):
                    features = self.model(inputs)
                    outputs_bag_type = self.task1_classifier(features)
                    outputs_bag_material = self.task2_classifier(features)
                    loss_bag_type = criterion_bag_type(outputs_bag_type, labels_bag_type)
                    loss_bag_material = criterion_bag_material(outputs_bag_material, labels_bag_material)
                    loss = loss_bag_type + loss_bag_material

                val_loss += loss.item()
                _, preds_bag_type = torch.max(outputs_bag_type, 1)
                _, preds_bag_material = torch.max(outputs_bag_material, 1)
                all_labels_bag_type.extend(labels_bag_type.cpu().numpy())
                all_preds_bag_type.extend(preds_bag_type.cpu().numpy())
                all_labels_bag_material.extend(labels_bag_material.cpu().numpy())
                all_preds_bag_material.extend(preds_bag_material.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc_bag_type = accuracy_score(all_labels_bag_type, all_preds_bag_type)
        val_acc_bag_material = accuracy_score(all_labels_bag_material, all_preds_bag_material)
        return val_loss, val_acc_bag_type, val_acc_bag_material

    def evaluate_model(self, val_loader):
        """Evaluate the model and display confusion matrices."""
        self.model.eval()
        self.task1_classifier.eval()
        self.task2_classifier.eval()
        all_labels_bag_type, all_preds_bag_type = [], []
        all_labels_bag_material, all_preds_bag_material = [], []

        with torch.no_grad():
            for inputs, labels_bag_type, labels_bag_material in val_loader:
                inputs = inputs.to(self.device)
                labels_bag_type = labels_bag_type.cpu().numpy()
                labels_bag_material = labels_bag_material.cpu().numpy()
                features = self.model(inputs)
                outputs_bag_type = self.task1_classifier(features)
                outputs_bag_material = self.task2_classifier(features)
                _, preds_bag_type = torch.max(outputs_bag_type, 1)
                _, preds_bag_material = torch.max(outputs_bag_material, 1)
                all_labels_bag_type.extend(labels_bag_type)
                all_preds_bag_type.extend(preds_bag_type.cpu().numpy())
                all_labels_bag_material.extend(labels_bag_material)
                all_preds_bag_material.extend(preds_bag_material.cpu().numpy())

        # Confusion matrix for bag type
        cm_bag_type = confusion_matrix(all_labels_bag_type, all_preds_bag_type)
        disp_bag_type = ConfusionMatrixDisplay(confusion_matrix=cm_bag_type, display_labels=self.classes_bag_type)
        disp_bag_type.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix Regular/Irregular")
        plt.show()

        # Confusion matrix for bag material
        cm_bag_material = confusion_matrix(all_labels_bag_material, all_preds_bag_material)
        disp_bag_material = ConfusionMatrixDisplay(confusion_matrix=cm_bag_material, display_labels=self.classes_bag_material)
        disp_bag_material.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix Material Type")
        plt.show()

    def save_model(self):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'task1_classifier_state_dict': self.task1_classifier.state_dict(),
            'task2_classifier_state_dict': self.task2_classifier.state_dict(),
            'classes_bag_type': self.classes_bag_type,
            'classes_bag_material': self.classes_bag_material
        }, self.args.model_save_path)
        print(f"Model saved to {self.args.model_save_path}")

    def load_model(self):
        """Load a pre-trained model."""
        checkpoint = torch.load(self.args.model_save_path)
        print("Checkpoint keys:", checkpoint.keys())

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.task1_classifier.load_state_dict(checkpoint['task1_classifier_state_dict'])
            self.task2_classifier.load_state_dict(checkpoint['task2_classifier_state_dict'])
            self.classes_bag_type = checkpoint.get('classes_bag_type', ['Regular', 'Irregular'])
            self.classes_bag_material = checkpoint.get('classes_bag_material', [])
        else:
            self.model.load_state_dict(checkpoint, strict=False)
            print("Loaded model state_dict directly.")

    def estimate_size(self, image_tensor):
        """Estimate the size of the luggage in the image."""
        image_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return w, h  
        
        return None, None

    def classify_image_without_transformations(self, image_path):
        """Classify an image without applying transformations."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
            image_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(image_tensor)
            
            self.model.eval()
            self.task1_classifier.eval()
            self.task2_classifier.eval()
            
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.no_grad():
                features = self.model(image_tensor)
                outputs_bag_type = self.task1_classifier(features)
                outputs_bag_material = self.task2_classifier(features)
                probabilities_bag_type = torch.nn.functional.softmax(outputs_bag_type, dim=1)
                probabilities_bag_material = torch.nn.functional.softmax(outputs_bag_material, dim=1)
                confidence_bag_type, predicted_bag_type = torch.max(probabilities_bag_type, 1)
                confidence_bag_material, predicted_bag_material = torch.max(probabilities_bag_material, 1)
            
            confidence_threshold = 0.4
            predicted_class_bag_type = (
                self.classes_bag_type[predicted_bag_type.item()] 
                if confidence_bag_type.item() >= confidence_threshold else "None"
            )
            predicted_class_bag_material = (
                self.classes_bag_material[predicted_bag_material.item()] 
                if confidence_bag_material.item() >= confidence_threshold else "None"
            )
            
            width, height = self.estimate_size(image_tensor)
            
            return predicted_class_bag_type, predicted_class_bag_material, width, height
        
        except Exception as e:
            print(f"Error in classifying image: {e}")
            return "Error", "Error", None, None

if __name__ == '__main__':
    args = parse_args()
    args.data_dir = dataset_path
    classifier = BaggageClassifier(
        model_name="efficientnet_b0",
        num_classes_bag_type=2,
        num_classes_bag_material=3,
        args=args
    )
    classifier.set_seeds()
    
    if not os.path.exists(args.model_save_path):
        train_loader, val_loader = classifier.prepare_data()
        classifier.train_model(train_loader, val_loader)
        print("Model training complete and saved.")
    else:
        classifier.load_model()
        print("Pre-trained model found and loaded.")
