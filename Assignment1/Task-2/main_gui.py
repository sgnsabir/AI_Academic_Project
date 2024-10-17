# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:00:43 2024

@author: Sabir
"""

import os
import pygame
import sys
from pygame.locals import *
from baggage_classifier import BaggageClassifier, parse_args
from tkinter import filedialog, Tk

# Initialize Tkinter and hide the main window
tk_root = Tk()
tk_root.withdraw()

pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Baggage Classifier")

# Set up fonts
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# Define buttons
upload_button = pygame.Rect(300, 50, 200, 50)
clear_button = pygame.Rect(300, 150, 200, 50)

# Variables to store results
result_text_bag_type = ""
result_text_bag_material = ""
size_text = ""

def train_if_needed():
    """Train the model if no pre-trained model exists."""
    args = parse_args()
    args.data_dir = 'D:/Masters/AI/Project/Assignments/1/task-2/Baggage'
    classifier = BaggageClassifier(
        model_name="efficientnet_b0",
        num_classes_bag_type=2,
        num_classes_bag_material=3,
        args=args
    )
    classifier.set_seeds()
    
    if not os.path.exists(args.model_save_path):
        print("No pre-trained model found. Starting training...")
        train_loader, val_loader = classifier.prepare_data()
        classifier.train_model(train_loader, val_loader)
        print("Training complete. Model saved.")
    else:
        classifier.load_model()
        print("Pre-trained model found. Loading model...")
    
    return classifier

def draw_button(text, rect):
    """Draw a button with centered text."""
    pygame.draw.rect(screen, BLUE, rect)
    text_surface = font.render(text, True, WHITE)
    # Center the text on the button
    screen.blit(
        text_surface, 
        (
            rect.x + (rect.width - text_surface.get_width()) // 2, 
            rect.y + (rect.height - text_surface.get_height()) // 2
        )
    )

def show_result(prediction_bag_type, prediction_bag_material, width, height):
    """Update the result display variables."""
    global result_text_bag_type, result_text_bag_material, size_text
    result_text_bag_type = f"Regular/Irregular: {prediction_bag_type}"
    result_text_bag_material = f"Materials: {prediction_bag_material}"
    size_text = f"Luggage Size: {width} x {height} pixels" if width and height else ""

def clear_display():
    """Reset the result display variables."""
    global result_text_bag_type, result_text_bag_material, size_text
    result_text_bag_type = ""
    result_text_bag_material = ""
    size_text = ""

def start_gui(classifier):
    """Run the main GUI loop."""
    image_display = None
    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if upload_button.collidepoint(event.pos):
                    # User clicked the upload button
                    file_path = filedialog.askopenfilename()
                    if file_path:
                        # Classify the image
                        prediction_bag_type, prediction_bag_material, width, height = classifier.classify_image_without_transformations(file_path)
                        show_result(prediction_bag_type, prediction_bag_material, width, height)
                        print(f"Classification Results: {prediction_bag_type}, {prediction_bag_material}, Size: {width}x{height}")
                        try:
                            # Load and scale the image for display
                            image_display = pygame.image.load(file_path)
                            image_display = pygame.transform.scale(image_display, (224, 224))
                        except Exception as e:
                            print(f"Error loading image: {e}")
                elif clear_button.collidepoint(event.pos):
                    # User clicked the clear button
                    clear_display()
                    image_display = None 
                    
        # Draw the buttons
        draw_button("Upload Image", upload_button)
        draw_button("Clear", clear_button)
        
        # Display the results
        y_offset = 250
        if result_text_bag_type:
            result_surface = small_font.render(result_text_bag_type, True, BLACK)
            screen.blit(result_surface, (50, y_offset))
            y_offset += 30
        if result_text_bag_material:
            result_surface = small_font.render(result_text_bag_material, True, BLACK)
            screen.blit(result_surface, (50, y_offset))
            y_offset += 30
        if size_text:
            size_surface = small_font.render(size_text, True, BLACK)
            screen.blit(size_surface, (50, y_offset))
            y_offset += 30
        if image_display:
            screen.blit(image_display, (550, 50))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    classifier = train_if_needed()
    start_gui(classifier)
