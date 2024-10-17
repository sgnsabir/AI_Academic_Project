# -*- coding: utf-8 -*-
"""
Created on Sat Sep 7 10:22:32 2024

@author: Sabir
"""

import random
import requests
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import config
import copy
import logging
import time 

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hugging Face API token
HUGGING_FACE_API_TOKEN = config.HUGGINGFACE_TOKEN

class LuggageGA:
    """Genetic Algorithm representation of a luggage design."""
    def __init__(self, weight, capacity, material, has_wheels):
        self.weight = weight
        self.capacity = capacity
        self.material = material
        self.has_wheels = has_wheels
        self.height = self.capacity * 0.24  
        self.width = self.capacity * 0.16 
        self.depth = self.capacity * 0.1

    def __repr__(self):
        return (f"LuggageGA(weight={self.weight:.2f}, capacity={self.capacity:.2f}, "
                f"material={self.material}, has_wheels={self.has_wheels})")

    def fitness(self):
        """Calculate the fitness score based on various factors."""
        material_factor = {'plastic': 1.0, 'fabric': 0.8, 'metal': 1.2}
        wheels_factor = 1.5 if self.has_wheels else 1.0
        fitness_value = ((self.capacity / self.weight) * material_factor.get(self.material, 1.0) * wheels_factor)
        return fitness_value

def create_initial_population(size):
    """Initialize a population of luggage designs."""
    materials = ['plastic', 'fabric', 'metal']
    population = []
    for _ in range(size):
        weight = random.uniform(2.0, 5.0)
        capacity = random.uniform(15, 50)
        material = random.choice(materials)
        has_wheels = random.choice([True, False])
        luggage = LuggageGA(weight, capacity, material, has_wheels)
        population.append(luggage)
    return population

def crossover(parent1, parent2):
    """Combine attributes from two parents to create a child."""
    child_weight = random.choice([parent1.weight, parent2.weight])
    child_capacity = random.choice([parent1.capacity, parent2.capacity])
    child_material = random.choice([parent1.material, parent2.material])
    child_has_wheels = random.choice([parent1.has_wheels, parent2.has_wheels])
    return LuggageGA(child_weight, child_capacity, child_material, child_has_wheels)

def mutate(luggage, mutation_rate=0.1):
    """Introduce random changes to a luggage design."""
    if random.random() < mutation_rate:
        luggage.weight = max(1.0, luggage.weight + random.uniform(-0.5, 0.5))
        luggage.capacity = max(10.0, luggage.capacity + random.uniform(-10, 10))
        if random.random() < 0.3:
            luggage.material = random.choice(['plastic', 'fabric', 'metal'])
        if random.random() < 0.1:
            luggage.has_wheels = not luggage.has_wheels
        luggage.height = luggage.capacity * 0.24
        luggage.width = luggage.capacity * 0.16
        luggage.depth = luggage.capacity * 0.1
    return luggage

def select_parents(population, num_parents):
    """Choose the top-performing designs to be parents."""
    sorted_pop = sorted(population, key=lambda x: x.fitness(), reverse=True)
    return sorted_pop[:num_parents]

def cluster_population(population, num_clusters):
    """Group the population into clusters using K-Means."""
    features = np.array([[l.weight, l.capacity] for l in population])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    clusters = [[] for _ in range(num_clusters)]
    for label, luggage in zip(labels, population):
        clusters[label].append(luggage)
    return clusters

def reduce_dimensions(population, num_components=2):
    """Reduce feature dimensions using PCA."""
    features = np.array([[l.weight, l.capacity] for l in population])
    pca = PCA(n_components=num_components, random_state=42)
    return pca.fit_transform(features)

def reinforce_fitness(population, previous_best):
    """Boost fitness scores that exceed the previous best."""
    for l in population:
        if l.fitness() > previous_best:
            l.fitness_value = l.fitness() + (l.fitness() - previous_best) * 0.1
        else:
            l.fitness_value = l.fitness()

def adapt_mutation_rate(current, previous, rate):
    """Adjust mutation rate based on fitness improvement."""
    if current > previous:
        return max(0.01, rate * 0.9)
    else:
        return min(0.5, rate * 1.1)

def evolve_population(population, generations, mutation_rate=0.1, clusters=3, components=2):
    """Run the genetic algorithm over multiple generations."""
    initial_size = len(population)
    best_fitness = float('-inf')
    elite_size = 2
    avg_history, best_history = [], []

    for gen in range(generations):
        logging.info(f"Generation {gen + 1}: Population size {len(population)}")
        pca_features = reduce_dimensions(population, components)
        clustered = cluster_population(population, clusters)
        parents = []
        for cluster in clustered:
            if cluster:
                parents.extend(select_parents(cluster, max(2, len(cluster) // 2)))
        next_gen = []
        sorted_pop = sorted(population, key=lambda x: x.fitness(), reverse=True)
        elites = copy.deepcopy(sorted_pop[:elite_size])
        next_gen.extend(elites)
        logging.info(f"Elites carried over: {elite_size} individuals.")
        while len(next_gen) < initial_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_gen.append(child)
        if not next_gen:
            raise ValueError("Next generation is empty.")
        best_luggage = max(next_gen, key=lambda x: x.fitness())
        best_fitness_current = best_luggage.fitness()
        avg_fitness = sum(l.fitness() for l in next_gen) / len(next_gen)
        avg_history.append(avg_fitness)
        best_history.append(best_fitness_current)
        logging.info(f"Best: {best_luggage} Fitness: {best_fitness_current:.2f}, Avg Fitness: {avg_fitness:.2f}")
        reinforce_fitness(next_gen, best_fitness)
        best_fitness = max(best_fitness, best_fitness_current)
        mutation_rate = adapt_mutation_rate(best_fitness_current, best_fitness, mutation_rate)
        logging.info(f"Mutation rate adjusted to {mutation_rate:.4f}")
        population = next_gen

    plot_fitness(avg_history, best_history)
    plot_pca(population, clusters, components)
    return max(population, key=lambda x: x.fitness())

def plot_fitness(avg, best):
    """Display fitness trends over generations."""
    generations = range(1, len(avg) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg, label="Average Fitness", color='blue')
    plt.plot(generations, best, label="Best Fitness", color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_progression.png')
    plt.close()
    logging.info("Saved fitness progression plot.")

def plot_pca(population, clusters, components):
    """Visualize clustering after PCA."""
    pca = PCA(n_components=components, random_state=42)
    features = np.array([[l.weight, l.capacity] for l in population])
    pca_features = pca.fit_transform(features)
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Clustering')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    plt.savefig('pca_clustering.png')
    plt.close()
    logging.info("Saved PCA clustering plot.")

def generate_luggage_description(luggage):
    """Create a concise description of the luggage design."""
    color = "gray" if luggage.material == "metal" else "blue" if luggage.material == "plastic" else "brown"
    description = (
        f"A modern, travel-friendly suitcase with dimensions {luggage.height:.2f}\"H x {luggage.width:.2f}\"W x {luggage.depth:.2f}\"D. "
        f"Constructed from durable {luggage.material}, featuring a sleek {color} finish. "
        f"It {'has four spinner wheels' if luggage.has_wheels else 'does not have spinner wheels'}, ensuring easy maneuverability. "
        "Includes a telescopic handle, TSA-approved locks, and an organized interior with compartments for garments and electronics. "
        "Designed to balance style with functionality for an optimal travel experience."
    )
    return description

def visualize_with_diffusion_model(description, max_retries=3, backoff_factor=2):
    """Generate an image using a diffusion model via Hugging Face API with retry logic."""
    logging.info(f"Generating image with description: {description}")
    
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}
    json_data = {
        "inputs": description,
        "options": {"wait_for_model": True}
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
                headers=headers,
                json=json_data
            )
            
            if response.status_code == 200:
                image_data = response.content
                image = Image.open(BytesIO(image_data))
                image.save('final_diffusion_image.png')
                logging.info("Image generated and saved as 'final_diffusion_image.png'.")
                return
            else:
                logging.error(f"Failed to generate image: {response.status_code}, {response.text}")
                if response.status_code >= 500:
                    # Retry on server-side errors
                    wait_time = backoff_factor ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Do not retry on client-side errors
                    break
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred: {e}")
            wait_time = backoff_factor ** attempt
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    logging.error("Image generation failed after multiple attempts.")

def plot_luggage_exterior(luggage):
    """Create a 3D plot of the luggage exterior."""
    color = "gray" if luggage.material == "metal" else "blue" if luggage.material == "plastic" else "brown"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vertices = np.array([[0, 0, 0],
                         [luggage.width, 0, 0],
                         [luggage.width, luggage.depth, 0],
                         [0, luggage.depth, 0],
                         [0, 0, luggage.height],
                         [luggage.width, 0, luggage.height],
                         [luggage.width, luggage.depth, luggage.height],
                         [0, luggage.depth, luggage.height]])
    faces = [[vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [7, 6, 2, 3]],
             [vertices[j] for j in [0, 3, 7, 4]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]]]
    for face in faces:
        poly = Poly3DCollection([face], color=color, alpha=0.6)
        poly.set_edgecolor('black')
        ax.add_collection3d(poly)
    ax.set_xlabel('Width (inches)')
    ax.set_ylabel('Depth (inches)')
    ax.set_zlabel('Height (inches)')
    ax.set_xlim([0, luggage.width * 1.2])
    ax.set_ylim([0, luggage.depth * 1.2])
    ax.set_zlim([0, luggage.height * 1.2])
    ax.set_title('Luggage Exterior')
    plt.savefig('exterior_view.png')
    plt.close()
    logging.info("Saved exterior view plot.")

def plot_luggage_interior(luggage):
    """Create a 2D plot of the luggage interior."""
    fig, ax = plt.subplots(figsize=(6, 8))
    main_height = luggage.height * 0.7
    upper_height = luggage.height * 0.3
    main = plt.Rectangle((0, 0), luggage.width, main_height, edgecolor='black', facecolor='lightblue', alpha=0.5)
    ax.add_patch(main)
    ax.text(luggage.width / 2, main_height / 2, 'Main Compartment', fontsize=12, ha='center', va='center')
    upper = plt.Rectangle((0, main_height), luggage.width, upper_height, edgecolor='black', facecolor='lightgreen', alpha=0.5)
    ax.add_patch(upper)
    ax.text(luggage.width / 2, main_height + upper_height / 2, 'Upper Compartment', fontsize=12, ha='center', va='center')
    electronics = plt.Rectangle((luggage.width * 0.1, luggage.height * 0.6), luggage.width * 0.3, luggage.height * 0.1, edgecolor='black', facecolor='yellow', alpha=0.5)
    ax.add_patch(electronics)
    ax.text(luggage.width * 0.25, luggage.height * 0.65, 'Electronics', fontsize=10, ha='center', va='center')
    garments = plt.Rectangle((luggage.width * 0.6, luggage.height * 0.6), luggage.width * 0.3, luggage.height * 0.1, edgecolor='black', facecolor='pink', alpha=0.5)
    ax.add_patch(garments)
    ax.text(luggage.width * 0.75, luggage.height * 0.65, 'Garments', fontsize=10, ha='center', va='center')
    ax.set_xlim(0, luggage.width)
    ax.set_ylim(0, luggage.height)
    ax.set_xlabel('Width (inches)')
    ax.set_ylabel('Height (inches)')
    ax.set_title('Luggage Interior View')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('interior_view.png')
    plt.close()
    logging.info("Saved interior view plot.")

def main():
    population_size = 20
    num_generations = 30
    mutation_rate = 0.15
    num_clusters = 3
    num_components = 2

    population = create_initial_population(population_size)
    best_luggage = evolve_population(population, num_generations, mutation_rate, num_clusters, num_components)
    
    # Determine the color based on material
    color = "gray" if best_luggage.material == "metal" else "blue" if best_luggage.material == "plastic" else "brown"

    # Generate optimized description
    description = generate_luggage_description(best_luggage)

    visualize_with_diffusion_model(description)
    plot_luggage_exterior(best_luggage)
    plot_luggage_interior(best_luggage)

if __name__ == "__main__":
    main()
