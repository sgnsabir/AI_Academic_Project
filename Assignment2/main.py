# -*- coding: utf-8 -*-
"""
Created on Sat Sept 14 10:53:04 2024

@author: Sabir
"""

import os
import pygame
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from robot_logic import RobotAgent, DDPGAgent, shared_memory, pheromone_map, Target, get_initial_plane_positions

# Initialize Pygame
pygame.init()

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 102, 204)
LIGHT_BLUE = (173, 216, 230)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
GRAY = (200, 200, 200)
DARK_GRAY = (169, 169, 169)

# Screen Dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Button Dimensions
BUTTON_WIDTH = 80
BUTTON_HEIGHT = 40
BUTTON_FONT_SIZE = 24

# Other Dimensions
TERMINAL_WIDTH = 300
TERMINAL_HEIGHT = 350
PARKING_AREA_WIDTH = 170
PARKING_AREA_HEIGHT = 80
PLANE_WIDTH = 50
PLANE_HEIGHT = 50
LAMP_WIDTH = 20
LAMP_HEIGHT = 220
CART_WIDTH = 80
CART_HEIGHT = 80
CHARGING_STATION_WIDTH = 60
CHARGING_STATION_HEIGHT = 60
ROBOT_WIDTH = 20
ROBOT_HEIGHT = 20
GRID_SIZE = 20

# Environment variable to avoid OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Creating Screen
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Airport Robot Simulation')

# Loading Images
try:
    plane_image = pygame.image.load('elements/plane.png')
    plane_image = pygame.transform.scale(plane_image, (PLANE_WIDTH, PLANE_HEIGHT))

    charging_station_image = pygame.image.load('elements/charging_station.png')
    charging_station_image = pygame.transform.scale(charging_station_image, (CHARGING_STATION_WIDTH, CHARGING_STATION_HEIGHT))

    cart_image = pygame.image.load('elements/cart.png')
    cart_image = pygame.transform.scale(cart_image, (CART_WIDTH, CART_HEIGHT))
except pygame.error as e:
    print(f"Error loading images: {e}")
    pygame.quit()
    sys.exit()

# Positions
terminal_x = WINDOW_WIDTH // 2 - TERMINAL_WIDTH // 2 + 200
charging_station_y = 50 + TERMINAL_HEIGHT - 50
"""
def get_initial_plane_positions():
    return [
        Target(pygame.Rect(50, 150, PLANE_WIDTH, PLANE_HEIGHT), id=1),
        Target(pygame.Rect(50, 250, PLANE_WIDTH, PLANE_HEIGHT), id=2),
        Target(pygame.Rect(50, 350, PLANE_WIDTH, PLANE_HEIGHT), id=3)
    ]
"""
plane_positions = get_initial_plane_positions()

# Define Obstacles
obstacles = [
    pygame.Rect(WINDOW_WIDTH // 2 - 50, 50, 100, 50),
    pygame.Rect(WINDOW_WIDTH // 2 - 150, 200, LAMP_WIDTH, LAMP_HEIGHT),
    pygame.Rect(WINDOW_WIDTH // 2 - PARKING_AREA_WIDTH // 2, WINDOW_HEIGHT - PARKING_AREA_HEIGHT - 20, PARKING_AREA_WIDTH, PARKING_AREA_HEIGHT),
    pygame.Rect(terminal_x + 50, 150, CART_WIDTH, CART_HEIGHT),
    pygame.Rect(terminal_x + TERMINAL_WIDTH - 50 - CART_WIDTH, 150, CART_WIDTH, CART_HEIGHT)
]

# Robot Starting Positions
robot_start_positions = [
    (terminal_x + 50, charging_station_y),
    (terminal_x + TERMINAL_WIDTH - 50 - CHARGING_STATION_WIDTH, charging_station_y)
]

def initialize_simulation():
    global ddpg_agent, robots, plane_positions
    shared_memory.clear()
    pheromone_map.clear()
    plane_positions = get_initial_plane_positions()
    ddpg_agent = DDPGAgent(state_size=8, action_size=4)
    robots = [
        RobotAgent(robot_start_positions[0][0], robot_start_positions[0][1], RED, id=1, robots=[],
                   ddpg_agent=ddpg_agent, obstacles=obstacles, grid_size=GRID_SIZE,
                   robot_width=ROBOT_WIDTH, robot_height=ROBOT_HEIGHT,
                   window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, plane_positions=plane_positions),
        RobotAgent(robot_start_positions[1][0], robot_start_positions[1][1], GREEN, id=2, robots=[],
                   ddpg_agent=ddpg_agent, obstacles=obstacles, grid_size=GRID_SIZE,
                   robot_width=ROBOT_WIDTH, robot_height=ROBOT_HEIGHT,
                   window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, plane_positions=plane_positions)
    ]
    for robot in robots:
        robot.robots = robots

initialize_simulation()

class MovingObject:
    def __init__(self, x, y, width, height, color, path, speed):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.path = path
        self.speed = speed
        self.current_target_index = 0

    def move(self):
        if self.current_target_index < len(self.path):
            target_pos = self.path[self.current_target_index]
            dx = target_pos[0] - self.rect.x
            dy = target_pos[1] - self.rect.y
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist != 0:
                dx /= dist
                dy /= dist
                self.rect.x += int(dx * self.speed)
                self.rect.y += int(dy * self.speed)
            if abs(self.rect.x - target_pos[0]) < self.speed and abs(self.rect.y - target_pos[1]) < self.speed:
                self.current_target_index += 1
        else:
            self.current_target_index = 0

# Moving Objects to simulate airport
moving_objects = [
    MovingObject(200, 100, 30, 30, YELLOW, [(200, 100), (200, 500), (200, 100)], speed=20),
    MovingObject(600, 500, 30, 30, GRAY, [(600, 500), (600, 100), (600, 500)], speed=20),
    MovingObject(400, 100, 30, 30, PURPLE, [(400, 100), (400, 500), (400, 100)], speed=20)
]

def update_moving_objects():
    for obj in moving_objects:
        obj.move()
        pygame.draw.rect(screen, obj.color, obj.rect)
        obstacles.append(obj.rect.copy())

def remove_moving_objects_from_obstacles():
    for obj in moving_objects:
        obstacles[:] = [ob for ob in obstacles if ob != obj.rect]

def create_button(text, x, y, width, height, color, font_size):
    pygame.draw.rect(screen, color, [x, y, width, height])
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, (x + (width - text_surface.get_width()) // 2, y + (height - text_surface.get_height()) // 2))

def check_button_click(x, y, width, height):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x < mouse[0] < x + width and y < mouse[1] < y + height:
        if click[0] == 1:
            return True
    return False

def save_training_data(robots):
    training_data = []
    for robot in robots:
        robot_data = {
            'id': robot.id,
            'states': [state.tolist() for state in robot.memory['states']],
            'actions': [action.tolist() for action in robot.memory['actions']],
            'rewards': robot.memory['rewards'],
            'episode_rewards': robot.episode_rewards,
            'path_lengths': robot.path_lengths,
        }
        training_data.append(robot_data)

    ddpg_data = {
        'actor_losses': ddpg_agent.actor_losses,
        'critic_losses': ddpg_agent.critic_losses,
    }

    try:
        with open('training_data.json', 'w') as f:
            json.dump({'robots': training_data, 'ddpg': ddpg_data}, f)
        print("Training data saved.")
    except IOError as e:
        print(f"Error saving training data: {e}")

def display_shared_memory():
    print("Shared Memory State:")
    for position, data in shared_memory.items():
        print(f"Position {position}: Collided = {data['collided']}, Collided With = {data['collided_with']}")

    print("Pheromone Map State:")
    for position, pheromone in pheromone_map.items():
        print(f"Position {position}: Pheromone Level = {pheromone}")

def plot_learning_curves(robots, ddpg_agent):
    # Plot rewards per episode for each robot
    plt.figure()
    for robot in robots:
        if robot.episode_rewards:  # Check if data exists
            plt.plot(robot.episode_rewards, label=f'Robot {robot.id} Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    # Ensure x-axis shows only integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot actor and critic losses from the agent
    plt.figure()
    if ddpg_agent.actor_losses and ddpg_agent.critic_losses:  # Check if data exists
        plt.plot(ddpg_agent.actor_losses, label='Actor Loss')
        plt.plot(ddpg_agent.critic_losses, label='Critic Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Actor and Critic Losses')
    plt.legend()

    # Plot path lengths per episode for each robot
    plt.figure()
    for robot in robots:
        if robot.path_lengths:  # Check if data exists
            plt.plot(robot.path_lengths, label=f'Robot {robot.id} Path Length')
    plt.xlabel('Episode')
    plt.ylabel('Path Length')
    plt.title('Path Lengths per Episode')
    plt.legend()
    # Ensure x-axis shows only integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Show all plots
    plt.show(block=True)

# Control Flags
running = False
simulation_active = True

while simulation_active:
    screen.fill(BLUE)

    # Buttons
    create_button('Start', 50, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT, GREEN, BUTTON_FONT_SIZE)
    create_button('Stop', 150, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT, YELLOW, BUTTON_FONT_SIZE)
    create_button('Exit', WINDOW_WIDTH - BUTTON_WIDTH - 50, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT, RED, BUTTON_FONT_SIZE)

    # Terminal
    pygame.draw.rect(screen, LIGHT_BLUE, [terminal_x, 100, TERMINAL_WIDTH, TERMINAL_HEIGHT])
    pygame.draw.rect(screen, BLACK, [terminal_x, 100, TERMINAL_WIDTH, TERMINAL_HEIGHT], 2)
    font = pygame.font.Font(None, 20)
    text = font.render('Terminal', True, BLACK)
    screen.blit(text, (terminal_x + TERMINAL_WIDTH//2 - text.get_width()//2, 120))

    # Planes
    for plane in plane_positions:
        screen.blit(plane_image, (plane.rect.x, plane.rect.y))

    # Fire Station
    pygame.draw.rect(screen, LIGHT_BLUE, [WINDOW_WIDTH//2 - 50, 10, 100, 50])
    text = font.render('Fire station', True, BLACK)
    screen.blit(text, (WINDOW_WIDTH//2 - text.get_width()//2, 25))

    # Lamps
    pygame.draw.rect(screen, DARK_GRAY, [WINDOW_WIDTH//2 - 150, 200, LAMP_WIDTH, LAMP_HEIGHT])
    text = pygame.font.Font(None, 20).render('LAMPS', True, BLACK)
    screen.blit(text, (WINDOW_WIDTH//2 - 150 + LAMP_WIDTH + 5, 250))

    # Parking Area
    pygame.draw.rect(screen, LIGHT_BLUE, [WINDOW_WIDTH//2 - PARKING_AREA_WIDTH//2, WINDOW_HEIGHT - PARKING_AREA_HEIGHT - 20, PARKING_AREA_WIDTH, PARKING_AREA_HEIGHT])
    text = font.render('Parking area for empty carts etc.', True, BLACK)
    screen.blit(text, (WINDOW_WIDTH//2 - text.get_width()//2, WINDOW_HEIGHT - PARKING_AREA_HEIGHT - 20 + PARKING_AREA_HEIGHT//2 - 10))

    # Charging Stations
    screen.blit(charging_station_image, (terminal_x + 50, charging_station_y))
    screen.blit(charging_station_image, (terminal_x + TERMINAL_WIDTH - 50 - CHARGING_STATION_WIDTH, charging_station_y))

    # Carts
    cart_y = 150
    screen.blit(cart_image, (terminal_x + 50, cart_y))
    screen.blit(cart_image, (terminal_x + TERMINAL_WIDTH - 50 - CART_WIDTH, cart_y))

    # Handling Button Clicks
    if check_button_click(50, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT):
        running = True
    if check_button_click(150, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT):
        running = False
        save_training_data(robots)
        display_shared_memory()
        initialize_simulation()
    if check_button_click(WINDOW_WIDTH - BUTTON_WIDTH - 50, WINDOW_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT):
        save_training_data(robots)
        pygame.quit()
        plot_learning_curves(robots, ddpg_agent)
        sys.exit()

    # Updating Moving Objects
    if running:
        remove_moving_objects_from_obstacles()
        update_moving_objects()
        for robot in robots:
            robot.coordinate_with_other_robots()
            robot.move()
            pygame.draw.rect(screen, robot.color, (robot.x, robot.y, ROBOT_WIDTH, ROBOT_HEIGHT))
            
            # Display battery status to the left of the robot
            font_status = pygame.font.Font(None, 18)
            battery_text = f"SOC: {int(robot.soc)}%"
            battery_surface = font_status.render(battery_text, True, BLACK)
            battery_x = robot.x - battery_surface.get_width() - 5  # Position text to the left of the robot
            battery_y = robot.y - 10  # Slightly above the robot
            screen.blit(battery_surface, (battery_x, battery_y))
            
            if robot.path:
                for pos in robot.path:
                    pygame.draw.circle(screen, YELLOW, pos, 2)
        remove_moving_objects_from_obstacles()

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_training_data(robots)
            pygame.quit()
            plot_learning_curves(robots, ddpg_agent)
            sys.exit()

    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()
plot_learning_curves(robots, ddpg_agent)
sys.exit()
