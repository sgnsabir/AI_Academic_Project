import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
import heapq

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Shared memory for mistakes (stores mistakes made by robots at certain positions)
shared_memory = defaultdict(lambda: {"collided": False, "collided_with": None})
pheromone_map = defaultdict(lambda: 0.1)

def get_initial_plane_positions():
    return [
        Target(pygame.Rect(50, 150, 50, 50), id=1),
        Target(pygame.Rect(50, 250, 50, 50), id=2),
        Target(pygame.Rect(50, 350, 50, 50), id=3)
    ]

class ReplayBuffer:
    def __init__(self, max_size, batch_size):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        if len(state.shape) == 1:
            state = state.unsqueeze(0) 
        if len(action.shape) == 1:
            action = action.unsqueeze(0)  

        # Check if both are 2D
        assert len(state.shape) == 2 and len(action.shape) == 2, "State and action must be 2D tensors"

        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class DDPGAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=0.0001,
        lr_critic=0.001,
        gamma=0.99,
        tau=0.001,
        max_grad_norm=0.5,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        # Actor and Critic networks
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_actor = Actor(state_size, action_size).to(device)
        self.target_critic = Critic(state_size, action_size).to(device)

        # Copy the weights of the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.memory = ReplayBuffer(max_size=10000, batch_size=64)

        # Metrics for plotting
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state, noise_stddev=0.2):
        state = torch.tensor(state, dtype=torch.float32).to(next(self.actor.parameters()).device)
        with torch.no_grad():
            action = self.actor(state).cpu().detach().numpy()
        noise = np.random.normal(0, noise_stddev, size=self.action_size)
        action += noise  # Add some noise for exploration
        return np.clip(action, -1, 1)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.memory.batch_size:
            return

        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Convert to tensors and ensure proper shape
        device = next(self.actor.parameters()).device
        states = torch.tensor(states, dtype=torch.float32).to(device).view(-1, self.state_size)
        actions = torch.tensor(actions, dtype=torch.float32).to(device).view(-1, self.action_size)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device).view(-1, self.state_size)
        dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1)

        # Critic loss
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

        current_q_values = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor loss
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Update Target Networks
        self.update_target_networks()

        # Store losses for plotting
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

    def update_target_networks(self):
        tau = self.tau
        with torch.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.copy_(tau * param + (1 - tau) * target_param)

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.copy_(tau * param + (1 - tau) * target_param)

# A* Pathfinding Algorithm
def heuristic(a, b):
    """Heuristic function for A* (Manhattan distance)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position, grid_size, obstacles, window_width, window_height, robot_width, robot_height):
    """Get valid neighboring positions for A* pathfinding."""
    neighbors = []
    x, y = position
    possible_moves = [(0, -grid_size), (0, grid_size), (-grid_size, 0), (grid_size, 0)]

    for dx, dy in possible_moves:
        neighbor = (x + dx, y + dy)
        if 0 <= neighbor[0] < window_width and 0 <= neighbor[1] < window_height:
            neighbor_rect = pygame.Rect(neighbor[0], neighbor[1], robot_width, robot_height)
            if not any(obstacle.colliderect(neighbor_rect) for obstacle in obstacles):
                neighbors.append(neighbor)
    return neighbors

def a_star_search(start, goal_rect, obstacles, grid_size, previous_collisions, window_width, window_height, robot_width, robot_height):
    """A* algorithm to find the best path from start to goal rectangle."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    goal_center = (goal_rect.centerx, goal_rect.centery)
    f_score = {start: heuristic(start, goal_center)}

    while open_set:
        _, current = heapq.heappop(open_set)

        current_rect = pygame.Rect(current[0], current[1], robot_width, robot_height)
        if goal_rect.colliderect(current_rect):
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid_size, obstacles, window_width, window_height, robot_width, robot_height):
            if neighbor in previous_collisions:
                continue

            tentative_g_score = g_score[current] + 1 
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_center)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return [] 

def reconstruct_path(came_from, current):
    """Reconstruct the path from start to goal after A* has finished."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

class Target:
    def __init__(self, rect, id):
        self.rect = rect
        self.id = id

    def __eq__(self, other):
        return isinstance(other, Target) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"Target(id={self.id}, rect={self.rect})"

class RobotAgent:
    def __init__(self, x, y, color, id, robots, ddpg_agent, obstacles, grid_size, robot_width, robot_height, window_width, window_height, plane_positions):
        self.id = id
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.robots = robots
        self.soc = 100
        self.carrying_cart = False
        self.state_size = 8  # Updated state size to include more context
        self.action_size = 4
        self.memory = self.initialize_memory()
        self.ddpg_agent = ddpg_agent
        self.timestep = 0
        self.target = None
        self.speed = 1.0
        self.path = []
        self.obstacles = obstacles
        self.grid_size = grid_size
        self.robot_width = robot_width
        self.robot_height = robot_height
        self.window_width = window_width
        self.window_height = window_height
        self.plane_positions = plane_positions
        self.previous_collisions = []
        self.exploration_rate = 0.6
        self.task_urgency = 0
        self.direction = 0.0

        self.episode_reward = 0  
        self.episode_rewards = [] 
        self.path_lengths = []

        
        self.state = "to_plane" 

        self.set_initial_target()

    def calculate_distance(self, x1, y1, x2, y2):
        """Calculate the Euclidean distance between two points."""
        return np.linalg.norm([x1 - x2, y1 - y2])

    def initialize_memory(self):
        return {"states": [], "actions": [], "rewards": [], "dones": [], "next_state": []}

    def reset(self):
        """Resets the robot's position, memory, and target after completing a task or encountering an issue."""
        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0
        if len(self.memory['states']) > 0:
            self.path_lengths.append(len(self.memory['states']))
        else:
            self.path_lengths.append(0)

        self.x, self.y = self.start_x, self.start_y
        self.soc = 100
        self.carrying_cart = False
        self.memory = self.initialize_memory()
        self.path = []
        self.previous_collisions = []
        print(f"Robot {self.id} reset.")
        self.state = "to_plane"
        self.set_initial_target()

    def set_initial_target(self):
        """Assign the robot an initial target from available planes."""
        assigned_targets = {
            robot.target for robot in self.robots if robot.target is not None and robot.id != self.id
        }
        available_targets = [plane for plane in self.plane_positions if plane not in assigned_targets]

        if available_targets:
            closest_plane = min(
                available_targets,
                key=lambda plane: np.linalg.norm([self.x - plane.rect.x, self.y - plane.rect.y]),
            )
            self.target = closest_plane
            print(f"Robot {self.id} assigned target: {self.target}")
        else:
            self.target = None
            print(f"Robot {self.id}: No available targets.")

    def update_shared_memory(self, path, success):
        """
        Update the shared memory and pheromone map based on whether the robot's pathfinding was successful or not.
        Args:
        - path: The path the robot took.
        - success: Whether the pathfinding attempt was successful or not.
        """
        global shared_memory, pheromone_map

        for step in path:
            if success:
                # Increase pheromone levels for successful paths
                pheromone_map[step] = min(pheromone_map[step] + 0.1, 1.0)
                shared_memory[step]["collided"] = False
                shared_memory[step]["collided_with"] = None
            else:
                # Decrease pheromone levels and mark collisions
                pheromone_map[step] = max(pheromone_map[step] * 0.9, 0.01)
                shared_memory[step]["collided"] = True
                shared_memory[step]["collided_with"] = self.id

    def move(self):
        """Main method for robot movement including action selection, collision detection, and memory updates."""
        # Check if robot has a target
        if self.target is None:
            print(f"Robot {self.id}: No target assigned.")
            if (self.x, self.y) != (self.start_x, self.start_y):
                if not self.path:
                    self.move_to_charging_station()
                if not self.path:
                    print(f"Robot {self.id}: No path to charging station.")
                    return
            else:
                return
        if len(self.previous_collisions) > 5 and all(
            c == (self.x, self.y) for c in self.previous_collisions[-5:]
        ):
            print(f"Robot {self.id} is stuck, performing random walk.")
            self.exploration_rate = 1.0 
            self.path = [] 
            self.previous_collisions = [] 
            
        if self.target is not None and (not self.path or self.target.rect.collidepoint(self.x, self.y)):
            self.path = self.find_best_path()

        if not self.path:
            print(f"Robot {self.id}: No path available to the target or destination.")
            self.reset()
            return
        
        next_position = self.path.pop(0)
        new_x, new_y = next_position
        
        if self.target is not None:
            target_x = self.target.rect.x
            target_y = self.target.rect.y
        else:
            target_x, target_y = self.start_x, self.start_y

        state = np.reshape(
            [
                self.x,
                self.y,
                self.soc,
                int(self.carrying_cart),
                target_x,
                target_y,
                self.calculate_distance(self.x, self.y, target_x, target_y),
                self.exploration_rate,
            ],
            [1, self.state_size],
        )
        
        action = self.ddpg_agent.select_action(state)

        # Convert continuous DDPG action to grid-based movement (discrete)
        action_mapping = [(0, -self.grid_size), (0, self.grid_size), (-self.grid_size, 0), (self.grid_size, 0)]
        action_index = np.argmax(action)
        action_dx, action_dy = action_mapping[action_index]
        proposed_x = self.x + action_dx
        proposed_y = self.y + action_dy
        
        if (proposed_x, proposed_y) != (new_x, new_y):
            print(f"Robot {self.id}: Adjusting action to follow path.")
            action_dx = new_x - self.x
            action_dy = new_y - self.y

        new_x = self.x + action_dx
        new_y = self.y + action_dy
        
        if 0 <= new_x < self.window_width and 0 <= new_y < self.window_height:
            if self.check_collision(new_x, new_y):
                reward, done = self.calculate_reward(new_x, new_y, collided=True)
                self.update_memory(state, action, reward, done)
                self.update_shared_memory(self.path, success=False)
                self.reset()
            else:
                reward, done = self.calculate_reward(new_x, new_y)
                next_state = np.reshape(
                    [
                        new_x,
                        new_y,
                        self.soc,
                        int(self.carrying_cart),
                        target_x,
                        target_y,
                        self.calculate_distance(new_x, new_y, target_x, target_y),
                        self.exploration_rate,
                    ],
                    [1, self.state_size],
                )
                self.ddpg_agent.store_experience(state, action, reward, next_state, done)

                self.update_memory(state, action, reward, done, next_state)

                if done:
                    if reward > 0:
                        print(f"Robot {self.id} reached the target at ({new_x}, {new_y})!")
                        self.update_shared_memory(self.path, success=True)
                        self.target_reached()
                    else:
                        self.update_shared_memory(self.path, success=False)
                        self.reset()
                else:
                    if self.x != new_x or self.y != new_y:
                        self.soc -= 1
                        if self.soc<=0:
                            self.reset()
                    self.x, self.y = new_x, new_y
                    
                    print(f"Robot {self.id} moved to position: ({self.x}, {self.y})")
        else:
            reward, done = self.calculate_reward(new_x, new_y, out_of_bounds=True)
            self.update_memory(state, action, reward, done)
            self.update_shared_memory(self.path, success=False)
            self.reset()

        self.ddpg_agent.update()

    def find_best_path(self, avoid_positions=None):
        """Find the best path to the target using A* and pheromone-based learning."""
        if self.target is None:
            print(f"Robot {self.id}: No target to find path to.")
            return []
        
        if avoid_positions is None:
            avoid_positions = []
            
        a_star_path = a_star_search(
            (self.x, self.y),
            self.target.rect,
            self.obstacles,
            self.grid_size,
            self.previous_collisions,
            self.window_width,
            self.window_height,
            self.robot_width,
            self.robot_height,
        )

        if a_star_path:
            adjusted_path = []
            for position in a_star_path:
                pheromone_level = pheromone_map[position]
                if pheromone_level < 0.6 or position in avoid_positions: 
                    continue 
                adjusted_path.append(position)
            if adjusted_path:
                return adjusted_path
            else:
                print(f"Robot {self.id}: No viable path found due to low pheromone levels.")
                return a_star_path
        else:
            print(f"Robot {self.id}: A* failed to find a path.")
            return []

    def reconstruct_path(self, came_from, current):
        """Reconstruct the path from start to goal after A* has finished."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def target_reached(self):
        """Handle the logic when a robot successfully reaches its target."""
        if self.state == "to_plane":
            if self.target in self.plane_positions:
                self.plane_positions.remove(self.target)

            self.state = "to_station"
            self.target = None
            print(f"Robot {self.id} has reached the plane. Now returning to charging station.")
            self.move_to_charging_station()
        elif self.state == "to_station":
            self.reset()

    def move_to_charging_station(self):
        """After completing a task, the robot moves to a charging station."""
        charging_station_rect = pygame.Rect(self.start_x, self.start_y, self.robot_width, self.robot_height)
        self.path = a_star_search(
            (self.x, self.y),
            charging_station_rect,
            self.obstacles,
            self.grid_size,
            self.previous_collisions,
            self.window_width,
            self.window_height,
            self.robot_width,
            self.robot_height,
        )
        if self.path:
            print(f"Robot {self.id} is moving to charging station.")
        else:
            print(f"Robot {self.id}: No path to charging station found.")

    def update_memory(self, state, action, reward, done, next_state=None):
        """Store experience in memory for DDPG training."""
        self.memory["states"].append(state)
        self.memory["actions"].append(action)
        self.memory["rewards"].append(reward)
        self.memory["dones"].append(done)
        if next_state is not None:
            self.memory["next_state"].append(next_state)

        self.episode_reward += reward

    def check_collision(self, new_x, new_y):
        """Check if the robot will collide with any obstacles or other robots."""
        robot_rect = pygame.Rect(new_x, new_y, self.robot_width, self.robot_height)
        if any(obstacle.colliderect(robot_rect) for obstacle in self.obstacles):
            return True

        for robot in self.robots:
            if robot.id != self.id:
                other_robot_rect = pygame.Rect(robot.x, robot.y, self.robot_width, self.robot_height)
                if robot_rect.colliderect(other_robot_rect):
                    return True

        return False

    def calculate_reward(self, new_x, new_y, collided=False, out_of_bounds=False):
        """Calculate the reward for the current action."""
        if self.target is not None:
            target_rect = self.target.rect
            dist_before = np.linalg.norm([self.x - target_rect.centerx, self.y - target_rect.centery])
            dist_after = np.linalg.norm([new_x - target_rect.centerx, new_y - target_rect.centery])
        else:
            dist_before = np.linalg.norm([self.x - self.start_x, self.y - self.start_y])
            dist_after = np.linalg.norm([new_x - self.start_x, new_y - self.start_y])

        if out_of_bounds:
            return -1000, True
        elif collided:
            return -700, True
        elif self.target is not None and target_rect.collidepoint(new_x, new_y):
            return 500, True
        elif self.target is None and (new_x, new_y) == (self.start_x, self.start_y):
            return 500, True
        else:
            reward = -1 + 20 * (dist_before - dist_after)
            return reward, False
        
    def coordinate_with_other_robots(self):
        """Coordinate with other robots to avoid collisions and share tasks."""
        if self.target is None:
            self.reassign_target()
    
        robots_sorted_by_proximity = sorted(self.robots, key=lambda r: np.linalg.norm([r.x - self.target.rect.x, r.y - self.target.rect.y]))
    
        for robot in robots_sorted_by_proximity:
            if robot.id != self.id and self.is_near(robot):
                if self.has_higher_priority(robot):
                    print(f"Robot {self.id} has priority, requesting Robot {robot.id} to adjust.")
                    robot.adjust_path_to_avoid(self)
                else:
                    print(f"Robot {self.id} adjusting to avoid collision with Robot {robot.id}")
                    self.adjust_path_to_avoid(robot)
                    
    def is_near(self, other_robot):
        """Check if another robot is near enough to potentially collide, considering speed."""
        distance = np.linalg.norm([self.x - other_robot.x, self.y - other_robot.y])
        speed_factor = max(self.speed, other_robot.speed)
        proximity_threshold = self.grid_size * 2 + speed_factor * 0.5 
        return distance < proximity_threshold
    
    def future_positions(self, steps=6):
        '''Predict future position'''
        future_positions = []
        for i in range(1, steps + 1):
            future_x = self.x + i * self.speed * np.cos(self.direction)
            future_y = self.y + i * self.speed * np.sin(self.direction)
            future_positions.append((future_x, future_y))
        return future_positions
    
    def adjust_path_to_avoid(self, other_robot):
        """Adjust the robot's path to avoid collisions with other robots."""
        self.previous_collisions.append((self.x, self.y))
    
        shared_memory[(self.x, self.y)] = {'collided': True, 'collided_with': other_robot.id}
    
        self.path = self.find_best_path(avoid_positions=[(other_robot.x, other_robot.y)] + other_robot.future_positions())
    
        if not self.path:
            print(f"Robot {self.id} slowing down to avoid Robot {other_robot.id}.")
            self.speed *= 0.5

    def has_higher_priority(self, other_robot):
        """Determine if the current robot has higher priority to maintain its path."""
        if self.target is None:
            print(f"Robot {self.id} has no target, assigning lower priority.")
            return False
        
        if other_robot.target is None:
            print(f"Other robot {other_robot.id} has no target, assigning higher priority.")
            return True 
        
        if not hasattr(self.target, 'rect'):
            print(f"Robot {self.id} has invalid target structure, assigning lower priority.")
            return False
    
        if not hasattr(other_robot.target, 'rect'):
            print(f"Other robot {other_robot.id} has invalid target structure, assigning higher priority.")
            return True
        
        distance_to_target = np.linalg.norm([self.x - self.target.rect.x, self.y - self.target.rect.y])
        other_distance_to_target = np.linalg.norm([other_robot.x - other_robot.target.rect.x, other_robot.y - other_robot.target.rect.y])
    
        return distance_to_target < other_distance_to_target or self.task_urgency > other_robot.task_urgency

    def find_closest_plane(self, available_targets):
        """Find the closest plane to the robot's current position."""
        if not available_targets:
            return None
        
        closest_plane = min(
            available_targets,
            key=lambda plane: np.linalg.norm([self.x - plane.rect.x, self.y - plane.rect.y])
        )
        return closest_plane

    def reassign_target(self):
        """Reassign the robot's target if necessary."""
        assigned_targets = {
            robot.target for robot in self.robots if robot.target is not None and robot.id != self.id
        }
        
        available_targets = [plane for plane in self.plane_positions if plane not in assigned_targets]
    
        closest_plane = self.find_closest_plane(available_targets)
    
        if closest_plane is None:
            shared_memory.clear()
            pheromone_map.clear()
            self.plane_positions = get_initial_plane_positions()
            available_targets = [plane for plane in self.plane_positions if plane not in assigned_targets]
            closest_plane = self.find_closest_plane(available_targets)
        
        if closest_plane is not None:
            self.target = closest_plane
            self.state = "to_plane"  # Ensure state is set correctly
            print(f"Robot {self.id} reassigned target to {self.target}")
        else:
            print(f"Robot {self.id}: No available targets to reassign.")

