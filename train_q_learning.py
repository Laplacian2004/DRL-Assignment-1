import numpy as np
import random
import pickle  # For saving/loading Q-table
import gym  # Import the Taxi environment
from simple_custom_taxi_env import SimpleTaxiEnv  # Import the Taxi environment
import matplotlib.pyplot as plt

def distance(pos_1, pos_2):
    return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

def get_near_station_dir(station, agent_pos):
    nearby_stations = [station[i] for i in range(4) if distance(station[i], agent_pos) <= 1]

    if not nearby_stations:
        return None  # No nearby station found

    chosen_station = random.choice(nearby_stations)
    return (chosen_station[0] - agent_pos[0], chosen_station[1] - agent_pos[1])  # Direction vector
def get_state(obs, station):
    station1_rel = (station[0][0]-obs[0], station[0][1]-obs[1])
    station2_rel = (station[1][0]-obs[0], station[1][1]-obs[1])
    station3_rel = (station[2][0]-obs[0], station[2][1]-obs[1])
    station4_rel = (station[3][0]-obs[0], station[3][1]-obs[1])
    return (station1_rel, station2_rel, station3_rel, station4_rel, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])
class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99995, min_epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.epsilon_decay = epsilon_decay  # How fast epsilon decreases
        self.min_epsilon = min_epsilon  # Minimum exploration rate
        self.action_size = action_size  # Number of actions
        self.q_table = {}  # Q-table stored as a dictionary

    def get_action(self, state):
        """Select action using epsilon-greedy strategy."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state, done):
        """Perform Q-learning update step."""
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    
def train_agent(num_episodes=100000, max_steps=500):
    """Train the Q-learning agent and save the Q-table."""
    agent = QLearningAgent(action_size=6)
    past_reward = []
    hit_wall = []
    for episode in range(num_episodes):
        # 🔹 Randomize grid size for this episode
        grid_size = random.randint(5, 10)
        env_config = {"grid_size": grid_size, "fuel_limit": 5000}

        # 🔹 Recreate the environment with new grid size
        env = SimpleTaxiEnv(**env_config)
        obs, _ = env.reset() 
        station = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])] 
        state = get_state(obs, station)
        total_reward = 0
        done = False
        step_count = 0
        prev_passenger_look = False
        destination_rec = None
        pickup_passenger = False
        hit_wall =0 
        while not done and step_count < max_steps:
            action = agent.get_action(state)
            obs, reward, done, _ = env.step(action)
            # Reward Shaping
            passenger_look, destination_look = obs[14], obs[15]
            agent_pos = (obs[0], obs[1])
            # action : south 0, north 1, east 2, west 3
            if (obs[10] and action == 1) or (obs[11] and action == 0) or (obs[12] and action == 2) or (obs[13] and action == 3):
                reward -= 1000
                hit_wall += 1
            if passenger_look :
                direction = get_near_station_dir(station, agent_pos)
                if direction == (0, 1) and action == 1:
                    reward += 50
                elif direction == (0, -1) and action == 0:
                    reward += 50 
                elif direction == (1, 0) and action == 2:
                    reward += 50
                elif direction == (-1, 0) and action == 3:
                    reward += 50
                elif direction == (0, 0) and action == 4:
                    pickup_passenger = True
                    reward += 100
                else:
                    reward -= 150 
            elif destination_look and pickup_passenger:
                direction = get_near_station_dir(station, agent_pos)
                destination_rec = (agent_pos[0] + direction[0], agent_pos[1] + direction[1])
                if direction == (0, 1) and action == 1:
                    reward +=50
                elif direction == (0, -1) and action == 0:
                    reward += 50 
                elif direction == (1, 0) and action == 2:
                    reward += 50
                elif direction == (-1, 0) and action == 3:
                    reward += 50
                elif direction == (0, 0) and action == 5:
                    reward += 100
                else:
                    reward -= 150
            else: 
                if action > 3:
                    reward -= 100
            if not done:
                reward -=1
            next_state = get_state(obs, station)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
        past_reward.append(total_reward)
        if episode % 1000 == 0:
            print(f"Episode {episode}: Grid Size = {grid_size}x{grid_size}, average_reward = {np.average(past_reward[-1000:])}, Epsilon = {agent.epsilon:.4f}")
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)  # Reduce exploration
        

    # Save trained Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Training complete! Q-table saved as q_table.pkl")
     
    plt.plot(past_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-learning Training Progress")
    plt.show()

if __name__ == "__main__":
    train_agent()

