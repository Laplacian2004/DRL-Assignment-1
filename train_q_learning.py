import numpy as np
import random
import pickle  # For saving/loading Q-table
import gym  # Import the Taxi environment
from simple_custom_taxi_env import SimpleTaxiEnv  # Import the Taxi environment
import matplotlib.pyplot as plt

def distance(pos_1, pos_2):
    return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

def get_station_index(station, agent_pos):
    nearby_stations = []
    for i in range(4):
        if distance(station[i], agent_pos)<=1:
            nearby_stations.append(i)
    if not nearby_stations:
        return None  # No nearby station found

    chosen_station = random.choice(nearby_stations)
    return chosen_station  # Direction vector

def get_state(obs, station, target_index, passenger_picken):
    target_rel = (station[target_index][0] - obs[0], station[target_index][1] - obs[1])
    if not passenger_picken:
        target_look = obs[14]
    else:
        target_look = obs[15]
    return (target_rel, passenger_picken, obs[10], obs[11], obs[12], obs[13], target_look)

class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99995, min_epsilon=0.01):
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
            if state[1] :
                return np.random.choice([0, 1, 2, 3, 5])
            else:
                return np.random.choice([0, 1, 2, 3, 4])
        return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state, done):
        """Perform Q-learning update step."""
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    
def train_agent(num_episodes=100000, max_steps=1000):
    """Train the Q-learning agent and save the Q-table."""
    agent = QLearningAgent(action_size=6)
    past_reward = []
    past_hit_wall = []
    for episode in range(num_episodes):
        # 🔹 Randomize grid size for this episode
        # print(f"episode = {episode}")
        grid_size = random.randint(5, 10)
        env_config = {"grid_size": grid_size, "fuel_limit": 5000}

        # 🔹 Recreate the environment with new grid size
        env = SimpleTaxiEnv(**env_config)
        obs, _ = env.reset() 
        station = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        '''
        for i in range(4):
            print(station[i])
        '''
        agent_pos = (obs[0], obs[1])
        passenger_picken = 0
        target_index = 0 
        state = get_state(obs, station, target_index, passenger_picken)
        total_reward = 0
        done = False
        step_count = 0
        prev_passenger_look = obs[14]
        prev_destination_look = obs[15]
        passenger_rec = None
        hit_wall_rate = 0 
        target_look = obs[14]
        prev_target_dist = distance(station[target_index], agent_pos)
        while not done and step_count < max_steps:
            prev_obstacle = (obs[10], obs[11], obs[12], obs[13])
            action = agent.get_action(state)
            # print(f"action = {action}")
            obs, reward, done, _ = env.step(action)
            # Reward Shaping
            passenger_look, destination_look = obs[14], obs[15]
            agent_pos = (obs[0], obs[1])
            # print(f"agent_pos = {agent_pos}, passenger_look = {passenger_look}, dest_look = {destination_look} station_check = {station_check}, passenger_picken ={passenger_picken}")

            target_dist = distance(station[target_index], agent_pos)
            next_state = get_state(obs, station, target_index, passenger_picken)
            # action : south 0, north 1, east 2, west 3
            # obstacles collison 
            if (prev_obstacle[0] and action == 1) or (prev_obstacle[1] and action == 0) or\
                (prev_obstacle[2] and action == 2) or (prev_obstacle[3] and action == 3):
                reward -= 1000.0
                hit_wall_rate +=1
            
            if target_dist < prev_target_dist:
                reward += 10.0 
            else :
                reward -= 10.0
            
            if target_dist == 0 and target_look:
                if passenger_look and not passenger_picken and action == 4:
                    passenger_picken = 1 
                    passenger_rec = target_index
                    reward += 100 
                elif destination_look and passenger_picken and action == 5:
                    reward += 100
                else :
                    reward -= 200
            if target_dist == 0 and action <3:
                reward += 100.0
            if target_dist!=0 and action >3:
                reward -= 1000.0
            if action == 5:
                passenger_picken = 0 

            agent.update_q_table(state, action, reward, next_state, done)

            if target_dist <= 1 and not target_look:
                target_index += 1
                if passenger_picken and target_index == passenger_rec:
                    target_index +=1
                if target_index >3:
                    target_index = 0

            next_state = get_state(obs, station, target_index, passenger_picken)

            prev_destination_look = destination_look
            prev_passenger_look = passenger_look
            prev_target_dist = target_dist
            state = next_state
            target_look = state[6]

            total_reward += reward
            step_count += 1

        past_reward.append(total_reward)
        past_hit_wall.append(hit_wall_rate)
        if episode % 1000 == 0:
            print(f"Episode {episode}: Grid Size = {grid_size}x{grid_size}, average_reward = {np.average(past_reward[-1000:])}, Epsilon = {agent.epsilon:.4f}")
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)  # Reduce exploration
        

    # Save trained Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)
    print("Training complete! Q-table saved as q_table.pkl")
     
    plt.plot(past_hit_wall)
    plt.xlabel("Episodes")
    plt.ylabel("Hit wall")
    plt.title("Q-learning Training Progress")
    plt.show()

if __name__ == "__main__":
    train_agent()

