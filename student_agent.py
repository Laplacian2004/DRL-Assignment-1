# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle  # Load Q-table 
def distance(pos_1, pos_2):
    return abs(pos_1[0] - pos_2[0]) + abs(pos_1[1] - pos_2[1])

def get_state(obs, station, target_index, passenger_picken):
    target_rel = (station[target_index][0] - obs[0], station[target_index][1] - obs[1])
    if not passenger_picken:
        target_look = obs[14]
    else:
        target_look = obs[15]
    return (target_rel, passenger_picken, obs[10], obs[11], obs[12], obs[13], target_look)

class StudentAgent:
    def __init__(self, q_table_path="q_table.pkl"):
        """Load the pre-trained Q-table."""
        with open(q_table_path, "rb") as f:
            self.q_table = pickle.load(f)
        self.action_size = 6  # Number of actions

    def get_action(self, state):
        """Select action based on Q-table."""
        if state not in self.q_table:
            return np.random.choice(range(self.action_size))  # If unseen state, choose random action
        return np.argmax(self.q_table[state])  # Select the best action

# Usage example
agent = StudentAgent()
passenger_picken = 0
target_look = 0
target_index = 0
passenger_rec = None
def get_action(obs):
    global target_index, passenger_picken, passenger_rec, target_look 
    station = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
    agent_pos = (obs[0], obs[1])
    if not passenger_picken:
        target_look = obs[14]
    else:
        target_look = obs[15] 
    # print(f"target_index = {target_index}")
    # print(f"target_look = {target_look}")
    # print(f"passenger_picken = {passenger_picken}")
    target = (station[target_index][0]-agent_pos[0], station[target_index][1]-agent_pos[1])
    
    target_dist = distance(station[target_index], agent_pos)
    # print(f"target_dist = {target_dist}")
    if target_dist <= 1 and not target_look:
        target_index += 1
        if passenger_picken and target_index == passenger_rec:
            target_index +=1
        if target_index >3:
            target_index = 0
    state = get_state(obs, station, target_index, passenger_picken)
    target_look = state[6]
    action = agent.get_action(state)
    # print(f"action = {action}")
    if target_dist == 0:
        if action == 4 and not passenger_picken:
            passenger_picken = 1
            passenger_rec = target_index
        if action == 5 and passenger_picken:
            passenger_picken = 0
    return action
