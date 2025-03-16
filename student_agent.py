# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle  # Load Q-table 
def get_state(obs):
    station = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])] 
    station1_rel = (station[0][0]-obs[0], station[0][1]-obs[1])
    station2_rel = (station[1][0]-obs[0], station[1][1]-obs[1])
    station3_rel = (station[2][0]-obs[0], station[2][1]-obs[1])
    station4_rel = (station[3][0]-obs[0], station[3][1]-obs[1])
    return (station1_rel, station2_rel, station3_rel, station4_rel, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])

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
def get_action(obs):
    return agent.get_action(get_state(obs))

