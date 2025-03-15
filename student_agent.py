# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle  # Load Q-table
def obs_to_state(obs):
    return obs

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
    return agent.get_action(obs_to_state(obs))

