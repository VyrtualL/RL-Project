from learning_agents.agent import Agent

import typing as t
import random

class TestingEpsGreedy(Agent):
    def __init__(self, legal_actions: t.List[int], agent: Agent, epsilon=0.05):
        self.legal_actions = legal_actions
        self.agent = agent
        self.epsilon = epsilon

    def update(self, previous_state, action, reward, new_state, new_is_terminal):
        pass

    def get_best_action(self, state):
        return self.agent.get_best_action(state)

    def get_action(self, state):
        action = self.agent.get_best_action(state)
        if random.uniform(0.0, 1.0) < self.epsilon:
            action = random.choice(self.legal_actions)
        return action
