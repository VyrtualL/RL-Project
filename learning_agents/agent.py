import typing as t
import random

class Agent:
    def __init__(self, legal_actions: t.List[int]):
        self.legal_actions = legal_actions

    def update(self, previous_state, action, reward, new_state, new_is_terminal):
        pass

    def get_best_action(self, state):
        return random.choice(self.legal_actions)

    def get_action(self, state):
        return self.get_best_action(state)