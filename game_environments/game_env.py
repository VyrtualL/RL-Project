import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from learning_agents.agent import Agent

class GameEnv:
    def __init__(self, env: gym.Env):
        self.env = env
        self.num_actions : int = self.env.action_space.n
        self.renders = []
        self.total_reward = 0

    def reset(self):
        start_vals = self.env.reset()
        self.current_state = start_vals[0]
        self.renders = [self.env.render()]
        self.total_reward = 0
        return start_vals

    def step(self, action: int):
        state, rec, done, unk, info = self.env.step(action)
        self.total_reward += rec
        self.renders.append(self.env.render())
        return state, rec, done, unk, info

    def train_agent(self, agent: Agent, max_iter: int | None = int(1e5)):
        state, _ = self.reset()
        if max_iter is None:
            while True:
                action = agent.get_action(state)
                new_state, reward, done, _, _ = self.step(action)
                agent.update(state, action, reward, new_state, done)
                state = new_state
                if done:
                    return self.total_reward
        for _ in range(max_iter):
            action = agent.get_action(state)
            new_state, reward, done, _, _ = self.step(action)
            agent.update(state, action, reward, new_state, done)
            state = new_state
            if done:
                break
        return self.total_reward

    def use_agent(self, agent: Agent, max_iter: int | None = int(1e5)):
        state, _ = self.reset()
        if max_iter is None:
            while True:
                action = agent.get_best_action(state)
                new_state, reward, done, _, _ = self.step(action)
                state = new_state
                if done:
                    return self.total_reward
        for _ in range(max_iter):
            action = agent.get_best_action(state)
            new_state, reward, done, _, _ = self.step(action)
            state = new_state
            if done:
                break
        return self.total_reward

    def animate(self, save: bool = False, path: str = ""):
        fig = plt.figure()
        plt.axis('off')
        anim_renders = [[plt.imshow(render, animated=True)] for render in self.renders]
        anim = animation.ArtistAnimation(fig, anim_renders)
        if save:
            anim.save(path)
        else:
            plt.show()
        plt.close()
