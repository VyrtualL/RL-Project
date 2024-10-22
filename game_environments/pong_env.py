import gymnasium as gym

from game_environments.game_env import GameEnv

class PongEnv(GameEnv):
    def __init__(self, mode: int = 0, difficulty: int = 0, frameskip: int = 4, render_fps: int = 60):
        super().__init__(gym.make("ALE/Pong-v5", render_mode="rgb_array", mode=mode, difficulty=difficulty, frameskip=frameskip))
        self.env.metadata['render_fps'] = render_fps
        self.adv_points = 0
        self.points = 0

    def reset(self):
        res = super().reset()
        self.adv_points = 0
        self.points = 0
        return res

    def step(self, action: int):
        state, rec, done, unk, info = super().step(action)
        if rec < 0:
            self.adv_points += 1
            print(f"Point loss: {self.adv_points} - {self.points}")
        elif rec > 0:
            self.points += 1
            print(f"Point gain: {self.adv_points} - {self.points}")
        return state, rec, done, unk, info
