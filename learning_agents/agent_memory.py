import random
import numpy as np
import torch

class MemoryBuffer:
    def __init__(self, size):
        self.max_size = size
        self.memory = []
        self.frames = [[]]
        self.memorized_steps = []
        self.step = 0

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((self.step, action, reward, done))
        self.step += 1
        self.memory = self.memory[-self.max_size:]

        if self.frames[-1] == []:
            if len(self.memorized_steps) == 0:
                curr_game = 0
            else:
                curr_game = self.memorized_steps[-1][0] + 1
            nb_step = 0
            for _ in range(4):
                self.frames[-1].append(state)
        else:
            nb_step = self.memorized_steps[-1][1] + 1
            curr_game = self.memorized_steps[-1][0]

        self.frames[-1].append(next_state)
        self.memorized_steps.append((curr_game, nb_step))

        start_game, start_step = self.memorized_steps[0]
        self.memorized_steps = self.memorized_steps[-self.max_size:]

        if start_game != self.memorized_steps[0][0]:
            self.frames = self.frames[self.memorized_steps[0][0] - start_game:]
        else:
            self.frames[0] = self.frames[0][self.memorized_steps[0][1] - start_step:]

        if done:
            self.frames.append([])

    def get_batch(self, batch_size):
        if len(self.memory) < batch_size:
            batch = self.memory
        else:
            batch = random.sample(self.memory, k=batch_size)

        prev_states, actions, rewards, next_states, dones = [], [], [], [], []
        start_step = self.memory[0][0]

        for step, action, reward, done in batch:
            pos = step - start_step
            sequence_pos = self.memorized_steps[pos]
            game_index = sequence_pos[0] - self.memorized_steps[0][0]
            if game_index == 0:
                game_step = sequence_pos[1] - self.memorized_steps[0][1]
            else:
                game_step = sequence_pos[1]

            frames = self.frames[game_index][game_step:game_step + 5]
            frame_tensors = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
            start_sequence = torch.stack(frame_tensors[:4], axis=0)
            end_sequence = torch.stack(frame_tensors[1:], axis=0)

            prev_states.append(start_sequence)
            actions.append(action)
            rewards.append(reward)
            next_states.append(end_sequence)
            dones.append(done)

        return prev_states, torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), next_states, torch.tensor(dones, dtype=torch.bool)

    def __len__(self):
        return len(self.memory)
