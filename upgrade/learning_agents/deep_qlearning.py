import typing as t
from skimage.transform import resize
import numpy as np
import torch
from torch.nn import Module, Conv2d, Linear
import random

from learning_agents.agent import Agent


class DQLModel(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.first_conv = Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.second_conv = Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.hidden_linear = Linear(in_features=32*9*9, out_features=256)
        self.output_linear = Linear(in_features=256, out_features=output_size)

    def forward(self, x):
        res = self.first_conv(x)
        res = torch.clamp(torch.relu(res), min=10, max=18)
        res = self.second_conv(res)
        res = torch.relu(res)
        res = res.flatten()
        res = self.hidden_linear(res)
        res = torch.relu(res)
        res = self.output_linear(res)
        return res


class DeepQLearning(Agent):
    def __init__(self, legal_actions: t.List[int], learning_rate: float = 0.001, gamma: float = 0.99, start_epsilon: float = 1, end_epsilon: float = 0.1, epsilon_steps : int = 1000000, memory: int = 1000000, mini_batch_size: int = 32):
        super().__init__(legal_actions)
        self.context = []
        self.frame_added = False
        self.q_value_model = DQLModel(output_size=len(legal_actions))
        self.gamma = gamma
        self.epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilons = np.linspace(start_epsilon, end_epsilon, epsilon_steps)
        self.N = memory
        self.mini_batch_size = mini_batch_size
        self.D = []
        self.learning_rate = learning_rate

        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
            device_name = torch.cuda.get_device_name(self.device)
            print(f"Using cuda device {device_name}")
        else:
            self.device = "cpu"
            print("Using cpu")
        self.curr_model = self.q_value_model.to(self.device)

        self.optimizer = torch.optim.Adam(self.q_value_model.parameters(), lr=learning_rate)
        self.step = 0
        self.training_stop = False
        self.prev = None
        self.device = None

    def set_device(self):
        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
            device_name = torch.cuda.get_device_name(self.device)
            print(f"Using cuda device {device_name}")
        else:
            self.device = "cpu"
            print("Using cpu")
        self.curr_model = self.q_value_model.to(self.device)

    def preprocess(self, image):
        gray_image = np.dot(image, [0.299, 0.587, 0.114])
        resized_image = resize(gray_image, (110, 84))
        x_off = (110 - 84) // 2
        gen_off = 5
        cropped_image = resized_image[x_off + gen_off:-x_off + gen_off, :]

        return cropped_image

    def add_to_context(self, state):
        self.context.append(self.preprocess(state))
        self.context = self.context[-4:]
        self.frame_added = True

    def reset_context(self):
        self.context = []
        self.frame_added = False

    def get_sequence(self):
        nb_context_frames = len(self.context)
        frames = []
        if nb_context_frames < 4:
            for _ in range(4 - nb_context_frames):
                frames.append(self.context[0].copy())
            for i in range(nb_context_frames):
                frames.append(self.context[i])
        else:
            frames = self.context
        return np.stack(frames, axis=0)

    def get_q_values(self, sequence):
        t = torch.tensor(sequence, dtype=torch.float32, requires_grad=True)
        t_device = t.to(self.device)
        if torch.cuda.is_available():
            t_device = t_device.cuda()
        return self.curr_model.forward(t_device)#q_value_model.forward(t_device)

    def get_q_value(self, sequence, action):
        return self.get_q_values(sequence)[action]

    def get_value(self, sequence):
        return max(self.get_q_values(sequence))

    def add_to_memory(self, memory_update):
        self.D.append(memory_update)
        self.D = self.D[-self.N:]

    def update(self, previous_state, action, reward, new_state, new_is_terminal):
        curr_phi = self.get_sequence()
        self.add_to_context(new_state)
        new_phi = self.get_sequence()
        self.add_to_memory((curr_phi, action, reward, new_phi, new_is_terminal))

        if len(self.D) < self.mini_batch_size:
            batch = self.D
        else:
            batch = random.sample(self.D, k=self.mini_batch_size)

        self.optimizer.zero_grad()
        for (start_j, action_j, reward_j, next_j, is_terminal_j) in batch:
            if is_terminal_j:
                y_j = reward_j
            else:
                y_j = reward_j + self.gamma * self.get_value(next_j)
            adjust = (y_j - self.get_q_value(start_j, action_j))**2
            adjust.backward()
        self.optimizer.step()

    def get_best_action(self, state):
        if not self.frame_added:
            self.add_to_context(state)

        sequence = self.get_sequence()
        q_values = self.get_q_values(sequence)
        action = torch.argmax(q_values).item()
        self.frame_added = False

        #print(q_values)
        return action

    def get_action(self, state):
        if not self.frame_added:
            self.add_to_context(state)

        rand = random.uniform(0.0, 1.0)
        if rand < self.epsilon:
            action = random.choice(self.legal_actions)
        else:
            action = self.get_best_action(state)

        self.step += 1
        if self.epsilon > self.end_epsilon:
            self.epsilon = self.epsilons[self.step]
            #print(self.epsilon)

        self.frame_added = False
        return action
