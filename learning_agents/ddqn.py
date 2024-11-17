import typing as t
from skimage.transform import resize
import numpy as np
import torch
from torch.nn import Module, Conv2d, Linear, ReLU, MSELoss
import random
import math

from learning_agents.agent import Agent
from learning_agents.agent_memory import MemoryBuffer

class DQLModel(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.first_conv = Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.second_conv = Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.hidden_linear = Linear(in_features=32*9*9, out_features=256)
        self.output_linear = Linear(in_features=256, out_features=output_size)
        self.relu = ReLU()

    def forward(self, x):
        res = self.first_conv(x)
        res = self.relu(res)
        res = self.second_conv(res)
        res = self.relu(res)
        res = res.flatten(start_dim=1)
        res = self.hidden_linear(res)
        res = self.relu(res)
        res = self.output_linear(res)
        return res


class DDQN(Agent):
    def __init__(self, legal_actions: t.List[int], learning_rate: float = 1e-4, gamma: float = 0.99, start_epsilon: float = 1, end_epsilon: float = 0.1, epsilon_steps : int = 1000000, memory: int = 1000000, mini_batch_size: int = 32, tau: float = 0.005):
        super().__init__(legal_actions)

        if torch.cuda.is_available():
            self.device = f"cuda:{torch.cuda.current_device()}"
            device_name = torch.cuda.get_device_name(self.device)
            print(f"Using cuda device {device_name}")
        else:
            self.device = "cpu"
            print("Using cpu")

        self.eps_start = start_epsilon
        self.eps_end = end_epsilon
        self.eps_decay = epsilon_steps
        self.context = []
        self.tau = tau
        self.batch_size = mini_batch_size
        self.gamma = gamma

        n_actions = len(legal_actions)
        self.model = DQLModel(n_actions).to(self.device)
        self.target_model = DQLModel(n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        self.loss = MSELoss()

        self.step = 0
        self.memory = MemoryBuffer(memory)
        self.frame_added = False

    def reset_context(self):
        self.context = []
        self.frame_added = False

    def preprocess(self, image):
        gray_image = np.dot(image, [0.299, 0.587, 0.114])
        resized_image = resize(gray_image, (110, 84))
        x_off = (110 - 84) // 2
        gen_off = 5
        cropped_image = resized_image[x_off + gen_off:-x_off + gen_off, :]
        return cropped_image

    def add_to_context(self, state):
        image = self.preprocess(state)
        tensor_image = torch.tensor(image, dtype=torch.float32)
        if self.context == []:
            for _ in range(4):
                self.context.append(tensor_image)
        else:
            self.context.append(tensor_image)
        self.context = self.context[-4:]
        self.frame_added = True

    def get_context(self):
        curr_context = torch.stack(self.context, axis=0)
        return curr_context.unsqueeze(0).to(self.device)

    def get_epsilon(self):
        if self.eps_decay <= self.step:
            return self.eps_end
        return 1 - self.step * (self.eps_start - self.eps_end) / self.eps_decay

    def get_best_action(self, state):
        if not self.frame_added:
            self.add_to_context(state)

        curr_context = self.get_context()
        with torch.no_grad():
            action = self.model(curr_context).max(1).indices.view(1, 1)

        self.frame_added = False
        return action

    def get_action(self, state):
        if not self.frame_added:
            self.add_to_context(state)

        if random.uniform(0.0, 1.0) < self.get_epsilon():
            action = random.choice(self.legal_actions)
        else:
            action = self.get_best_action(state)

        self.step += 1
        self.frame_added = False
        return action

    def update(self, previous_state, action, reward, new_state, new_is_terminal):
        self.memory.add_to_memory(self.preprocess(previous_state), action, reward, self.preprocess(new_state), new_is_terminal)

        if len(self.memory) < self.batch_size:
            return

        # Get batch for update
        starts_b, actions_b, rewards_b, ends_b, dones_b = self.memory.get_batch(self.batch_size)
        starts, actions, rewards, ends, dones = torch.stack(starts_b).to(self.device), actions_b.to(self.device), rewards_b.to(self.device), torch.stack(ends_b).to(self.device), dones_b.to(self.device)

        # Get qvalues of actions
        indices = np.arange(self.batch_size)
        pred_values = self.model(starts)[indices, actions]

        # Get next states values
        with torch.no_grad():
            next_state_values = self.model(ends).max(1).values
            next_state_values[dones] = 0.0

        # Get target qvalues
        expected_values = rewards + self.gamma * next_state_values

        # Update model weights
        self.optimizer.zero_grad()
        loss = self.loss(pred_values, expected_values)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # Update evaluation model
        if self.step % self.tau == 0:
            self.target_model.load_state_dict(self.model.state_dict())
