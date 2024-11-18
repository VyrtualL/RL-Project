import typing as t
from skimage.transform import resize
import numpy as np
import torch
from torch.nn import Module, Conv2d, Linear, ReLU, MSELoss
import random
import math

from learning_agents.agent import Agent
from learning_agents.agent_memory import MemoryBuffer

#create the noisy linear layer
class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        #self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
            
    #def reset_noise(self):
    #    epsilon_in = self.scale_noise(self.in_features)
    #    epsilon_out = self.scale_noise(self.out_features)
    #    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    #    self.bias_epsilon.copy_(epsilon_out)
            
    def forward(self, input):
        weight_epsilon = torch.empty_like(self.weight_epsilon).normal_()
        bias_epsilon = torch.empty_like(self.bias_epsilon).normal_()

        weight = self.weight_mu + self.weight_sigma * weight_epsilon
        bias = self.bias_mu + self.bias_sigma * bias_epsilon
        return torch.nn.functional.linear(input, weight, bias)



#We implement Dueling DQL model, we have to decompose Q value to "state value" and "action advantage"
#And after, we add V and A with the formula 
class DuelingDQLModel(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.first_conv = Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.second_conv = Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.hidden_linear = NoisyLinear(in_features=32*9*9, out_features=256)
        self.relu = ReLU()
        self.value_stream = NoisyLinear(in_features=256, out_features=1)
        self.advantage_stream = NoisyLinear(in_features=256, out_features=output_size)

    def forward(self, x):
        res = self.first_conv(x)
        res = self.relu(res)
        res = self.second_conv(res)
        res = self.relu(res)
        res = res.flatten(start_dim=1)
        res = self.hidden_linear(res)
        res = self.relu(res)
        #V
        value = self.value_stream(res)
        #A(s, a)
        advantage = self.advantage_stream(res)
        res = value + (advantage - advantage.mean(dim=1, keepdim=True))
        #res = torch.softmax(res, dim=2)
        return res


class DDQN(Agent):
    def __init__(self, legal_actions: t.List[int], learning_rate: float = 1e-5, gamma: float = 0.99, start_epsilon: float = 1, end_epsilon: float = 0.1, epsilon_steps : int = 1000000, memory: int = 1000000, mini_batch_size: int = 32, tau: float = 0.005):
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
        self.model = DuelingDQLModel(n_actions).to(self.device)
        self.target_model = DuelingDQLModel(n_actions).to(self.device)
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
            #q_distri = self.model(curr_context)
            #q_values = (q_distri * self.support).sum(dim=2)
            #action = q_values.argmax(dim=1).view(1, 1)
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
        torch.autograd.set_detect_anomaly(True)
        
        self.memory.add_to_memory(self.preprocess(previous_state), action, reward, self.preprocess(new_state), new_is_terminal)

        if len(self.memory) < self.batch_size:
            return

        # Get batch for update
        starts_b, actions_b, rewards_b, ends_b, dones_b = self.memory.get_batch(self.batch_size)
        starts, actions, rewards, ends, dones = torch.stack(starts_b).to(self.device), actions_b.to(self.device), rewards_b.to(self.device), torch.stack(ends_b).to(self.device), dones_b.to(self.device)

        # Get qvalues of actions
        indices = np.arange(self.batch_size)
        pred_values = self.model(starts)[indices, actions]
        _ = self.model(starts)

        # Get next states values
        with torch.no_grad():
            _ = self.target_model(ends)
            next_state_values = self.model(ends).max(1).values.clone()
            next_state_values[dones] = 0.0
            #next_dist = self.target_model(ends)
            #next_q_values = (next_dist * self.support).sum(dim=2)
            #next_actions = next_q_values.argmax(dim=1)
            #next_dist = next_dist[torch.arange(self.batch_size), next_actions]
            #target_dist = self.project_distribution(next_dist, rewards, dones)

        # Get target qvalues
        expected_values = rewards + self.gamma * next_state_values
        
        #current_dist = self.model(starts)
        #current_dist = current_dist[torch.arange(self.batch_size), actions]
        #loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1).mean()

        # Update model weights
        self.optimizer.zero_grad()
        loss = self.loss(pred_values, expected_values)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # Update evaluation model
        model_dict = self.model.state_dict()
        target_dict = self.target_model.state_dict()
        for key in target_dict:
            target_dict[key] = model_dict[key] * self.tau + target_dict[key] * (1 - self.tau)
        self.target_model.load_state_dict(target_dict)
