from collections import namedtuple
import random
import math
import gym
import torch
import torchvision.transforms as T
import numpy as np


Experience = namedtuple(
    "Experience",
    ("state", "action", "next_state", "reward")
)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience

        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        exp = math.exp(-1. * current_step * self.decay)
        return self.end + (self.start - self.end) * exp


class EnvironmentManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make("Breakout-v0").unwrapped
        self.env = gym.wrappers.Monitor(self.env, 'tmp/', force=True)
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    @property
    def is_start(self):
        return self.current_screen is None

    def get_state(self):
        if self.is_start or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_creen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render("rgb_array").transpose((2, 0, 1))
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]

        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # User torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((60, 40)),
            T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = (
            next_states.
            flatten(start_dim=1).
            max(dim=1)[0].
            eq(0).
            type(torch.bool)
        )

        non_final_state_locations = (~final_state_locations)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = (
            target_net(non_final_states).
            max(dim=1)[0].
            detach()
        )

        return values
