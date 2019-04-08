from collections import deque

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


EXPLORATION_DECAY = 0.995
REPLAY_SIZE = 20
GAMMA = 0.95


def convert_image(image):
    """Convert a (400, 600, 3) image into (60, 90, 1) image"""
    pass


class DQN_Simple(nn.Module):
    """The DQN model"""

    def __init__(self, explore_rate=1.0):
        super(DQN_Simple, self).__init__()

        self.explore_rate = explore_rate
        self.memory = deque(maxlen=1000000)
        self.model = nn.Sequential(
            nn.Linear(in_features=4, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=24),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=2)
        )

        self.optim = optim.Adam(self.model.parameters(), lr=1e-3)
        self.cost = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Add the experience into the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def forward(self, state):
        return self.model(state)

    def act(self, state):
        if np.random.random() < self.explore_rate:
            return np.random.choice(2)

        q_values = self.model(state)
        return np.argmax(q_values.squeeze().cpu().data.numpy())

    def experience_replay(self):
        """Perform experience replay"""
        if len(self.memory) < REPLAY_SIZE:
            return
        
        batch = []
        for idx in np.random.choice(len(self.memory), REPLAY_SIZE, replace=False):
            batch.append(self.memory[idx])

        for state, action, reward, state_next, terminal in batch:

            self.optim.zero_grad()
            if not terminal:
                # this is the expected value of this state, taken action 'action', after when we know the reward
                q_update = (reward + GAMMA * np.max(self.model(state_next).squeeze().cpu().data.numpy()))
            else:
                # if it is the terminal state, then there is no state after, the value is
                # just the reward
                q_update = reward

            q_values = self.model(state)

            q_target = q_values.clone().detach()
            q_target[0][action] = q_update      # we correct the q_value and take this as label

            cost = self.cost(q_values, q_target)
            cost.backward()
            self.optim.step()


        self.explore_rate *= EXPLORATION_DECAY
        self.explore_rate = max(0.01, self.explore_rate)


class DQN_Visual(nn.Module):
    """The DQN model works on the cartpole visual output
    
    The input image is (400, 600), should be reduced to (60, 90)
    """

    def __init__(self, explore_rate=1.0):
        super(DQN_Visual, self).__init__()

        self.explore_rate = explore_rate
        self.memory = deque(maxlen=100000)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=2)
        )

        self.optim = optim.Adam(self.model.parameters, lr=1e-3)
        self.cost = nn.MSELoss()
    
    def forward(self, x):
        return x


def main_simple():
    """Run the simple architecture"""

    env = gym.make('CartPole-v1')
    agent = DQN_Simple()

    run = 0
    avg = 0
    while True:
        images = []
        run += 1
        state = env.reset()

        image = env.render('rgb_array')
        images.append(image)
        import pdb
        pdb.set_trace()

        state = torch.FloatTensor(state).unsqueeze(0)
        step = 0
        while True:
            step += 1
            action = agent.act(state)
            state_next, reward, terminal, info = env.step(action)

            image = env.render('rgb_array')
            images.append(image)

            reward = reward if not terminal else -reward
            state_next = torch.FloatTensor(state_next).unsqueeze(0)
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                if run % 20 == 0:
                    imageio.mimsave('images/frame_{:03d}.gif'.format(run), images, fps=20)
                avg = (avg * (run - 1) + step) / run
                print('Run: {}, exploration: {}, avg: {}, steps: {}'.format(
                    run, agent.explore_rate, int(round(avg)), step))
                break

            agent.experience_replay()


def main_visual():
    """Doing RL based on pixel level"""
    pass


if __name__ == '__main__':
    main_simple()
