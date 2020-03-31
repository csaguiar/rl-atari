import torch
import torch.optim as optim
from environment import (
    EnvironmentManager,
    EpsilonGreedyStrategy,
    ReplayMemory
)
from agent import Agent
from dqn import DQN
from qlearning import train

if __name__ == "__main__":
    batch_size = 256
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 100000
    lr = 0.001
    num_episodes = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_manager = EnvironmentManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, env_manager.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    input_shape = (3, 60, 40)
    n_actions = 4

    policy_net = DQN(
        input_shape,
        n_actions
    ).to(device)

    target_net = DQN(
        input_shape,
        n_actions
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Switch target to inference mode

    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    train(
        num_episodes,
        env_manager,
        agent,
        policy_net,
        target_net,
        memory,
        batch_size,
        gamma,
        optimizer,
        target_update
    )
