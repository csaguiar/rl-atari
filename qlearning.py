from itertools import count
from tools import extract_tensors
from environment import Experience, QValues
import torch.nn.functional as F


def train(
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
):
    episode_durations = []
    episode_rewards = []
    for episode in range(num_episodes):
        env_manager.reset()
        state = env_manager.get_state()

        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = env_manager.take_action(action)
            next_state = env_manager.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = \
                    extract_tensors(experiences)

                current_q_values = \
                    QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(
                    current_q_values,
                    target_q_values.unsqueeze(1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if env_manager.done:
                episode_durations.append(timestep)
                episode_rewards.append(reward)
                print(
                    "Episode", episode,
                    "duration", timestep
                )
                # plot(episode_durations, 100)
                # pd.DataFrame({
                #     "episode": np.arange(episode+1),
                #     "duration": episode_durations,
                #     "reward": episode_rewards
                # }).to_csv("tmp/history.csv")
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env_manager.close()
