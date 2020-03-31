import torch
from environment import Experience
import matplotlib.pyplot as plt


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


def plot(values, moving_avg_period, ):
    plt.figure(2)
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print(
        "Episode", len(values),
        "\n", moving_avg_period,
        "episode moving avg:", moving_avg[-1]
    )


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = (
            values.unfold(dimension=0, size=period, step=1).
            mean(dim=1).
            flatten(start_dim=0)
        )
    else:
        moving_avg = torch.zeros(len(values))

    return moving_avg.numpy()
