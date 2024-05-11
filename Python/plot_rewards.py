import matplotlib.pyplot as plt
import torch


# her episodeda ne kadar reward alinmis connection bitene kadar olan rewardlari dondurur
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.get_current_fig_manager().set_window_title('Rewards')
    rewards_t = torch.tensor(rewards, dtype=torch.float)

    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.plot(rewards_t.numpy(), label='Rewards per Episode')

    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-Episode Moving Average')

    plt.legend()
    plt.show()
