import numpy as np


# bunu kullanacagiz yuksek ihtimalle
def min_max_normalize(rewards):
    min_val = np.min(rewards)
    max_val = np.max(rewards)
    normalized_rewards = (rewards - min_val) / (max_val - min_val)
    return normalized_rewards


def standard_deviation_normalize(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    normalized_rewards = (rewards - mean) / std
    return normalized_rewards


def z_score_normalize(rewards):
    mean = np.mean(rewards)
    std = np.std(rewards)
    normalized_rewards = (rewards - mean) / std
    return normalized_rewards


def clip_rewards(rewards, min_val=-1, max_val=1):
    clipped_rewards = np.clip(rewards, min_val, max_val)
    return clipped_rewards


def running_average_normalize(rewards, beta=0.99):
    normalized_rewards = np.zeros_like(rewards)
    running_mean = 0
    for i in range(len(rewards)):
        running_mean = beta * running_mean + (1 - beta) * rewards[i]
        normalized_rewards[i] = rewards
