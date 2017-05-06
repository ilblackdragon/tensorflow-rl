import numpy as np


def discount_rewards(rewards, gamma=0.99):
    """Takes 1D float array of rewards and compute discounted reward."""
    discounted = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add
    return discounted


def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s
