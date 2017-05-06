
import numpy as np
import tensorflow as tf
import gym

from a3c import A3C

from policies import base

def main():
    env = gym.make('Breakout-v0')
    policy = base.fc_policy(env.action_space)
    baseline = None
    agent = A3C(env=env, policy=policy, baseline=baseline)
    agent.train()

if __name__ == '__main__':
    main()
