import collections
import numpy as np
import tensorflow as tf
import gym

import utils

Sample = collections.namedtuple('Sample', ['observations', 'actions', 'rewards'])


def rollout(env, policy, max_path_length=np.inf, animated=False):
    observations, actions, rewards = [], [], []
    observation = env.reset()
    path_length = 0
    while path_length < max_path_length:
        action = policy.get_action(observation)
        next_observation, reward, done, _ = env.step(policy.get_action(observation))
        observations.append(observation)
        rewards.append(reward)
        actions.append(action)
        path_length += 1
        if done:
            break
        observation = next_observation
    return Sample(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards)
    )


class BatchSampler(object):

    def __init__(self, env):
        self._env = env

    def sample(self, policy, batch_size, max_path_length=np.inf):
        samples = []
        for i in range(batch_size):
            samples.append(rollout(env, policy, max_path_length, animated=False))
        return samples


class BatchPolicyGradient(object):

    def __init__(self, env, policy, batch_size=32):
        self.env = env
        self.policy = policy
        self._batch_size = batch_size
        self._batch_sampler = BatchSampler(env)
        self._action_size = env.action_space.n
        self._observation_shape = list(env.observation_space.shape)

    def train(self):
        graph = tf.Graph()
        with graph.as_default():
            inputs = tf.placeholder(shape=[None] + self._observation_shape, dtype=tf.float32)
            pred_actions, pred_value = self.policy.create_graph(inputs)
            # Training procedure.
            rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            actions = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_onehot = tf.one_hot(actions, self._action_size)
            responsible_outputs = tf.reduce_sum(pred_actions * actions_onehot, [1])
            loss = - tf.reduce_mean(tf.log(responsible_outputs) * rewards)
            tvars = tf.trainable_variables()
            gradients = tf.gradients(loss, tvars)
            gradients, tvars = zip(*[(g, v) for g, v in zip(gradients, tvars) if g is not None])
            gradients = list(gradients)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
            grad_placeholder = []
            for idx, _ in enumerate(tvars):
                grad_placeholder.append(tf.placeholder(tf.float32, name='%d_grad' % idx))
            train_op = optimizer.apply_gradients(zip(grad_placeholder, tvars))

            total_episodes = 100
            max_path_length = 10
            with tf.train.MonitoredSession() as sess:
                with sess._sess._sess._sess._sess.as_default():
                    episode = 0
                    while episode < total_episodes:
                        samples = self._batch_sampler.sample(
                            self.policy, self._batch_size, max_path_length=max_path_length)
                        all_grads = []
                        for i in range(max_path_length):
                            sample_obs = [samples[j].observations[i] for j in range(self._batch_size) if len(samples[j].observations) > i]
                            sample_act = [samples[j].actions[i] for j in range(self._batch_size) if len(samples[j].observations) > i]
                            sample_r = [samples[j].rewards[i] for j in range(self._batch_size) if len(samples[j].observations) > i]
                            if len(sample_obs) == 0:
                                break
                            sample_r = np.vstack(sample_r).reshape([-1])
                            discounted_r = utils.discount_rewards(sample_r)
                            print(sample_r, discounted_r)
                            discounted_r -= np.mean(discounted_r)
                            discounted_r /= np.std(discounted_r)
                            print(sample_r, discounted_r)
                            feed_dict = {
                                inputs: np.vstack(sample_obs),
                                actions: np.vstack(sample_act).reshape([-1]),
                                rewards: discounted_r
                            }
                            l, cur_grads = sess.run([loss, gradients], feed_dict)
                            if i == 0:
                                print(l)
                            if all_grads:
                                for idx, grad in enumerate(cur_grads):
                                    all_grads[idx] += grad
                            else:
                                all_grads = cur_grads
                        if all_grads:
                            sess.run(train_op, feed_dict=dict(zip(grad_placeholder, all_grads)))
                        episode += self._batch_size


if __name__ == "__main__":
    from policies import base
    env = gym.make('CartPole-v0')
    policy = base.MLPPolicy(env.action_space.n)
    algo = BatchPolicyGradient(env, policy)
    algo.train()
