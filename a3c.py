
import numpy as np
import scipy
import tensorflow as tf


def update_target_graph(from_scope, to_scope):
    """Returns assign op from one scope to another."""
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


class A3C(object):

    def __init__(self, env, policy, baseline):
        """Constructor.

        Args:
            env: Env object.
            policy: callable, function to go from state to actions.
            baseline: callable, ...
        """
        self._env = env
        self._policy = policy
        self._baseline = baseline

        # ...
        self._workers = 4
        self._action_size = env.action_space.n
        self._state_size = 10

    def _local_model(self, scope, trainer):
        with tf.variable_scope(scope):
            inputs = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32)
            policy, value = self._policy(inputs)

            # Compute loss
            actions = tf.placeholder(shape=[None], dtype=tf.int32)
            actions_onehot = tf.one_hot(actions, self._action_size, dtype=tf.float32)
            target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            advantages = tf.placeholder(shape=[None], dtype=tf.float32)
            train_inputs = {
                'actions': actions,
                'target_v': target_v,
                'advantages': advantages,
            }

            responsible_outputs = tf.reduce_sum(policy * actions_onehot, [1])
            value_loss = 0.5 * tf.reduce_sum(tf.square(target_v - value))
            entropy = - tf.reduce_sum(policy * tf.log(policy))
            policy_loss = tf.reduce_sum(tf.log(responsible_outputs) * advantages)
            loss = 0.5 * value_loss + policy_loss - entropy * 0.01

            # Apply gradients
            grads = tf.gradients(
                loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            apply_grads = trainer.apply_gradients(zip(grads, global_vars))
            return inputs, policy, value, train_inputs, loss, apply_grads

    def _create_models(self):
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # Global model.
        with tf.variable_scope('global'):
            inputs = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32)
            policy, value = self._policy(inputs)
        workers = {}
        for worker in range(self._workers):
            workers[worker] = self._local_model('worker_%d' % worker, trainer)
        return (inputs, policy, value), workers

    def obtain_samples():
        return []

    Observation = collections.namedtuple(['observations', 'actions', 'rewards', 'values'])

    def train_worker(self, sess, worker_id, worker):
        inputs, policy, value, train_inputs, loss, apply_grads = worker
        # train
        max_iter = 100
        global_step = tf.train.get_global_step()
        while True:
            global_step_value = sess.run(global_step)
            if global_step_value > max_iter:
                break
            samples = self.obtain_samples()
            value_est = sess.run(value, feed_dict={
                inputs: samples.observations
            })

            # Compute advantages and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            rewards_plus = np.asarray(samples.rewards.tolist() + [0])
            discounted_rewards = discount(rewards_plus, gamma)[:-1]
            value_plus = samples.values.tolist() + [0]
            advantages = samples.rewards + gamma * value_plus[1:] - value_plus[:-1]
            advantages = discount(advantages, gamma)

            loss, _ = sess.run(
                [loss, apply_grads], feed_dict={
                    inputs: samples.observations,
                    train_inputs['actions']: samples.actions,
                    train_inputs['target_v']: discounted_rewards,
                    train_inputs['advantages']: advantages
                }
            )

    def train(self):
        graph = tf.Graph()
        with graph.as_default():
            global_net, workers = self._create_models()
            with tf.train.MonitoredSession() as sess:
                self.train_worker(sess, 0, workers[0])
                # for worker in range(self._workers):
                #     func = lambda: self.train_worker(worker)
                #     t = threading.Thread(target=(func))
                #     t.start()
                #     time.sleep(0.5)
