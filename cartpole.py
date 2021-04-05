import numpy as np
import gym
from gym.envs.registration import register
import random
import tensorflow as tf
import dqn
from collections import deque
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.max_episode_steps = 10000

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

learning_rate = 1e-1
discount_rate = 1.5
REPLAY_MEMORY = 50000
results = []


def replay_train(main_dqn, target_dqn, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = main_dqn.predict(state)

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + discount_rate * \
                np.max(target_dqn.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return main_dqn.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(main_dqn):
    s = env.reset()
    reward_sum = 0
    while True:
        # env.render()
        a = np.argmax(main_dqn.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print("Total score : {}".format(reward_sum))
            break


def main():
    max_episodes = 2000
    replay_buffer = deque()

    with tf.Session() as sess:
        main_dqn = dqn.DQN(sess, input_size, output_size, name="main")
        target_dqn = dqn.DQN(sess, input_size, output_size, name="target")
        tf.global_variables_initializer().run()


        copy_ops = get_copy_var_ops(
            dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for i in range(max_episodes):
            #초기화
            state = env.reset()
            e = 1. / ((i / 10) + 1)
            step_count = 0
            done = False

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(main_dqn.predict(state))

                next_state, reward, done, _ = env.step(action)

                if done:
                    reward = -10

                replay_buffer.append((state, action, reward, next_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break

            print("Episode : {}, steps : {}".format(i, step_count))
            results.append(step_count)


            if i % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = replay_train(main_dqn, target_dqn, minibatch)
                print("Loss : ", loss)
                sess.run(copy_ops)

        bot_play(main_dqn)
        plt.title("(2015) Total step count on each episode")
        plt.plot(range(len(results)), results)
        plt.show()


if __name__ == "__main__":
    main()