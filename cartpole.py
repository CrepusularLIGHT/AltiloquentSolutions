import gym
import sys
import keras
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

num_episodes = 1000
n_win_ticks = 195
max_env_steps = None

# Training parameters
gamma = 1.0 # discount factor
epsilon = 1.0 # exploration
epsilon_min = 0.01
epsilon_decay = 0.995
alpha = 0.01 # learning rate
alpha_decay = 0.01

batch_size = 64
monitor = False
quiet = False

# Environment parameters
memory = deque(maxlen=100000)
if max_env_steps is not None: env.max_peisode_steps = max_env_steps

# Model Definition
model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu')) # 24 neurons, 4 parameters(state), rectified linear unit
model.add(Dense(48, activation='relu'))
model.add(Dense(2, activation='relu'))
model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

# Defining necessary function

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def choose_action(state, epsilon):
    return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(model.predict(state))

def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t+1)*epsilon_decay)))

def preprocess_state(state):
    return np.reshape(state, [1, 4])

def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(memory, min(len(memory), batch_size))

    for state, action, reward, next_state, done in minibatch:
        y_target = model. predict(state)
        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])
        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Run function
def run():
    scores = deque(maxlen=100)

    for e in range(num_episodes):
        state = preprocess_state(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            env.render()
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)

        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e-100))
            return e - 100
        if e % 20 == 0 and not quiet:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        replay(batch_size, get_epsilon(e))

    if not quiet: print('Did not solve after {} episodes'.format(e))
    return e

run()

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         #print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         # if done:
#         #     print("Episode finished after {} timesteps".format(t+1))
#         #     break
# env.close()

# Building neural netowrk 
