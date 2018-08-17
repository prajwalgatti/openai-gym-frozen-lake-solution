#FROZEN LAKE GAME [Q LEARNING]

import numpy as np
import gym
import random

# --ENVIRONMENT CREATION--
env = gym.make("FrozenLake-v0")

# --Q-TABLE INITIALISATION--
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
#print(qtable)
#print("action_size: ", action_size)
#print("state_size: ", state_size)

# --HYPERPARAMETERS--
total_episodes = 10000		# Total episodes
learning_rate = 0.8 		# Learning rate
max_steps = 99				# Max steps per episode
gamma = 0.95 				# Discounting rate

# Exploration parameters
epsilon = 1.0 				# Exploration rate
max_epsilon = 1.0 			# Exploration probability at start
min_epsilon = 0.01			# Min exploration probability
decay_rate = 0.01			# Exponential decay rate for exploration probability

# --Q-LEARNING ALGORITHM--
rewards = [] # List of rewards

for episode in range(total_episodes):
	# Reset the environment
	state = env.reset()
	step = 0
	done = False
	total_rewards = 0 # Total rewards obtained in the episode

	for step in range(max_steps):
		exp_exp_tradeoff = random.uniform(0, 1)
		if exp_exp_tradeoff > epsilon:
			# Exploitation --> taking the biggest Q-value for this state
			action = np.argmax(qtable[state, :])
		else:
			# Exploration
			action = env.action_space.sample()

		# Take the action (a) and observe the outcome state (s') and reward(r')
		new_state, reward, done, info = env.step(action)

		# Update Q(s,a):= Q(s,a) + lr [reward + gamma * max Q(s',a) - Q(s,a)]
		# qtable[new_state,:] : all the actions we can take from new_state (s')
		qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

		total_rewards += reward

		# Move to new state
		state = new_state

		# If done (agent died) --> finish episode
		if done:
			break

	episode += 1

	# Reduce epsilon (because we need less and less exploration)
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
	rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards)/total_episodes))
print(qtable)



# Play Frozen Lake!

env.reset()

for episode in range(5):
	state = env.reset()
	step = 0
	done = False
	print("**********************************************************")
	print("EPISODE ", episode)
	print(qtable)

	for step in range(max_steps):
		env.render()
		# Take the action (index) that has the maximum expected future reward given that state
		action = np.argmax(qtable[state, :])

		new_state, reward, done, info = env.step(action)

		if done:
			break

		state = new_state

env.close()