import gym
import numpy as np
from model import PolicyGradientAgent
from utils import plotLearning
from gym import wrappers

def preprocess(observation):
	try:
		observation = np.mean(observation[15:200, 30:125], axis=2)
	except:
		pass
	finally:
		#no pre-processing required.
		if observation.shape == (185, 95):
			return observation
		else:
			exit()

def stack_frames(stacked_frames, frame, buffer_size):
	#if no frames exist, we'll make black frames.
	if stacked_frames is None:
		stacked_frames = np.zeros((buffer_size, *frame.shape), dtype='float32')
		#iterate to set each row to current observation.
		for idx, _ in enumerate(stacked_frames):
			stacked_frames[idx, :] = frame
	#we want to pop off the bottom observation, shift everything down, 
	#	and set last spot to be current observation
	else:
		stacked_frames[0:buffer_size-1, :] = stacked_frames[1:,:]
		frame = np.array(frame, dtype='float32') 

	return stacked_frames



if __name__ == '__main__':
	load_checkpoint = False
	#Since we're using RMSProp as our optimizer, learning rate doesn't matter too much. Just be sure it doesn't start off too small.
	agent = PolicyGradientAgent(learning_rate=0.001, discount_factor=0.98, num_actions=6, fc1=256, chkpt_dir='tmp/checkpoint')
	filename = 'space-invaders-alpha001-newGcalc.png'

	if load_checkpoint:
		agent.load_checkpoint()

	env = gym.make('SpaceInvaders-v0')
	score_history = []
	score = 0
	num_episodes = 2000
	stack_size = 4

	for i in range(num_episodes):
		done = False
		#average score of past 20 episodes to give us idea if we're learning
		avg_score = np.mean(score_history[max(0, i-20): (i+1)])

		if i % 20 == 0 and i > 0:
			print('episode: ', i, ' score: ', score, 'average score %.3f' % avg_score)
			plotLearning(score_history, filename=filename, window=20)
		else:
			print('episode: ', i, ' score: ', score)

		observation = env.reset()
		observation = preprocess(observation)
		stacked_frames = None
		stacked_frames = stack_frames(stacked_frames, observation, stack_size)

		score = 0
		while not done:
			action = agent.choose_action(stacked_frames)
			#We have done info. we may want to store this. 
			obseravtion, reward, done, info = env.step(action)
			observation = preprocess(observation)
			stacked_frames = stack_frames(stacked_frames, observation, stack_size)
			agent.store_transition(observation, action, reward, done)

			score += reward

		score_history.append(score)

		if i % 10 == 0:
			#learn every 10 episodes. 
			#This way we have a different variation of states/rewards/actions to learn from
			agent.learn()
			agent.save_checkpoint()

	plotLearning(score_history, filename=filename, window=20)




