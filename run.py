import gym
from main import stack_frames, preprocess
from model import PolicyGradientAgent

agent = PolicyGradientAgent(learning_rate=0.001, discount_factor=0.9, num_actions=6, chkpt_dir='tmp/checkpoint')
agent.load_checkpoint()

env = gym.make('SpaceInvaders-v0')
observation = env.reset()
observation = preprocess(observation)

stack_size = 4
stacked_frames = None
stacked_frames = stack_frames(stacked_frames, observation, stack_size)

done = False
while not done:
	env.render()

	action = agent.choose_action(stacked_frames)
	obseravtion, reward, done, info = env.step(action)

	observation = preprocess(observation)
	stacked_frames = stack_frames(stacked_frames, observation, stack_size)

env.close()