import os 
import numpy as np
import tensorflow as tf



#contrast to Deep Q Learning, Policy Gradients don't actually try to learn 
#the action value or value function, rather approximate the actual policy of the agent
class PolicyGradientAgent(object):
	def __init__(self, learning_rate, discount_factor=0.99, num_actions=6
		, fc1=256, input_shape=(185, 95), channels=1
		, chkpt_dir='tmp/checkpoints', gpu={'GPU': 1}):

		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.action_space = [i for i in range(num_actions)]
		self.input_height = input_shape[0]
		self.input_width = input_shape[1]
		self.channels = channels
		self.num_actions = num_actions
		self.state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.done_memory = []
		self.gpu = gpu
		self.fc1 = fc1
		#configuration to tell tensorflow which GPU we want to use
		#should only need to do this if you have more than one gpu
		config = tf.ConfigProto(device_count = self.gpu)
		#session graph that uses our configuration
		self.sess = tf.Session(config=config)
		#need to build our network immediately afterwards
		self.build_net()
		#initialize our weights
		self.sess.run(tf.global_variables_initializer())
		#setup a save and a checkpoint file
		self.saver = tf.train.Saver()
		#self.checkpoint_file = os.path.join(chkpt_dir)

	def build_net(self):
		with tf.variable_scope('parameters'):
			#placeholder tensors so we can build the network.
			self.input = tf.placeholder(tf.float32
				, shape=[None, self.input_height, self.input_width, self.channels], name='input')
			#label corresponds to the action the agent takes. shape is of batch size
			self.label = tf.placeholder(tf.int32, shape=[None], name = 'label')
			#discounted future rewards following a given timestep. shape is of batch size
			self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

		#this is our 3 convolutional layers and two dense layers.
		with tf.variable_scope('conv_layer_1'):
			#purpose of initializer is to initialize parameters in such a way that network won't have 
			#network with one layer w/ parameters significantly larger than the others
			conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=(8,8), strides=4
				, name='conv1', kernel_initializer=tf.contrib.layers.xavier_initializer())
			batch1 = tf.layers.batch_normalization(inputs=conv1, epsilon = 1e-5, name='batch1')
			conv1_activated = tf.nn.relu(batch1)

		with tf.variable_scope('conv_layer_2'):
			conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64, kernel_size=(4,4), strides=2
				, name='conv2d', kernel_initializer=tf.contrib.layers.xavier_initializer())
			batch2 = tf.layers.batch_normalization(inputs = conv2, epsilon=1e-5, name='batch2')
			conv2_activated = tf.nn.relu(batch2)

		with tf.variable_scope('conv_layer_3'):
			conv3 = tf.layers.conv2d(inputs=conv2_activated, filters = 128, kernel_size=(3,3), strides=1
				, name='conv3', kernel_initializer=tf.contrib.layers.xavier_initializer())
			batch3 = tf.layers.batch_normalization(inputs = conv3, epsilon=1e-5, name='batch3')
			conv3_activated = tf.nn.relu(batch3)

		with tf.variable_scope('fc1'):
			flatten = tf.layers.flatten(conv3_activated)
			dense1 = tf.layers.dense(flatten, units=self.fc1, activation=tf.nn.relu)

		with tf.variable_scope('fc2'):
			dense2 = tf.layers.dense(dense1, units=self.num_actions
				, kernel_initializer=tf.contrib.layers.xavier_initializer())

		#determine output
		self.actions = tf.nn.softmax(dense2, name='actions')

		#calculate loss with activated output of network
		with tf.variable_scope('loss'):
			negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=dense2, labels=self.label)
			self.loss = tf.reduce_mean(negative_log_prob * self.G)

		#optimize the network here
		with tf.variable_scope('train'):
			self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99
				, momentum=0, epsilon=1e-6).minimize(self.loss)

	#code up the action selection algo for agent
	#we're trying to approx policy, so we're trying to approx the distribution by which the agent
	#	chooses actions given it's in some state 's'

	def choose_action(self, observation):
		observation = np.array(observation).reshape((-1, self.input_height, self.input_width, self.channels))
		# observation = observation[np.newaxis, :]
		#returns tuple value. we only care about the 0th index
		probabilities = self.sess.run(self.actions, feed_dict={self.input:observation})[0]
		action = np.random.choice(self.action_space, p=probabilities)
		return action

	def store_transition(self, observation, action, reward, done):
		self.state_memory.append(observation)
		self.action_memory.append(action)
		self.reward_memory.append(reward)
		self.done_memory.append(done)
		#we might want to store terminal memory here to so we can track done flags.

	#big problem to solve is the fact that policy gradients are sample inefficient.
	#monte-carlo methods: end of an episode the agent is learning (throws away all experience required in prior episodes)
	#we can queue up a batch of episodes and learn off the batch. (make sure rewards to spill into other epiosdes)
	def learn(self):
		#we need to change our memory into numpy arrays so we can feed them into our tensorflow learning func
		state_memory = np.array(self.state_memory, dtype='float32')
		state_memory = np.expand_dims(state_memory, axis=3)
		print(state_memory.shape)

		action_memory = np.array(self.action_memory)
		reward_memory = np.array(self.reward_memory)
		done_memory = np.array(self.done_memory)

		#calculate expected future rewards starting from any given state
		G = np.zeros_like(reward_memory)

		#iterate over entire memory and take into account the rewards agent receives for all subsequent timesteps.
		#also make sure we don't take into account rewards for the next episode
		# This is solving for G sub t
		print('done memory should look like... ', done_memory)

		for t in range(len(reward_memory)):
			G_sum = 0.0
			discount = 1.0
			#WE SHOULD KEEP TRACK OF THE DONE FLAG TO MAKE SURE WE KNOW WHERE THE EP IS ENDING.
			#This code is SUPER inefficient...
			for k in range(t, len(reward_memory)):
				#check for done flag here. if done, break. else keep accumulating G_sum.
				if t > 0 and done_memory[k]:
					break
				G_sum += reward_memory[k] * discount
				discount *= self.discount_factor
			#this stores the discounted future rewards at some timestep
			G[t] = G_sum

		#lets normalize our G value
		mean = np.mean(G)
		std = np.std(G) if np.std(G) > 0 else 1
		G = (G-mean)/std


		#we are going to run train_op while passing in state memory, action memory and G into our placeholder tensors
		_ = self.sess.run(self.train_op,
			feed_dict={self.input:state_memory
				, self.label:action_memory
				, self.G: G})

		#done. clear out agent's memories.
		self.state_memory = []
		self.action_memory=[]
		self.reward_memory=[]
		self.done_memory=[]

	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.saver.restore(self.sess, 'rl-SpaceInvaders-model')

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		self.saver.save(self.sess, 'rl-SpaceInvaders-model')







