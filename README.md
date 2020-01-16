# RL-SpaceInvaders
A Reinforcement Learning model that utilizes a convolutional neural network to determine the loss of a particular action given the state and discounted future rewards. It is optimized through RMSProp in order to converge to a policy that maximizes score (rewards).

NOTE: This current implementation converges poorly. I'm going to need to refine some of the hyperparameters and double check my logic, however the code runs and the concepts are demonstrated.

All you need to do is execute main.py. Be sure to train on a significant amount of episodes and to train using a GPU or else it will take an incredibly long time to train. Begin execution from main.py
