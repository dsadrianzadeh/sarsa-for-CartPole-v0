import numpy as np


class Agent:

    def __init__(self, alpha, epsilon, gamma, state_space, action_space):

        self.alpha = alpha  # step-size parameter
        self.epsilon = epsilon  # probability of taking a random action in an ε-greedy policy
        self.gamma = gamma  # discount-rate parameter (discount factor)
        self.state_space = state_space
        self.action_space = action_space

        self.Q = {}
        for s in self.state_space:
            for a in range(2):
                self.Q[s, a] = 0

    def policy(self, state):

        q_values = np.array([self.Q[state, a] for a in range(2)])

        if np.random.random() < self.epsilon:
            action = self.action_space.sample()  # random action - exploration
        else:
            action = np.argmax(q_values)  # greedy action - exploitation

        return action

    def update_policy(self, state, action, reward, state_, action_, t):

        delta = reward + self.gamma * self.Q[state_, action_] - self.Q[state, action]  # TD error
        self.Q[state, action] = self.Q[state, action] + self.alpha * delta

        # Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state–action
        # pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy
        # (which can be arranged, for example, with ε-greedy policies by setting ε = 1/t).
        if t != 0:
            self.epsilon = 1 / t
        else:
            self.epsilon = 1.0
