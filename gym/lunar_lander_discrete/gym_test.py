import gym
import numpy as np
from agent import Agent


env = gym.make("LunarLander-v2")
n_actions = 4
ag = Agent(alpha=0.001, input_dims=env.observation_space.shape[0], n_actions=n_actions, tau=0.05,
           batch_size=100, epsilon=1, epsilon_decay=0.99, min_epsilon=0.01)
ag.model.load_weights(r'C:\Users\Nirkoren\PycharmProjects\DQN\gym\lunar_lander_discrete\agent\agent')
score = 0

if __name__ == '__main__':
    observation = env.reset()
    while True:
        env.render()
        action = ag.take_an_action_for_real(observation)
        observation, reward, done, info = env.step(action)
        score += reward
        if done:
            observation = env.reset()
            print(score)
            score = 0
