import numpy as np
import gym
import sys
from agent import Agent
import animate

env = gym.make("LunarLander-v2")
n_actions = 4
ag = Agent(alpha=0.001, input_dims=env.observation_space.shape[0], n_actions=n_actions, tau=1,
           batch_size=100, epsilon=1, epsilon_decay=0.95, min_epsilon=0.01)
score_history = []
history_size = 40
epsilon = 1


if __name__ == '__main__':
    loop = 0
    best_score = -1000000000000000
    avg_score = best_score - 1
    while True:
        loop += 1
        observation = env.reset()
        score = 0
        while True:
            action = ag.take_an_action(observation)
            ob = observation
            observation, reward, done, info = env.step(action)
            ag.memory.save_step(ob, action, reward, observation, done)
            score += reward
            ag.learn(done)
            if done:
                break

        score_history.append(score)
        if len(score_history) >= history_size:
            avg_score = np.mean(score_history[-history_size:])
            animate.update(avg_score)
        if avg_score > best_score:
            print('')
            ag.save_agent(r'C:\Users\Nirkoren\PycharmProjects\DQN\gym\lunar_lander_discrete\agent\agent', score)
            best_score = avg_score
        sys.stdout.write(f"\rloop - {loop}  score - {score}  best - {best_score}  avg score - {avg_score}")
        sys.stdout.flush()
