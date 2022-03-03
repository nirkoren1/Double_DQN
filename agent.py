from Dqn_model import ModelQ
from replay_buffer import ReplayBuffer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


class Agent:
    def __init__(self, alpha, input_dims, n_actions, tau, epsilon, epsilon_decay, min_epsilon, gamma=0.99, update_target_every=5,
                 max_size=1_000_000, layer1_size=300, layer2_size=200, batch_size=100):
        self.n_actions = n_actions
        self.tau = tau
        self.gamma = gamma
        self.update_target_every = update_target_every
        self.memory = ReplayBuffer(input_dims, n_actions, max_size)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = ModelQ(layer1_size, layer2_size, n_actions)

        self.target_model = ModelQ(layer1_size, layer2_size, n_actions)

        self.model.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')
        self.target_model.compile(optimizer=Adam(learning_rate=alpha), loss='mean_squared_error')

        # self.update_nets_parameters(tau=1)
        self.target_model.set_weights(self.model.get_weights())

    def take_an_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            a = self.model.feed_forward(state)[0]
            return np.argmax(a)
        return np.random.randint(self.n_actions)

    def take_an_action_for_real(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        a = self.model.feed_forward(state)[0]
        return np.argmax(a)

    def remember(self, state, action, reward, state_, done):
        self.memory.save_step(state, action, reward, state_, done)

    def learn(self, terminal):
        if self.memory.cntr < self.batch_size:
            return
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        # learning of the networks
        with tf.GradientTape() as tape:
            # getting the target actor actions
            predicted_qs = self.model.feed_forward(states)
            future_predicted_qs = self.target_model.feed_forward(states_)

            max_future_q = np.max(future_predicted_qs, axis=1)
            new_q = rewards + self.gamma * max_future_q * (1 - dones)
            current_qs = predicted_qs
            current_qs = np.array(current_qs, dtype=float)
            actions = np.array(actions, dtype=int)
            # for index in range(len(current_qs)):
            #     current_qs[index][actions[index]] = float(new_q[index])
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            current_qs[batch_index, actions] = new_q
            current_qs = tf.convert_to_tensor(current_qs, dtype=tf.float32)

            # calculate losses
            model_loss = tf.losses.MSE(current_qs, predicted_qs)

        # calculate the gradients of the two critics
        model_gradient = tape.gradient(model_loss, self.model.trainable_weights)

        self.model.optimizer.apply_gradients(zip(model_gradient, self.model.trainable_variables))
        if terminal:
            self.learn_step_cntr += 1
            self.epsilon *= self.epsilon_decay
            self.epsilon = min(self.epsilon, self.min_epsilon)

        if self.learn_step_cntr % self.update_target_every != 0:
            return
        # self.update_nets_parameters(self.tau)
        self.target_model.set_weights(self.model.get_weights())

    def update_nets_parameters(self, tau):
        # update actor weights
        weights = []
        target_weights = self.target_model.weights
        for idx, weight in enumerate(self.model.weights):
            weights.append(weight * tau + (1 - tau) * target_weights[idx])
        self.target_model.set_weights(weights)

    def save_agent(self, path, score):
        files = os.listdir(path)
        self.model.save_weights(path)
        # self.actor.save(path + f"/{len(files) + 1}-fitness=" + str(score)[:7])
        print(f"Agent saved with {score} score")
