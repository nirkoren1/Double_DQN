import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os


class ModelQ(keras.Model):
    def __init__(self, l1_dims, l2_dims, n_actions):
        super(ModelQ, self).__init__()
        self.l1_dims = l1_dims
        self.l2_dims = l2_dims
        self.n_actions = n_actions

        self.l1 = Dense(l1_dims, activation='relu')
        self.l2 = Dense(l2_dims, activation='relu')
        self.a = Dense(n_actions, activation=None)

    def feed_forward(self, state):
        output = self.l1(state)
        output = self.l2(output)
        a = self.a(output)
        return a
