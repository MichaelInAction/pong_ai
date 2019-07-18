from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add

class Agent(object):

    def __init__(self, player, filename):
        self.reward = 0
        self.gamma = 0.9
        self.learning_rate = 0.0005
        self.model = self.network()
        #uncomment if you want to use pretrained weights
        #self.model = self.network(filename)
        self.epsilon = 0
        self.actual = []
        self.memory = []
        self.player = player

    #Calculate the current state of the game
    def get_state(self, game, ball):

        x_dist = 0;
        if(ball.x > self.player.x):
            x_dist = ball.x - self.player.x
        else:
            x_dist = self.player.x - ball.x

        y_dist = ball.y - self.player.y

        state = [
            #Is the ball above the player
            ball.y < self.player.y,
            #Is the ball below the player
            ball.y > self.player.y,
            #Y distance to ball
            abs(y_dist)
        ]

        return np.asarray(state)

    #Calculate what reward should be given
    def set_reward(self, bounced, scored_on, state_old, state_new):
        self.reward = 0
        if bounced:
            self.reward = 20
        if scored_on:
            self.reward = -20
        elif abs(state_new[2]) - abs(state_old[2]) < 0:
            self.reward = 4
        elif abs(state_new[2]) - abs(state_old[2]) > 0:
            self.reward = -4
        return self.reward

    #Construct the network
    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=3))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=2, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        #If weights are included, load them into the network
        if weights:
            model.load_weights(weights)
        return model

    #Commit to "memory" the reward associated with the last action and the states before and after
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1,3)))[0])
        target_f = self.model.predict(state.reshape((1,3)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1,3)), target_f, epochs=1, verbose=0)
