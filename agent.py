from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add

class Agent(object):
    def __init__(self):
        self.EPSILON = 80
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.0005
        self.reward = 0
        self.memory = []
        self.model = self.create_model()
        #self.model = self.create_model("weights.hdf5")
    
    def get_state(self, game):
        snake = game.snake
        food = game.food

        # find dangers for turning a certain direction, the snake's position relative to the food and 
        # the current direction
        state = [
            self.detect_dangers([1, 0, 3, 2], game, game.direction, snake),
            self.detect_dangers([2, 3, 0, 1], game, game.direction, snake),
            self.detect_dangers([3, 2, 1, 0], game, game.direction, snake),
            food.position[0] < snake.position[0][0],
            food.position[0] > snake.position[0][0],
            food.position[1] < snake.position[0][1],
            food.position[1] > snake.position[0][1],
            game.direction == 0,
            game.direction == 1,
            game.direction == 2,
            game.direction == 3
        ]
        return np.asarray([int(i) for i in state])

    def detect_dangers(self, directions, game, curr_dir, snake):
        left = (curr_dir == directions[0] and ((list(map(add, snake.position[0], [- 20, 0])) in snake.position) or snake.position[0][0] - 20 < 20))
        right = (curr_dir == directions[1] and ((list(map(add, snake.position[0], [20, 0])) in snake.position) or snake.position[0][0] + 20 >= (game.game_height - 20)))
        up = (curr_dir == directions[2] and ((list(map(add, snake.position[0], [0, -20])) in snake.position) or snake.position[0][1] - 20 < 20))
        down = (curr_dir == directions[3] and ((list(map(add, snake.position[0], [0, 20])) in snake.position) or snake.position[0][1] + 20 >= (game.game_height - 20)))
        return left or right or up or down
        
    def set_reward(self, playing, prev_score, new_score):
        self.reward = 0
        if not playing:
            self.reward = -10
        elif new_score > prev_score:
            self.reward = 10
        return self.reward

    def train_batch(self):
        memory = self.memory
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        # get a batch from memory and train
        for state, action, reward, next_state, playing in minibatch:
            self.quick_train(state, action, reward, next_state, playing)

    def quick_train(self, state, action, reward, next_state, playing):
        if playing:
            target = reward + self.GAMMA * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        else:
            target = reward
        next_Q = self.model.predict(state.reshape((1, 11))) # predict Q values
        next_Q[0][np.argmax(action)] = target # set the predicted move to the target
        self.model.fit(state.reshape((1, 11)), next_Q, epochs=1, verbose=0)

    def create_model(self, weights = None):
        model = Sequential()
        model.add(Dense(activation = 'relu', units = 100, input_dim = 11))
        model.add(Dropout(0.1))
        model.add(Dense(activation = 'relu', units = 100))
        model.add(Dropout(0.1))
        model.add(Dense(activation = 'softmax', units = 3)) # three direction to turn
        optimizer = Adam(self.LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = optimizer)

        if weights:
            model.load_weights(weights)
        return model