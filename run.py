import pygame
from agent import Agent
from game import Game
from random import randint
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def train(game_width, game_height):
    pygame.init()
    agent = Agent()
    scores = []
    counter_list =[]
    counter = 0
    high_score = 0

    while counter < 100:

        game = Game(500, 500)
        snake = game.snake
        food = game.food

        display = pygame.display.set_mode((game_width, game_height))

        game.update_display(display)

        while game.playing:
            # print('Snake: ' + str(snake.position[0]) + ' Food: ' + str(food.position))
            
            # get the current state of the game
            current_state = agent.get_state(game)

            # use the epsilon for randomness in actions
            if randint(0, 200) < agent.EPSILON - counter:
                action = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict the next move
                prediction = agent.model.predict(current_state.reshape((1,11)))
                # find the index of the predicted move 
                action = to_categorical(np.argmax(prediction[0]), num_classes=3)

            curr_score = game.score

            # choose the correct direction and update the snake's position
            game.direction = snake.choose_direction(action, game.direction)
            game.score += snake.update_position(game.direction, food)

            # check if the snake is still alive
            if game.snake.collision_with_boundaries() or game.snake.collision_with_self():
                game.playing = False

            # get the new state of the game after making a move
            new_state = agent.get_state(game)

            # reward the snake for moving, eating the food or crashing
            reward = agent.set_reward(game.playing, curr_score, game.score)
            # train the snake based the move made and the resulting new state
            agent.quick_train(current_state, action, reward, new_state, game.playing)
            # save the result of making the move into memory
            agent.memory.append((current_state, action, reward, new_state, game.playing))
            
            high_score = max(high_score, game.score)
            pygame.display.set_caption('Snake Gen ' + str(counter) + '     High Score: ' + str(high_score) )
            game.update_display(display)

            pygame.time.wait(1)

        # train the model again after snake as died
        agent.train_batch()
        print('Generation ' + str(counter) + '      Score: ' + str(game.score))
        counter += 1
        scores.append(game.score)
        counter_list.append(counter)
    create_graph(scores, counter_list)
    agent.model.save_weights('weights.hdf5')


def create_graph(scores, counter_list):
    plt.figure('graph')
    plt.plot(counter_list, scores, color = 'black', marker = 'd', linestyle = '-')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Score at Each Generation')
    plt.savefig('results.png', dpi=1000)
    plt.show()


train(500,500)