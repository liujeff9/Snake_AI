import pygame
import random
import numpy as np

class Game:
    def __init__(self, game_width, game_height):
        self.playing = True
        self.food = Food()
        self.snake = Snake()
        self.direction = 1
        self.score = 0
        self.game_width = game_width
        self.game_height = game_height

    def manual_controls(self, direction, prev_direction):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                playing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and prev_direction != 0:
                    direction = 1
                elif event.key == pygame.K_RIGHT and prev_direction != 1:
                    direction = 0
                elif event.key == pygame.K_UP and prev_direction != 2:
                    direction = 3
                elif event.key == pygame.K_DOWN and prev_direction != 3:
                    direction = 2
                else:
                    direction = direction
        return direction
    
    def update_display(self, display):
        display.fill((200, 200, 200)) # grey
        self.snake.draw_snake(display, (0, 255, 0)) # green
        self.food.draw_food(display, (255, 0, 0)) # red
        pygame.display.update()

        
class Snake(object):
    def __init__(self):
        self.x = 240
        self.y = 240
        self.position = [[self.x, self.y]]

    def choose_direction(self, move, direction):
        if np.array_equal(move ,[1, 0, 0]):
            return direction
        elif np.array_equal(move,[0, 1, 0]):
            if direction == 0:
                return 2
            if direction == 1:
                return 3
            if direction == 2:
                return 1
            if direction == 3:
                return 0
        elif np.array_equal(move, [0, 0, 1]):
            if direction == 0:
                return 3
            if direction == 1:
                return 2
            if direction == 2:
                return 0
            if direction == 3:
                return 1
        return

    def update_position(self, direction, food):
        snake_head = self.position[0].copy()
        if direction == 0:
            snake_head[0] += 20
        elif direction == 1:
            snake_head[0] -= 20
        elif direction == 2:
            snake_head[1] += 20
        elif direction == 3:
            snake_head[1] -= 20
        else:
            pass

        if snake_head == food.position:
            food.spawn(self)
            self.position.insert(0, snake_head)
            return 1
        else:
            self.position.insert(0, snake_head)
            self.position.pop()
            return 0

    def collision_with_boundaries(self):
        snake_head = self.position[0]
        if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
            return True
        else:
            return False

    def collision_with_self(self):
        snake_head = self.position[0]
        if snake_head in self.position[1:]:
            return True
        else:
            return False

    def draw_snake(self, display, color):
        for p in self.position:
            pygame.draw.rect(display, color, pygame.Rect(p[0], p[1], 20, 20))


class Food(object):
    def __init__(self):
        self.position = [240, 200]

    def spawn(self, snake):
        self.position[0] = random.randrange(1,25) * 20
        self.position[1] = random.randrange(1,25) * 20
        if self.position not in snake.position:
            return
        else:
            self.spawn(snake)

    def draw_food(self, display, color):
        pygame.draw.rect(display, color, pygame.Rect(self.position[0], self.position[1], 20, 20))