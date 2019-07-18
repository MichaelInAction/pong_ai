import pygame
import random
from random import randint
import numpy as np
from Agent import Agent
from keras.utils import to_categorical

speed = 0
pygame.font.init()

class Game:

    def __init__(self, width, height):
        pygame.display.set_caption('Pong')
        self.width = width
        self.height = height
        self.gameDisplay = pygame.display.set_mode((width, height + 80))
        self.goal = False
        self.player1 = Player(self, 'left')
        self.player2 = Player(self, 'right')
        self.ball = Ball(self)

    def reset(self):
        self.player1.y = self.height / 2
        self.player2.y = self.height / 2
        self.player1.scored = False
        self.player2.scored = False
        self.ball = Ball(self)

class Player(object):

    def __init__(self, game, side):
        if side == 'left':
            self.x = 30
            self.agent = Agent(self, "weights1.hdf5")
        if side == 'right':
            self.x = game.width - 30
            self.agent = Agent(self, "weights2.hdf5")
        self.y = game.height / 2
        self.height = 50
        self.game = game
        self.score = 0
        self.scored = False
        self.bounced = False

    def display_player(self, game):
        pygame.draw.rect(game.gameDisplay, (255, 255, 255), pygame.Rect(self.x-5, self.y-self.height/2, 10, self.height))

    def move(self, ball, other_player):
        self.agent.epsilon = 500 - (self.score + other_player.score)
        state_old = self.agent.get_state(self.game, ball)
        if randint(0, 700) < self.agent.epsilon:
            next_move = to_categorical(randint(0, 1), num_classes=2)
        else:
            prediction = self.agent.model.predict(state_old.reshape((1,3)))
            next_move = to_categorical(np.argmax(prediction[0]), num_classes=2)

        if np.array_equal(next_move, [0,1]) and self.y-self.height/2 > 0:
            self.y = self.y - 1
        if np.array_equal(next_move, [1,0]) and self.y+self.height/2 < self.game.height:
            self.y = self.y + 1

        state_new = self.agent.get_state(self.game, ball)

        reward = self.agent.set_reward(self.bounced, other_player.scored, state_old, state_new)

        self.agent.train_short_memory(state_old, next_move, reward, state_new, other_player.scored)

        self.agent.remember(state_old, next_move, reward, state_new, other_player.scored)

        #Keyboard controls for non-AI play
        #if keyboard.is_pressed("up") and self.y-self.height/2 > 0:
            #self.y = self.y - 1
        #if keyboard.is_pressed("down") and self.y+self.height/2 < self.game.height:
            #self.y = self.y + 1


class Ball(object):

    def __init__(self, game):
        self.x = game.width / 2
        self.y = game.height / 2
        self.x_vel = randint(0, 1)
        if self.x_vel == 0:
            self.x_vel = -1
        self.y_vel = random.uniform(-1, 1)
        self.game = game

    def display_ball(self, game):
        pygame.draw.rect(game.gameDisplay, (255, 255, 255), pygame.Rect(self.x-5, self.y-5, 10, 10))

    def move(self):
        self.x = self.x + self.x_vel
        self.y = self.y + self.y_vel
        self.bounce()
        self.checkPoint()

    def bounce(self):
        if self.y >= self.game.height or self.y <= 0:
            self.y_vel = self.y_vel * -1
        if (self.x > self.game.player1.x-5 and self.x < self.game.player1.x+5) and (self.y > self.game.player1.y-self.game.player1.height/2 and self.y < self.game.player1.y+self.game.player1.height/2):
            self.game.player1.bounced = True
            self.x_vel = self.x_vel * -1
        if (self.x > self.game.player2.x-5 and self.x < self.game.player2.x+5) and (self.y > self.game.player2.y-self.game.player2.height/2 and self.y < self.game.player2.y+self.game.player2.height/2):
            self.game.player2.bounced = True
            self.x_vel = self.x_vel * -1

    def checkPoint(self):
        if self.x > self.game.width - 30:
            self.game.player1.scored = True
            self.game.player1.score = self.game.player1.score + 1
        if self.x < 30:
            self.game.player2.scored = True
            self.game.player2.score = self.game.player2.score + 1



def display_ui(game, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score_p1 = myfont.render('SCORE: ', True, (255, 255, 255))
    text_score_number_p1 = myfont.render(str(game.player1.score), True, (255, 255, 255))
    text_score_p2 = myfont.render('SCORE: ', True, (255, 255, 255))
    text_score_number_p2 = myfont.render(str(game.player2.score), True, (255, 255, 255))
    game.gameDisplay.blit(text_score_p1, (45, 440))
    game.gameDisplay.blit(text_score_number_p1, (120, 440))
    game.gameDisplay.blit(text_score_p1, (300, 440))
    game.gameDisplay.blit(text_score_number_p2, (375, 440))

def display(game, ball, player1, player2):
    game.gameDisplay.fill((0, 0, 0))
    display_ui(game, 0)
    ball.display_ball(game)
    player1.display_player(game)
    player2.display_player(game)

def update_screen():
    pygame.display.update()

def run():
    pygame.init()
    game = Game(440, 440)
    ball = game.ball
    player1 = game.player1
    player2 = game.player2
    display(game, ball, player1, player2)
    i = 0
    while (i > -1):
        i = i + 1
        update_screen()
        ball.move()
        player1.move(ball, player2)
        player2.move(ball, player1)
        display(game, ball, player1, player2)
        if player1.scored or player2.scored:
            game.reset()
            ball = game.ball
            #player1.agent.model.save_weights("weights1.hdf5")
            #player2.agent.model.save_weights("weights2.hdf5")
        pygame.time.wait(speed)


run()
