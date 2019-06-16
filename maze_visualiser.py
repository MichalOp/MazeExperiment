import pygame, sys
import numpy as np
from maze_game import maze_game
import random
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE = (  0,   0, 255)

class visualizer:
    
    def __init__(self, target_game, screen_size):
        self.game = target_game
        self.screen_size = screen_size
        
        pygame.init()
        self.helper_board = np.zeros([self.game.map_size*2-1,self.game.map_size*2-1])
        self.DISPLAYSURF = pygame.display.set_mode((screen_size, screen_size), 0, 32)
        pygame.display.set_caption('Maze')
        
    def draw(self):
        
        self.DISPLAYSURF.fill(BLACK)
        
        board = self.game.lab
        player_pos = self.game.player_pos
        
        B = self.screen_size//board.shape[0]
        
        if self.game.episode_steps == 0:
            self.helper_board = np.zeros(board.shape)
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                
                pos = (i*B, j*B, B, B)
                
                if board[i][j] == 1:
                    pygame.draw.rect(self.DISPLAYSURF, WHITE, pos)
                if board[i][j] == 2:
                    pygame.draw.rect(self.DISPLAYSURF, GREEN, pos)
                if self.helper_board[i][j] == 1:
                    pygame.draw.rect(self.DISPLAYSURF, BLUE, pos)
                #if player_pos == (i,j):
        self.helper_board[player_pos[0],player_pos[1]] = 1
        pygame.draw.rect(self.DISPLAYSURF, RED, (player_pos[0]*B,player_pos[1]*B,B,B))
        
        pygame.display.update()
        
        
if __name__ == '__main__':
    game = maze_game(10,5,1000,1000)
    vis = visualizer(game,400)
    game.reset()
    for _ in range(10000):
        vis.draw()
        game.step(random.randrange(4))
        
        
        
        
        
        
