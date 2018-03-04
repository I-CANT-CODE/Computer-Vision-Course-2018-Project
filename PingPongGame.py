import pygame
import numpy as np
import math
import random

UP = [1,0,0]
DONT_MOVE = [0,1,0]
DOWN = [0,0,1]

#COLORS
white = (255,255,255)
black = (0,0,0)

DIM = 320
Disp = pygame.display.set_mode((DIM,DIM))


class PongGame:
    def __init__(self, W_H, Mode):
        self.WIN_DIM = W_H
        self.PADDLE_W = 20
        self.PADDLE_H = 50
        self.BALL_DIM = 10
        self.MODE = Mode
        self.screen_shot = []
        #PADDLE LEFT
        self.PADDLE_LEFT_X = 0
        self.PADDLE_LEFT_Y_INIT = self.WIN_DIM/2
        self.PADDLE_LEFT_Y = self.PADDLE_LEFT_Y_INIT

        #PADDLE RIGHT
        self.PADDLE_RIGHT_X = self.WIN_DIM-self.PADDLE_W
        self.PADDLE_RIGHT_Y_INIT = self.WIN_DIM/2
        self.PADDLE_RIGHT_Y = self.PADDLE_RIGHT_Y_INIT

        #BALL VARIABLES
        self.BALL_X_INIT = self.BALL_Y_INIT = self.WIN_DIM/2
        self.BALL_X = self.BALL_X_INIT
        self.BALL_Y = self.BALL_Y_INIT

        #BALL VELOCITIES
        self.BALL_V_X = 2
        self.BALL_V_Y = 2

        #SPEEDS
        self.PADDLE_SPEED_L=self.PADDLE_SPEED_R = 18
        self.INIT_BALL_SPEED = 6
        self.BALL_SPEED = self.INIT_BALL_SPEED
        self.COLLISION_MARGIN = 10
        
        #init game
        
        self.L_POINTS = 0
        self.R_POINTS = 0
        
        self.L_REWARD_SIGNAL = 0
        self.R_REWARD_SIGNAL = 0
        
        self.frame_skipping = 4
        self.buffer = list()
        self.RenderFrame()

    def Move(self, PADDLE_LEFT_ACTION, PADDLE_RIGHT_ACTION ):
        
        if np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(UP):
            self.PADDLE_RIGHT_Y = self.PADDLE_RIGHT_Y-self.PADDLE_SPEED_R
            
        elif np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(DOWN):
            self.PADDLE_RIGHT_Y = self.PADDLE_RIGHT_Y+self.PADDLE_SPEED_R
            
        elif np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(DONT_MOVE):
            self.PADDLE_RIGHT_Y = self.PADDLE_RIGHT_Y
            
        if np.argmax(PADDLE_LEFT_ACTION)==np.argmax(UP):
            self.PADDLE_LEFT_Y = self.PADDLE_LEFT_Y-self.PADDLE_SPEED_L
            
        elif np.argmax(PADDLE_LEFT_ACTION)==np.argmax(DOWN):
            self.PADDLE_LEFT_Y = self.PADDLE_LEFT_Y+self.PADDLE_SPEED_L
            
        elif np.argmax(PADDLE_LEFT_ACTION)==np.argmax(DONT_MOVE):
            self.PADDLE_LEFT_Y = self.PADDLE_LEFT_Y
            
        self.BALL_X = self.BALL_X + self.BALL_V_X
        self.BALL_Y = self.BALL_Y + self.BALL_V_Y

        if self.PADDLE_RIGHT_Y >= self.WIN_DIM-self.PADDLE_H:
            self.PADDLE_RIGHT_Y = self.WIN_DIM-self.PADDLE_H
        if self.PADDLE_RIGHT_Y <= 0:
            self.PADDLE_RIGHT_Y = 0
            
        if self.PADDLE_LEFT_Y >= self.WIN_DIM-self.PADDLE_H:
            self.PADDLE_LEFT_Y = self.WIN_DIM-self.PADDLE_H
        if self.PADDLE_LEFT_Y <= 0:
            self.PADDLE_LEFT_Y = 0

    def CheckCollisions(self):
        LEFT_COLLISION = (self.BALL_X<(self.PADDLE_LEFT_X+self.PADDLE_W))&(self.BALL_X>self.PADDLE_LEFT_X)&((self.BALL_Y+self.BALL_DIM)>(self.PADDLE_LEFT_Y))&(self.BALL_Y<(self.PADDLE_LEFT_Y+self.PADDLE_H))
        RIGHT_COLLISION= (self.BALL_X>(self.PADDLE_RIGHT_X-self.BALL_DIM))&(self.BALL_X<(self.PADDLE_RIGHT_X+self.PADDLE_W))&((self.BALL_Y+self.BALL_DIM)>self.PADDLE_RIGHT_Y)&(self.BALL_Y<(self.PADDLE_RIGHT_Y+self.PADDLE_H))
        LEFT_PADDLE_FAIL = self.BALL_X+self.BALL_DIM<=0
        RIGHT_PADDLE_FAIL = self.BALL_X> self.WIN_DIM
        FLOOR_COLLISION = self.BALL_Y>(self.WIN_DIM-self.BALL_DIM)
        CEILING_COLLISION = self.BALL_Y<0
        if LEFT_COLLISION:
            #self.L_POINTS = self.L_POINTS+.1
            self.BALL_SPEED = self.BALL_SPEED + .1
            self.BALL_X = self.PADDLE_LEFT_X+self.PADDLE_W
            
            BALL_PADDLE_LEFT_COORDINATE = self.BALL_Y + self.BALL_DIM/2 - self.PADDLE_LEFT_Y
            if BALL_PADDLE_LEFT_COORDINATE < 0:
                BALL_PADDLE_LEFT_COORDINATE = 0
            if BALL_PADDLE_LEFT_COORDINATE > self.PADDLE_H:
                BALL_PADDLE_LEFT_COORDINATE = self.PADDLE_H
            #convert from [0,70] to [1.309,-1.309]
            G = BALL_PADDLE_LEFT_COORDINATE/self.PADDLE_H 
            BALL_PADDLE_LEFT_COORDINATE = .8*(1-G)-.8*(G)
            
            self.BALL_V_X = self.BALL_SPEED*math.cos(BALL_PADDLE_LEFT_COORDINATE)
            self.BALL_V_Y = self.BALL_SPEED*-math.sin(BALL_PADDLE_LEFT_COORDINATE)

        if RIGHT_COLLISION:
            self.BALL_SPEED = self.BALL_SPEED + .1
            self.BALL_X = self.PADDLE_RIGHT_X-self.BALL_DIM
            
            BALL_PADDLE_RIGHT_COORDINATE = self.BALL_Y + self.BALL_DIM/2 - self.PADDLE_RIGHT_Y
            if BALL_PADDLE_RIGHT_COORDINATE < 0:
                BALL_PADDLE_RIGHT_COORDINATE = 0
            if BALL_PADDLE_RIGHT_COORDINATE > self.PADDLE_H:
                BALL_PADDLE_RIGHT_COORDINATE = self.PADDLE_H
            #convert from [0,70] to [1.8326,4.45059]
            G = BALL_PADDLE_RIGHT_COORDINATE/self.PADDLE_H 
            BALL_PADDLE_RIGHT_COORDINATE = .8*(1-G)-.8*(G)

            
            self.BALL_V_X = self.BALL_SPEED*-math.cos(BALL_PADDLE_RIGHT_COORDINATE)
            self.BALL_V_Y = self.BALL_SPEED*-math.sin(BALL_PADDLE_RIGHT_COORDINATE)
        if CEILING_COLLISION:
            
            self.BALL_Y = 0
            self.BALL_V_Y = self.BALL_V_Y * -1
        if FLOOR_COLLISION:
            self.BALL_Y = self.WIN_DIM-self.BALL_DIM
            self.BALL_V_Y = self.BALL_V_Y * -1
        if LEFT_PADDLE_FAIL:
            self.BALL_SPEED = self.INIT_BALL_SPEED
            self.PADDLE_LEFT_Y = self.PADDLE_RIGHT_Y = self.WIN_DIM/2
            self.BALL_X = self.WIN_DIM/5
            self.BALL_Y = self.WIN_DIM/2
            rand_theta = random.uniform(-.8,.8)
            self.BALL_V_X = self.BALL_SPEED*math.cos(rand_theta)
            self.BALL_V_Y = self.BALL_SPEED*-math.sin(rand_theta)
            self.R_POINTS = self.R_POINTS + 1
            
            
            
        if RIGHT_PADDLE_FAIL:
            self.BALL_SPEED = self.INIT_BALL_SPEED
            self.PADDLE_LEFT_Y = self.PADDLE_RIGHT_Y = self.WIN_DIM/2
            self.BALL_X = self.WIN_DIM*4/5
            rand_theta = random.uniform(-.8,.8)
            self.BALL_V_X = self.BALL_SPEED*-math.cos(rand_theta)
            self.BALL_V_Y = self.BALL_SPEED*-math.sin(rand_theta)
            self.BALL_Y = self.WIN_DIM/2
            self.L_POINTS = self.L_POINTS + 1
            
           
            
    def RenderFrame(self):
        Disp.fill(black)#fill black background
        pygame.draw.rect(Disp, white, [self.PADDLE_RIGHT_X,self.PADDLE_RIGHT_Y,self.PADDLE_W,self.PADDLE_H])#draw first paddle
        pygame.draw.rect(Disp, white, [self.PADDLE_LEFT_X,self.PADDLE_LEFT_Y,self.PADDLE_W,self.PADDLE_H])#draw first paddle
        pygame.draw.rect(Disp, white, [self.BALL_X,self.BALL_Y,self.BALL_DIM,self.BALL_DIM])#draw ball
        pygame.display.update()

        if self.MODE == "pixels":
        #get screen shot
            self.screen_shot = pygame.surfarray.array3d(Disp)
            self.screen_shot = self.screen_shot[1::5,1::5,:]
            self.screen_shot = self.screen_shot[:,:,0]
            self.screen_shot = np.divide(self.screen_shot,255)

    def Run4Frames(self, AL, AR):
        IPL = self.L_POINTS;
        IPR = self.R_POINTS;

        if self.MODE == "location":
            paddle_R_tracking =[0,0,0,0]
            paddle_L_tracking =[0,0,0,0]
            ball_x_tracking = [0,0,0,0]
            ball_y_tracking = [0,0,0,0]
        if self.MODE == "pixels":
            self.buffer =list()
            
        for i in range (self.frame_skipping):
            #clock.tick(60)
            self.Move(AL, AR)
            self.CheckCollisions()
            self.RenderFrame()
            if self.MODE == "location":
                paddle_R_tracking[i] = self.PADDLE_RIGHT_Y
                paddle_L_tracking[i] = self.PADDLE_LEFT_Y
                ball_x_tracking[i] = self.BALL_X
                ball_y_tracking[i] = self.BALL_Y
                
            if self.MODE == "pixels":
                self.buffer.append(self.screen_shot)
                
            
        L_rew = (self.L_POINTS-IPL)-(self.R_POINTS-IPR)
        R_rew = (self.R_POINTS-IPR)-(self.L_POINTS-IPL)
        

        if self.MODE == "location":
            return  L_rew,R_rew, [paddle_L_tracking,paddle_R_tracking,ball_x_tracking,ball_y_tracking]
        else:
            return L_rew, R_rew, [self.buffer[3]-.5*self.buffer[0]]
    def QUITGAME(self):
        pygame.quit()
        quit()
            

        

        
        

        
        
        
