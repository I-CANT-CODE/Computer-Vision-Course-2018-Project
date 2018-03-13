import PingPongGame as PONG
import numpy as np
import math
import random
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt

clock = pygame.time.Clock()
fstring = input("enter a name for the reward data file:")
results_file = open("./"+fstring,'w')
BATCH_SIZE = 32
ALPHA = 1e-4
GAMMA = .99
EPSILON = .97
rewards_array = list()

MODEL_L_STR = input ("enter the name of a model (left side player)")
#MODEL_R_STR = input ("enter the name of a model (right side player)")

LOAD_MODEL = False

def GRAPH_REWARDS(REW_ARRAY):
        plt.plot(REW_ARRAY)
        plt.show()
        
def ONE_HOT_ACTIONS(array):
        maxI = np.argmax(array)
        new_array = [0,0,0]
        new_array[maxI] = 1
        return new_array

def RANDOM_ONE_HOT():
        new_array = [0,0,0]
        new_array[random.randint(0,2)] = 1
        return new_array


def NN(x, reuse = False):
    x = tf.layers.conv2d(x, filters=32, kernel_size=(8, 8), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_1', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_2', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides = 1, padding='same', activation=tf.nn.relu, name='conv2d_5', reuse=reuse)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,units = 512, activation = tf.nn.relu, name = 'fullyconected', reuse = reuse)
    Action_Vals = tf.layers.dense(x,units = 3, name = 'FC7', reuse = reuse)
    return Action_Vals

session = tf.Session()
session.run(tf.global_variables_initializer())

State_InL = tf.placeholder(tf.float32, shape = [None, 1,64,64])
with tf.variable_scope("paddleL"):
    Q_L = NN(State_InL, reuse = False)
saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='paddleL'))
saver1.restore(session, MODEL_L_STR)


Game = PONG.PongGame(320, "pixels")




AVL=AVR = [1,0,0]
L,R, STATE = Game.Run4Frames(AVL,AVR)
AVL=AVR=5
NUM_ROUNDS_PLAYED = 0
time_step = 0
LRewSUM=0
RRewSUM=0
training_data = list()
RANDOM_FACTOR=0

temp = 0
oldREWSUM = 0
while (1):
        clock.tick(30)
        for event in pygame.event.get():
            #print(event)
                    
            if event.type == pygame.QUIT:#for exiting the game
                    Game.QUITGAME()
        #here I model the built in player:------------------------------------------------------------------------
        
        if time_step%25 ==0:
                RANDOM_FACTOR=random.randint(-25,25)
                #print(RANDOM_FACTOR)
                
        if Game.BALL_Y+Game.BALL_DIM/2 > Game.PADDLE_RIGHT_Y+25+RANDOM_FACTOR:
                AVR=[0,0,1]
        else:
                AVR=[1,0,0]
        #----------------------------------------------------------------------------------------------------------
        

        
        AVL = session.run(Q_L, feed_dict = {State_InL: [STATE]})
        AVL = ONE_HOT_ACTIONS(AVL)
        
        L,R, STATE = Game.Run4Frames(AVL,AVR)
        LRewSUM = L+LRewSUM
        RRewSUM = R+RRewSUM
        print(LRewSUM,RRewSUM)



        
