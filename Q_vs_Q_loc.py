import PingPongGame as PONG
import numpy as np
import math
import random
import ONE_HOT
import NN
import tensorflow as tf
import pygame

clock = pygame.time.Clock()

results_file = open("./locational_Multiagent_results_file",'w')
BATCH_SIZE = 32
ALPHA = 1e-6
GAMMA = .99
EPSILON = .97

LOAD_MODEL = False

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
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,units = 128,activation = tf.nn.relu,  name = 'FC1', reuse = reuse)
    x = tf.layers.dense(x,units = 256,activation = tf.nn.relu, name = 'FC2', reuse = reuse)
    x = tf.layers.dense(x,units = 512,activation = tf.nn.relu,  name = 'FC3', reuse = reuse)
    x = tf.layers.dense(x,units = 512,activation = tf.nn.relu,  name = 'FC4', reuse = reuse)
    x = tf.layers.dense(x,units = 256,activation = tf.nn.relu, name = 'FC5', reuse = reuse)
    x = tf.layers.dense(x,units = 128,activation = tf.nn.relu, name = 'FC6', reuse = reuse)
    Action_Vals = tf.layers.dense(x,units = 3, name = 'FC7', reuse = reuse)
    return Action_Vals

#make 2 graphs
State_InL = tf.placeholder(tf.float32, shape = [None, 4,4])
with tf.variable_scope("paddleL"):
    Q_L = NN(State_InL, reuse = False)

State_InR = tf.placeholder(tf.float32, shape = [None, 4,4])
with tf.variable_scope("paddleR"):
    Q_R = NN(State_InR, reuse = False)


#define loss function for left side player
GT_L = tf.placeholder(tf.float32, shape = [BATCH_SIZE])
Action_Placeholder_L = tf.placeholder(tf.float32,shape = [BATCH_SIZE,3])
approximation_L = tf.reduce_sum(tf.multiply(Action_Placeholder_L,Q_L),1)
Loss_L = tf.reduce_mean(tf.square(GT_L-approximation_L))
train_step_L = tf.train.AdamOptimizer(ALPHA).minimize(Loss_L)

#then the right
GT_R = tf.placeholder(tf.float32, shape = [BATCH_SIZE])
Action_Placeholder_R = tf.placeholder(tf.float32,shape = [BATCH_SIZE,3])
approximation_R = tf.reduce_sum(tf.multiply(Action_Placeholder_R,Q_R),1)
Loss_R = tf.reduce_mean(tf.square(GT_R-approximation_R))
train_step_R = tf.train.AdamOptimizer(ALPHA).minimize(Loss_R)

Game = PONG.PongGame(320, "location")
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())
if LOAD_MODEL == True:
        saver.restore(session, 'Mult_Ag_Loc_Models')

#initializing stuff
AVL=AVR = [1,0,0]
L,R, STATE = Game.Run4Frames(AVL,AVR)
AVL=AVR=5
NUM_ROUNDS_PLAYED = 0
time_step = 0
LRewSUM=0
RRewSUM=0
training_data = list()


temp = 0

while (1):
        #print('playing')
        
        if np.random.binomial(1,EPSILON):
                AVL = session.run(Q_L, feed_dict = {State_InL: [STATE]})
                AVL = ONE_HOT_ACTIONS(AVL)
                
        else:
                AVL = RANDOM_ONE_HOT()
        
        if Game.BALL_Y > Game.PADDLE_RIGHT_Y+35:
                AVR=[0,0,1]
        else:
                AVR=[1,0,0]

        '''if np.random.binomial(1,EPSILON):
                AVR = session.run(Q_R, feed_dict = {State_InR: [STATE]})
                AVR = ONE_HOT_ACTIONS(AVR)
        else:
                AVR = RANDOM_ONE_HOT()'''
        
        OLD_STATE=np.copy(STATE)
        L,R, STATE = Game.Run4Frames(AVL,AVR)
        
        #print(AVR, AVL)
        LRewSUM = LRewSUM + L
        RRewSUM = RRewSUM + R
        #print([L,R,AVL,AVR, DATA])
        training_data.append([np.copy(OLD_STATE),L,R,AVL[:],AVR[:], np.copy(STATE)])
       
        if (L!=0):
                NUM_ROUNDS_PLAYED = NUM_ROUNDS_PLAYED + 1
                
        
        time_step = time_step + 1
        
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:#for exiting the game
                gameExit = True

        if time_step%5000 == 0:
                line = str(time_step)+','+ str(LRewSUM)+','+ str(RRewSUM)+ ','+str(NUM_ROUNDS_PLAYED-temp)+';'
                print(line)
                results_file.write(line)
                temp = NUM_ROUNDS_PLAYED

        #training time
        # only start training after at least 1000 things in QUE
        if time_step>100000:
                training_data.pop(0)
                
        
                
        if time_step>1000:
                if (time_step%50==0):
                        if (GAMMA <.98):
                                GAMMA = GAMMA + .1
                                if GAMMA>.99:
                                        GAMMA = .99
                
                                
                #train
                batch = random.sample(training_data, BATCH_SIZE)
                SO_ = [item[0] for item in batch]
                RL_ = [item[1] for item in batch]
                RR_ = [item[2] for item in batch]
                AL_ = [item[3] for item in batch]
                AR_ = [item[4] for item in batch]
                SN_ = [item[5] for item in batch]
                #print(np.shape(SN_), np.shape(SO_))

                #x = input()
                #TRAIN LEFT SIDE FIRST
                target = session.run(Q_L,feed_dict = {State_InL:SN_})#get q values of next state (Q')
                #print(target)
                target_ = [None]*len(batch)
                #print(target_)
                for i in range(len(batch)):
                        target_[i] = max(target[i])#get max values of Q' values of next state
                #print(target_)
                target_ = [i*GAMMA for i in target_]#discount future rewards of Q'
                #print("target * GAMMA ",target_)
                target_F = [j+i for i,j in zip(RL_,target_)]#now we have our target value of r + max(Q')
                #print(target_F)
                #[print(i-j) for i,j in zip(target_F,target_)]
                #x = input()
                #print(np.shape(target_),np.shape(AL_))
                session.run(train_step_L, feed_dict = {GT_L:target_F, Action_Placeholder_L:AL_, State_InL:SO_})
                '''
                batch = random.sample(training_data, BATCH_SIZE)
                SO_ = [item[0] for item in batch]
                RL_ = [item[1] for item in batch]
                RR_ = [item[2] for item in batch]
                AL_ = [item[3] for item in batch]
                AR_ = [item[4] for item in batch]
                SN_ = [item[5] for item in batch]           
                #TRAIN RIGHT SIDE
                target = session.run(Q_R,feed_dict = {State_InR:SN_})#get q values of next state (Q')
                target_ = [None]*len(batch)
                for i in range(len(batch)):
                        target_[i] = max(target[i])#get max values of Q' values of next state                        
                target_ = [i*GAMMA for i in target_]#discount future rewards of Q'
                target_F = [j+i for i,j in zip(RR_,target_)]#now we have our target value of r + max(Q')
                session.run(train_step_R, feed_dict = {GT_R:target_F, Action_Placeholder_R:AR_, State_InR:SO_})
        '''
                
        
        

