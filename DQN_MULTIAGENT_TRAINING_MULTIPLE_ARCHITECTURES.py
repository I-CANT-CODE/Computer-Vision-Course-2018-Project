import PingPongGame as PONG
import numpy as np
import math
import random
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt
import MyFunctions

clock = pygame.time.Clock()
fstring = input("enter a name for the reward data file:")
results_file = open("./"+fstring,'w')
BATCH_SIZE = 32
ALPHA = 1e-4
GAMMA = .99
EPSILON = .97
rewards_array_L = list()
rewards_array_R = list()
rounds_array = list()
MODEL_L_STR = "DUELING_DQN_LEFT_MODEL-500000"
MODEL_R_STR = "DQN_RIGHT_MODEL-500000"
model_string= "MULTI_AGENT_TRAIN_FROM_0_MODEL"
NUM_ROUNDS_PLAYED = 0

LOAD_MODEL = False

session = tf.Session()
session.run(tf.global_variables_initializer())

State_InL = tf.placeholder(tf.float32, shape = [None, 1,64,64])
with tf.variable_scope("paddleL"):
    Q_L = MyFunctions.DUELING_DQN(State_InL, reuse = False)

State_InR = tf.placeholder(tf.float32, shape = [None, 1,64,64])
with tf.variable_scope("paddleR"):
    Q_R = MyFunctions.DQN(State_InR, reuse = False)

GT_L = tf.placeholder(tf.float32, shape = [BATCH_SIZE])
Action_Placeholder_L = tf.placeholder(tf.float32,shape = [BATCH_SIZE,3])
approximation_L = tf.reduce_sum(tf.multiply(Action_Placeholder_L,Q_L),1)
Loss_L = tf.reduce_mean(tf.square(GT_L-approximation_L))
train_step_L = tf.train.AdamOptimizer(ALPHA).minimize(Loss_L)

GT_R = tf.placeholder(tf.float32, shape = [BATCH_SIZE])
Action_Placeholder_R = tf.placeholder(tf.float32,shape = [BATCH_SIZE,3])
approximation_R = tf.reduce_sum(tf.multiply(Action_Placeholder_R,Q_R),1)
Loss_R = tf.reduce_mean(tf.square(GT_R-approximation_R))
train_step_R = tf.train.AdamOptimizer(ALPHA).minimize(Loss_R)

session.run(tf.global_variables_initializer())
'''
saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='paddleL'))
saver1.restore(session, MODEL_L_STR)

saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='paddleR'))
saver2.restore(session, MODEL_R_STR)
'''


saver = tf.train.Saver()
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
        clock.tick(60)
        for event in pygame.event.get():
            #print(event)
                    
            if event.type == pygame.QUIT:#for exiting the game
                    Game.QUITGAME()
            if event.type == pygame.KEYDOWN:
                    if event.key ==pygame.K_DOWN:
                            MyFunctions.GRAPH_REWARDS(rewards_array_L, rewards_array_R, rounds_array)

        
        if np.random.binomial(1,EPSILON):
                #print (np.shape(STATE))
                AVR = session.run(Q_R, feed_dict = {State_InR: [STATE]})
                AVR = MyFunctions.ONE_HOT_ACTIONS(AVR)
        else:
                AVR = MyFunctions.RANDOM_ONE_HOT()

        
        if np.random.binomial(1,EPSILON):
                #print (np.shape(STATE))
                AVL = session.run(Q_L, feed_dict = {State_InL: [STATE]})
                AVL = MyFunctions.ONE_HOT_ACTIONS(AVL)
        else:
                AVL = MyFunctions.RANDOM_ONE_HOT()

        OLD_STATE=np.copy(STATE)
        L,R, STATE = Game.Run4Frames(AVL,AVR)
        if (L>0)|(R>0):
                NUM_ROUNDS_PLAYED = NUM_ROUNDS_PLAYED + 1;
        LRewSUM = L+LRewSUM
        RRewSUM = R+RRewSUM
        
        #Fprint(LRewSUM,RRewSUM)
        training_data.append([np.copy(OLD_STATE),L,R,AVL[:],AVR[:], np.copy(STATE)])

        if time_step%5000 == 0:
                
                line = str(time_step)+','+ str(LRewSUM)+ ',' + str(RRewSUM)+ ',' + str(NUM_ROUNDS_PLAYED)+';'
                rewards_array_L.append([LRewSUM])
                rewards_array_R.append([RRewSUM])
                rounds_array.append([NUM_ROUNDS_PLAYED])
                print(line)
                results_file.write(line)
                NUM_ROUNDS_PLAYED = 0
                LRewSUM = 0
                RRewSUM = 0

        if (time_step%50000==0)&(time_step<250000):
                saver.save(session, './'+model_string, global_step = time_step)
                
        if time_step>200000:
                training_data.pop(0)
        if time_step == 500000:
                results_file.close()
                Game.QUITGAME()
        time_step = time_step + 1
        if (time_step>1000)&(time_step%4==0):
                
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
                target = session.run(Q_R,feed_dict = {State_InR:SN_})#get q values of next state (Q')
                #print(target)
                target_ = [None]*len(batch)
                #print(target_)
                for i in range(len(batch)):
                        target_[i] = max(target[i])#get max values of Q' values of next state
                #print(target_)
                target_ = [i*GAMMA for i in target_]#discount future rewards of Q'
                #print("target * GAMMA ",target_)
                target_F = [j+i for i,j in zip(RR_,target_)]#now we have our target value of r + max(Q')
                #print(target_F)
                #[print(i-j) for i,j in zip(target_F,target_)]
                #x = input()
                #print(np.shape(target_),np.shape(AL_))
                #print("1.5")
                session.run(train_step_R, feed_dict = {GT_R:target_F, Action_Placeholder_R:AR_, State_InR:SO_})
        
                #print("2. here?")
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
        



        
