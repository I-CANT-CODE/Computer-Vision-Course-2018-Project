import PingPongGame as PONG
import numpy as np
import math
import random
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt
config=tf.ConfigProto(device_count={'GPU': 0})
import MyFunctions

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

clock = pygame.time.Clock()
fstring = "DQN_RIGHT_REWARDS"
results_file = open("./"+fstring,'w')
BATCH_SIZE = 32
ALPHA = 1e-4
GAMMA = .99
EPSILON = .97
rewards_array = list()

LOAD_MODEL = False


#make 2 graphs
State_InR = tf.placeholder(tf.float32, shape = [None, 1,64,64])
with tf.variable_scope("paddleR"):
    Q_R = MyFunctions.DQN(State_InR, reuse = False)

#define loss function for left side player
GT_R = tf.placeholder(tf.float32, shape = [BATCH_SIZE])
Action_Placeholder_R = tf.placeholder(tf.float32,shape = [BATCH_SIZE,3])
approximation_R = tf.reduce_sum(tf.multiply(Action_Placeholder_R,Q_R),1)
Loss_R = tf.reduce_mean(tf.square(GT_R-approximation_R))
train_step_R = tf.train.AdamOptimizer(ALPHA).minimize(Loss_R)

Game = PONG.PongGame(320, "pixels")
Game.PADDLE_SPEED_L = 4
saver = tf.train.Saver()
session = tf.Session()
session.run(tf.global_variables_initializer())

model_string = "DQN_RIGHT_MODEL"
print("press down key at any time to check reward progress on graph")



#initializing stuff
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
        #print('playing')
        clock.tick(10)
        
        if np.random.binomial(1,EPSILON):
                #print (np.shape(STATE))
                AVR = session.run(Q_R, feed_dict = {State_InR: [STATE]})
                AVR = MyFunctions.ONE_HOT_ACTIONS(AVR)
        else:
                AVR = MyFunctions.RANDOM_ONE_HOT()

#here I model the built in player:------------------------------------------------------------------------
        
        if time_step%25 ==0:
                MyFunctions.RANDOM_FACTOR=random.randint(-35,35)
                #print(RANDOM_FACTOR)
                
        if Game.BALL_Y+Game.BALL_DIM/2 > Game.PADDLE_LEFT_Y+35+RANDOM_FACTOR:
                AVL=[0,0,1]
        else:
                AVL=[1,0,0]
#----------------------------------------------------------------------------------------------------------
        
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
            if event.type == pygame.KEYDOWN:
                if event.key ==pygame.K_DOWN:
                    MyFunctions.GRAPH_REWARDS(rewards_array)
                    
            if event.type == pygame.QUIT:#for exiting the game
                results_file.close()
                Game.QUITGAME()
                

        if time_step%5000 == 0:
                line = str(time_step)+','+ str(RRewSUM)+';'
                rewards_array.append(RRewSUM)
                print(line)
                results_file.write(line)
                temp = NUM_ROUNDS_PLAYED
                LRewSUM = 0
                RRewSUM = 0
                
        if time_step%100000==0:
                saver.save(session, './'+model_string, global_step = time_step)
                
                

        #training time
        # only start training after at least 1000 things in QUE
        if time_step>200000:
                training_data.pop(0)
        if time_step == 500000:
                results_file.close()
                Game.QUITGAME()
                
        
        
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
                session.run(train_step_R, feed_dict = {GT_R:target_F, Action_Placeholder_R:AR_, State_InR:SO_})
        

