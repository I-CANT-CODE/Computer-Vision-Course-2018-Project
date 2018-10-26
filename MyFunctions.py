import PingPongGame as PONG
import numpy as np
import math
import random
import tensorflow as tf
import pygame
import matplotlib.pyplot as plt

def GRAPH_REWARDS(REW_ARRAY_L, REW_ARRAY_R, ROUNDS):
        plt.plot(REW_ARRAY_L)
        plt.plot(REW_ARRAY_R)
        plt.plot(ROUNDS)
        plt.legend(['Left Rewards (Dueling DQN)','Right Rewards (DQN)','num rounds played'])
        plt.show()

def GRAPH_REWARDS_1(REW_ARRAY):
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


def DQN(x, reuse = False):
    x = tf.layers.conv2d(x, filters=32, kernel_size=(8, 8), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_1', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_2', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides = 1, padding='same', activation=tf.nn.relu, name='conv2d_5', reuse=reuse)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,units = 512, activation = tf.nn.relu, name = 'fullyconected', reuse = reuse)
    Action_Vals = tf.layers.dense(x,units = 3, name = 'FC7', reuse = reuse)
    return Action_Vals

def DUELING_DQN(x, reuse = False):
    x = tf.layers.conv2d(x, filters=32, kernel_size=(8, 8), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_1', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(5, 5), strides = 2, padding='same', activation=tf.nn.relu, name='conv2d_2', reuse=reuse)
    x = tf.layers.conv2d(x, filters=64, kernel_size=(3, 3), strides = 1, padding='same', activation=tf.nn.relu, name='conv2d_5', reuse=reuse)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x,units = 521, activation = tf.nn.relu, name = 'fullyconected', reuse = reuse)
    x_A_1 = tf.layers.dense(x,units = 256, activation = tf.nn.relu, name = 'A1', reuse = reuse)
    x_A_2 = tf.layers.dense(x_A_1,units = 3, name = 'A2', reuse = reuse)
    print(x_A_2)
    x_A_2_B = tf.reduce_mean(x_A_2,1)
    print(x_A_2)
    x_A_3 = tf.stack([x_A_2_B,x_A_2_B,x_A_2_B],axis = 1)
    print(x_A_3)
    x_A_4 = tf.subtract(x_A_2, x_A_3)
    print(x_A_4)
    x_V_1 = tf.layers.dense(x,units = 256, activation = tf.nn.relu, name = 'V1', reuse = reuse)
    x_V_2 = tf.layers.dense(x_V_1,units = 1, name = 'V2', reuse = reuse)
    x_V_3 = tf.stack([x_V_2,x_V_2,x_V_2],axis = 1)
    Action_Vals = tf.add(x_A_4, x_A_3)
    return Action_Vals
