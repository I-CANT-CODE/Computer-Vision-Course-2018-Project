import PingPongGame as PONG
import numpy as np

Game = PONG.PongGame(320, "location")

while (1):
        L,R, DATA = Game.Run4Frames([1,0,0],[0,0,1])
        print (L,R, [DATA])
        #x = input()

