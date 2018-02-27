import PingPongGame as PONG
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

Game = PONG.PongGame(320, "pixels")

while (1):
        L,R, DATA = Game.Run4Frames([1,0,0],[0,0,1])
        print (L,R)
        print(np.shape(DATA))
        plt.imshow(DATA[0])
        plt.show()

