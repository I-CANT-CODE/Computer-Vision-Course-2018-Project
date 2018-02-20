import PingPongGame as PONG

Game = PONG.PongGame(320, "location")

while (1):
        R, L, DATA = Game.Run4Frames([1,0,0],[0,0,1])
        print (R,L, DATA)
        x = input()

