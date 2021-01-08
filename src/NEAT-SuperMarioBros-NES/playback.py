import retro
import numpy as np
import cv2 
import neat
import pickle
import time
import os
os.environ['DISPLAY'] = ':1'

env = retro.make('SuperMarioBros-Nes', 'Level1-1') 
# env = retro.make('SuperMarioBros-Nes', 'Level1-1', record=True) 

imgarray = []

xpos_end = 0



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neural-config')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_lo = 0
xpos_hi = 0
xpos_max = 0
stag_counter = 0

done = False

while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    for x in ob:
        for y in x:
            imgarray.append(y)
    
    nnOutput = net.activate(imgarray)
    
    ob, rew, done, info = env.step(nnOutput)
    imgarray.clear()
    
    xpos_lo = info['xscrollLo']

    next_xpos_hi = info['xscrollHi']

    if(next_xpos_hi > xpos_hi):
        xpos_hi = next_xpos_hi

    if xpos_hi >= 12 and not won:
        fitness_current *= xpos_hi
    else:
        fitness_current += rew

    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        stag_counter = 0
    else:
        stag_counter += 1

    if stag_counter > 250:
        done = True
    