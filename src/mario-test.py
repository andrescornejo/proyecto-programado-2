import retro
import numpy as np
import cv2
import neat
import pickle

import os
os.environ['DISPLAY'] = ':1'

env = retro.make('SuperMarioBros-Nes', 'Level1-1')

imgarray = []

xpos_end = 0

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # get the first observation of the level.
        ob = env.reset()
        # Obtener la primera accion.
        ac = env.action_space.sample()

        # Obtener el tamano de la imagen y los colores.
        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 400:
                done = True
                #print(genome_id, fitness_current)
            
            genome.fitness = fitness_current

            # imgarray = np.delete(imgarray)


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'testconf')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open ('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)