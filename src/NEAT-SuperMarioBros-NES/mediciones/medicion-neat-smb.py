import retro
import csv
import numpy as np
import cv2
import neat
import pickle
import time
import os
os.environ['DISPLAY'] = ':1'

DEBUG = False
fitness_goal = 700

asig = 0 # Variable que contiene las asignaciones del algoritmo.
comp = 0 # Variable que contiene las comparaciones del algoritmo.

#Crear el ambiente de gym-retro
env = retro.make('SuperMarioBros-Nes', 'Level1-1') 
asig += 1

def downscale_image(ob, size_x, size_y):
    """Baja la resolucion de la pantalla para alistarla para la red neuronal."""
    global asig

    ob = cv2.resize(ob, (size_x, size_y))
    asig += 1 # Asignacion por funcion externa.  

    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    asig += 1 # Asignacion por funcion externa.  

    ob = np.reshape(ob, (size_x, size_y))
    asig += 1 # Asignacion por funcion externa.  

    imgarray = np.ndarray.flatten(ob)
    asig += 1 # Asignacion por funcion externa.  
    return imgarray

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        global asig
        global comp

        ob = env.reset()
        asig += 1 # Asignacion por funcion externa.  
        
        size_x, size_y, img_color= env.observation_space.shape
        asig+=3 # Tres asignaciones por las tres variables.

        size_x = int(size_x/8)
        size_y = int(size_y/8)
        asig += 2 # Dos asignaciones por las dos variables.

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        asig += 1 # Asignacion por funcion externa.  

        current_max_fitness = 0
        fitness_current = 0
        stag_counter = 0
        asig+=3 # Tres asignaciones por las tres variables.

        xpos_lo = 0
        xpos_hi = 0
        asig += 2 # Dos asignaciones por las dos variables.

        done = False
        won = False
        asig += 2 # Dos asignaciones por las dos variables.

        while not done:
            if DEBUG:
                env.render()
            comp += 1

            imgarray = downscale_image(ob, size_x, size_y)
            asig += 1 # Asignacion por funcion externa.  

            nnOutput = net.activate(imgarray)
            asig += 1 # Asignacion por funcion externa.  

            ob, rew, done, info = env.step(nnOutput)
            asig += 4 # Cuatro asignaciones por cuatro variables.  

            xpos_lo = info['xscrollLo']
            asig += 1 # Asignacion por funcion externa.  

            next_xpos_hi = info['xscrollHi']
            asig += 1 # Asignacion por funcion externa.  

            if(next_xpos_hi > xpos_hi):
                xpos_hi = next_xpos_hi
                asig += 1 # Asignacion por una variable.
            comp += 1 # Comparacion por el if.

            if fitness_current >= fitness_goal and not won:
                fitness_current *= 100000
                won = True
                asig += 2 # Dos asignaciones por dos variables.
            else:
                fitness_current += rew
                asig += 1 # Asignacion por una variable.
            comp += 1 # Comparacion por el if/else.

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                stag_counter = 0
                asig += 2 # Dos asignaciones por dos variables.
            else:
                stag_counter += 1
                asig += 1 # Asignacion por una variable.
            comp += 1 # Comparacion por el if/else.

            if stag_counter > 250:
                done = True
                asig += 1 # Asignacion por una variable.
            comp += 1 # Comparacion por el if.
            
            genome.fitness = fitness_current
            asig += 1 # Asignacion por una variable.

        if DEBUG:
            print('Genome ID:', genome_id, 'Genome Fitness:', fitness_current)
            print('xpos_lo:', xpos_lo, 'xpos_hi:', xpos_hi)
        comp += 1 # Comparacion por el if.

start = time.process_time()
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neural-config')
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
winner = p.run(eval_genomes)

total_time = time.process_time() - start

asig += 7 # Asignaciones de las 7 variables anteriores.

# Guardar el algoritmo ganador en un archivo .pkl
with open ('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

row = [fitness_goal, asig, comp, total_time]
fields = ['fitness_goal', 'asignaciones', 'comparaciones', 'tiempo']  
filename = "medicion_resultados.csv"

print(fields)
print(row)

if os.path.isfile(filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
else:
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerow(row)