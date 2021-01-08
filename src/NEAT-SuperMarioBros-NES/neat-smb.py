import retro
import numpy as np
import cv2
import neat
import pickle
import os
os.environ['DISPLAY'] = ':1'

# Variable debug: utilizada para imprimir la pantalla del NES, y ver la posicion de Mario en el eje x.
DEBUG = True
#DEBUG = False

#Crear el ambiente de gym-retro
env = retro.make('SuperMarioBros-Nes', 'Level1-1') 

def downscale_image(ob, size_x, size_y):
    """Baja la resolucion de la pantalla para alistarla para la red neuronal."""
    # Cambiar el tamano de la imageen a las dimensiones especificadas.
    ob = cv2.resize(ob, (size_x, size_y))
    # Cambiar la imagen a blano y negro.
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    # Convertir la imagen en una matriz de dos dimensiones.
    ob = np.reshape(ob, (size_x, size_y))
    # Convertir el la matriz en un arreglo unidimensional.
    imgarray = np.ndarray.flatten(ob)
    return imgarray

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Obtener la primera observacion del nivel.
        ob = env.reset()

        # Obtener el tamano de la imagen y los colores.
        size_x, size_y, img_color= env.observation_space.shape

        # Definir el tamano de la imagen reducida. (1/8 de la original)
        size_x = int(size_x/8)
        size_y = int(size_y/8)

        # Crear la red neuronal recurrente.
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        # Variable para guardar el fitness maximo actual.
        current_max_fitness = 0
        # Variable para guardar el fitness de cada red neuronal.
        fitness_current = 0
        # Variable para reconocer si la red neuronal se ha quedado estancada.
        stag_counter = 0

        # Posicion de Mario en el eje x. (255 es su valor maximo)
        xpos_lo = 0
        # Cuadrante del nivel en el que se encuentra Mario.
        # Mario gana si se encuentra en el cuadrante 12.
        xpos_hi = 0

        # Condicion de fin de simulacion.
        done = False
        # Condicion de si el nivel ha sido completado.
        won = False

        while not done:
            if DEBUG:
                env.render()

            # Conseguir la observacion actual y reducirla en tamano.
            imgarray = downscale_image(ob, size_x, size_y)

            # Conseguir las siguientes entradas al control del NES de la red neuronal.
            nnOutput = net.activate(imgarray)

            # Efectuar las entradas dadas al emulador.
            ob, rew, done, info = env.step(nnOutput)

            # Conseguir la posicion x de Mario del RAM del emulador.
            xpos_lo = info['xscrollLo']

            # Conseguir el cuadrante en el que se encuentra Mario de la RAM del emulador.
            next_xpos_hi = info['xscrollHi']

            # Mantener el valor maximo del cuadrante en caso de overflow.
            if(next_xpos_hi > xpos_hi):
                xpos_hi = next_xpos_hi

            if xpos_hi >= 12 and not won:
                # En caso que el nivel sea ganado, suministrar una recompensa a la red neuronal.
                # Esta recompensa garantiza que la red se va a dar cuenta que resolvio el problema.
                fitness_current *= xpos_hi
                won = True
            else:
                # En caso de no ganar, solo suministrar la recompensa normal.
                fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                # Si el fitness actual es mas alto que el maximo, reiniciar el contador de estancado.
                stag_counter = 0
            else:
                # Si la red no tiene fitness mayor, incrementar el contador de estancado.
                stag_counter += 1

            if stag_counter > 250:
                # Si la red ha estado estancada por 250 iteraciones, terminar la simulacion.
                done = True
            
            # Asignarle el fitness al genoma.
            genome.fitness = fitness_current

        if DEBUG:
            print('Genome ID:', genome_id, 'Genome Fitness:', fitness_current)
            print('xpos_lo:', xpos_lo, 'xpos_hi:', xpos_hi)


# Configurar la instancia de la red neuronal.
# La configuracion de la red neuronal se encuentra en el archivo de 'neural config'.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neural-config')

# Si se quiere empezar a entrenar desde cero, utilizar la siguiente linea.
#p = neat.Population(config)

# Si se quiere empezar a entrenar desde cierto punto, utilizar la siguiente linea.
p = neat.checkpoint.Checkpointer.restore_checkpoint('winner/winner_run')

# Imprimir la estadisticas generacionales y generar checkpoints.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# Correr el algoritmo.
winner = p.run(eval_genomes)

# Guardar el algoritmo ganador en un archivo .pkl
with open ('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)