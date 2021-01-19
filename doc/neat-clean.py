env = retro.make('SuperMarioBros-Nes', 'Level1-1')

def downscale_image(ob, size_x, size_y):
    ob = cv2.resize(ob, (size_x, size_y))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (size_x, size_y))
    imgarray = np.ndarray.flatten(ob)
    return imgarray

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()

        size_x, size_y, img_color= env.observation_space.shape

        size_x = int(size_x/8)
        size_y = int(size_y/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        stag_counter = 0

        xpos_lo = 0
        xpos_hi = 0

        done = False
        won = False

        while not done:
            imgarray = downscale_image(ob, size_x, size_y)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)

            xpos_lo = info['xscrollLo']

            next_xpos_hi = info['xscrollHi']

            if(next_xpos_hi > xpos_hi):
                xpos_hi = next_xpos_hi

            if xpos_hi >= 12 and not won:
                fitness_current *= xpos_hi
                won = True
            else:
                fitness_current += rew

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                stag_counter = 0
            else:
                stag_counter += 1

            if stag_counter > 250:
                done = True
            
            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neural-config')

p = neat.Population(config)

winner = p.run(eval_genomes)

with open ('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
