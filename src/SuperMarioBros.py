import time
import retro

#VSCode terminal fix
import os
os.environ['DISPLAY'] = ':1'

env = retro.make(game='SuperMarioBros-Nes', record=True)
print(env.action_space)
env.reset()

done = False

while not done: 
    env.render()

    action = [0,0,0,0,0,0,0,1,0]
    # action = env.action_space.sample()
    ob, rew, done, info = env.step(action)
    print("Action:", action)
    # ob = Observation, rew = Reward
    time.sleep(0.01)
    if done:
        obs = env.reset()
    #action = env.action_space.sample()
    #print(action)
    action = [0,0,0,0,0,0,0,0,1]
    # action = env.action_space.sample()
    ob, rew, done, info = env.step(action)
    print("Action:", action)
    # ob = Observation, rew = Reward
    time.sleep(0.01)
    if done:
        obs = env.reset()

env.close()