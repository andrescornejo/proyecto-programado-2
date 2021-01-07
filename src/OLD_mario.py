import time
import retro

#VSCode terminal fix
import os
os.environ['DISPLAY'] = ':1'

env = retro.make('SuperMarioBros-Nes', 'Level1-1')
print(env.action_space)
env.reset()

done = False

while not done: 
    env.render()

    action = [0,0,0,0,0,0,0,1,0]
    # action = env.action_space.sample()
    ob, rew, done, info = env.step(action)
    # print("Action:", action)
    print('Action', action, 'reward: ',rew)
    # ob = Observation, rew = Reward
    if done:
        obs = env.reset()
    

env.close()