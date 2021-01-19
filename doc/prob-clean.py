env = retro.make(game='Airstriker-Genesis', record=True)
env.reset()
done = False
accionesCorrectas = []
iniciando = True
def dormir():
    time.sleep(0.0001)

while True:
    env.render()
    probabilidadDisparo = random()
    action = [0, 0, 0, 0,   0,    0,      randint(0,1),      randint(0,1),         0,   1, 0, 0]
    ob, rew, done, info = env.step(action)
    dormir()

    if iniciando and info['gameover'] ==9:
        iniciando = False
        for i in range(len(accionesCorrectas)):
            env.render()
            ob, rew, done, info = env.step(accionesCorrectas[i])
            dormir()
            if info['gameover'] ==2:
                break

    if info['gameover'] ==9:
        accionesCorrectas.append(action)
    if info['gameover'] ==2:
        iniciando = True
        obs = env.reset()
        if (len(accionesCorrectas)>100):
            for e in range(100):
                accionesCorrectas.pop(len(accionesCorrectas)-e-1)

    if probabilidadDisparo <= 0.20:
        action = [randint(0, 1), 0, 0, 0, 0, 0, randint(0, 1), randint(0, 1), 0, 1, 0, 0]
        ob, rew, done, info = env.step(action)
        dormir()
        if info['gameover'] == 9:
            accionesCorrectas.append(action)

    if done:
        obs = env.reset()


env.close()
