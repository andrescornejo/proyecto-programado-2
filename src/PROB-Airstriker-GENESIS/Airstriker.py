#...
import time
import retro
from random import randint, uniform,random

"""Se crea el ambiente del juego"""
env = retro.make(game='Airstriker-Genesis', record=True)

"""Se inicia el ambiente"""
env.reset()

"""Done: esta en false mientras el juego este activo"""
done = False

"""Acciones donde el avi'on no explota, consideradas correctas o no erroneas"""
accionesCorrectas = []

"""Bandera que avisa si el juego se esta reiniciado o no"""
iniciando = True

"""Realiza un sleep para controlar la visualización del juego"""
def dormir():
    time.sleep(0.0001)

while True:
    env.render()

    """Determina si disparar o no en este intento"""
    probabilidadDisparo = random()

    """Acci'on aleatoria de moverse a la izquierda o derecha, o mantenerse el el centro"""
    action = [0, 0, 0, 0,   0,    0,      randint(0,1),      randint(0,1),         0,   1, 0, 0]
#             0=b 1 2  3   4=Up   5=down  6=left  7=right  8=A  9  10 11
    ob, rew, done, info = env.step(action)
    dormir()
    #print("ob",ob,"Action ", action, "Reward ", rew, "done ", done, "info", info)

    """Si el juego se esta reiniciando """
    if iniciando and info['gameover'] ==9:
        iniciando = False
        for i in range(len(accionesCorrectas)):
            env.render()
            ob, rew, done, info = env.step(accionesCorrectas[i])
            dormir()
            #print("ob", ob, "Action ", action, "Reward ", rew, "done ", done, "info", info)
            if info['gameover'] ==2:
                break


    """Si el juego esta activo, la acción se añade como correcta"""
    if info['gameover'] ==9:
        accionesCorrectas.append(action)
    """Si la nave exploto se activa la bandera iniciando, para denotar que el juego se esta reiniciando"""
    if info['gameover'] ==2:
        iniciando = True
        obs = env.reset()
        if (len(accionesCorrectas)>100):
            for e in range(100):
                accionesCorrectas.pop(len(accionesCorrectas)-e-1)


    """Para evitar desperdicio de balaz, y una solución más rapida, se redujo la probabilidad de disparoz a un 20% de las ocaciones"""
    if probabilidadDisparo <= 0.20:
        action = [randint(0, 1), 0, 0, 0, 0, 0, randint(0, 1), randint(0, 1), 0, 1, 0, 0]
        #             0=b 1 2  3   4=Up   5=down  6=left  7=right  8=A  9  10 11
        ob, rew, done, info = env.step(action)
        dormir()
        #print("ob", ob, "Action ", action, "Reward ", rew, "done ", done, "info", info)
        """si no muere con la acción se añade a acciones correctas"""
        if info['gameover'] == 9:
            accionesCorrectas.append(action)

    """Si el juego acabo lo reinicia"""
    if done:
        obs = env.reset()


env.close()
