#CartPoleRandom

import gym

import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')#création de l'environnement 'CartPole' dans ce cas

num_episode = 1000 # on crée le nombre d'épisode max ou le nombre de boucle pour l'entrainement

steps_total = []
for i_episode in range(num_episode):
    state = env.reset()# création de l'état qui reset à chaque itération notre environnement

    step = 0

    while True:

        step += 1

    #for t in range(100): #seconde boucle pour les pas, range(100) pas obligatoire car l'ia n'ira pas jusque la pour le moment
        action = env.action_space.sample() #création de notre action, dans ce cas 1 ou 0 (gauche ou droite)

        new_state, reward, done, info = env.step(action) #récupération de nos variable via env.step qui va lancer la simulation

        #print(new_state)
        print(info)

        env.render() #pour voir le resultat de notre agent
        if done: # si l'étape est réussie, on sort de la boucle
            steps_total.append(step)
            print("Episode finished after "+ str(step) +" steps")
            break

print("Average number of step :" + str(sum(steps_total)/num_episode))
plt.plot(steps_total)
plt.show()
