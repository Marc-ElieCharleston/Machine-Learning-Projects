#FrozenLake bstochastic
import time

import gym
import torch

import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1')#création de l'environnement

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.95
learning_rate = 0.9

Q = torch.zeros([number_of_states,number_of_actions])

num_episode = 1000 # on crée le nombre d'épisode max ou le nombre de boucle pour l'entrainement

steps_total = []
rewards_total = []
for i_episode in range(num_episode):
    state = env.reset()# création de l'état qui reset à chaque itération notre environnement

    step = 0

    while True:

        step += 1

    #for t in range(100): #seconde boucle pour les pas, range(100) pas obligatoire car l'ia n'ira pas jusque la pour le moment
        random_values = Q[state] + torch.rand(1,number_of_actions ) /1000

        action = torch.max(random_values,1)[1][0]

        new_state, reward, done, info = env.step(action.item()) #récupération de nos variable via env.step qui va lancer la simulation

        Q[state, action] = (1- learning_rate) * Q[state, action] + learning_rate * (reward + gamma * torch.max(Q[new_state]))# bellmanman equation

        state = new_state

        #print(new_state)
        #print(info)
        #time.sleep(0.4)

        #env.render() #pour voir le resultat de notre agent
        if done: # si l'étape est réussie, on sort de la boucle
            steps_total.append(step)
            rewards_total.append(reward)
            print("Episode finished after "+ str(step) +" steps")
            break


print(Q)
print('Percent of episodes finished succesfully : '+str(sum(rewards_total)/num_episode))
print('Percent of episodes finished succesfully : '+str(sum(rewards_total[-100:])/100))

print("Average number of step :" + str(sum(steps_total)/num_episode))
print("Average number of step :" + str(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha = 0.6, color = 'green')
plt.show()


plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha = 0.6, color = 'red')
plt.show()
