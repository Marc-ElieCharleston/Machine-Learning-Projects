#FrozenLake bellman deterministic
import time

import gym
import torch

import matplotlib.pyplot as plt
from gym.envs.registration import register
register(id='FrozenLakeNotSlippery-v1',
         entry_point='gym.envs.toy_text:FrozenLakeEnv',
         kwargs= {'map_name' : '4x4', 'is_slippery' : False}) #basculer is_slippery en False pour que l'environnement soit deterministe et non stochastique

env = gym.make('FrozenLakeNotSlippery-v1')#création de l'environnement 'CartPole' dans ce cas

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.9

egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999

Q = torch.zeros([number_of_states,number_of_actions])

num_episode = 1000 # on crée le nombre d'épisode max ou le nombre de boucle pour l'entrainement

steps_total = []
rewards_total = []
egreedy_total = []
for i_episode in range(num_episode):
    state = env.reset()# création de l'état qui reset à chaque itération notre environnement

    step = 0

    while True:
        step += 1

        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > egreedy:
            random_values = Q[state] + torch.rand(1,number_of_actions ) /1000
            action = torch.max(random_values,1)[1][0]
            action = action.item()
        else:
            action = env.action_space.sample()

        if egreedy > egreedy_final:
            egreedy*= egreedy_decay

        new_state, reward, done, info = env.step( action) #récupération de nos variable via env.step qui va lancer la simulation

        Q[state, action] = reward + gamma * torch.max(Q[new_state])# boellan equation

        state = new_state

        #print(new_state)
        #print(info)
        #time.sleep(0.4)
        if i_episode == num_episode - 1:
            env.render() #pour voir le resultat de notre agent
        if done: # si l'étape est réussie, on sort de la boucle
            steps_total.append(step)
            rewards_total.append(reward)
            egreedy_total.append(egreedy)
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

plt.figure(figsize=(12,5))
plt.title("Egreedy")
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha = 0.6, color = 'blue')
plt.show()



