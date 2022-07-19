#CartPoleRandom

import gym
import torch
import random
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor

env = gym.make('CartPole-v1')#création de l'environnement 'CartPole' dans ce cas

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
learning_rate = 0.01
num_episode = 2000 # on crée le nombre d'épisode max ou le nombre de boucle pour l'entrainement
gamma = 0.85

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500
####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    return epsilon

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(number_of_inputs,number_of_outputs)

    def forward(self,x):
        output = self.linear1(x)
        return output

class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        #loss_func = nn.SmoothL1Loss

        self.optimizer = optim.adam(params=self.nn.parameters(),lr = learning_rate)
        #optimizer = optim.RMSprop(params=mynn.parameters(), lr = learning_rate)

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]
        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn, 0)[1]
                action = action.item()
        else:
            action = env.action_space.sample()
        return action

    def optimize(self, state, action, new_state, reward, done):

        state = Tensor(state).to(device)
        new_state = Tensor(new_state).to(device)

        reward = Tensor([reward]).to(device)
        if done:
            target_value = reward
        else:
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            target_value = reward * max_new_state_values

        predicted_value = self.nn(state)[action]

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #Q[state, action] = reward + gamma * torch.max(Q[new_state])# bowman equation




qnet_agent = QNet_Agent()
steps_total = []
frames_total = 0

for i_episode in range(num_episode):
    state = env.reset()# création de l'état qui reset à chaque itération notre environnement

    step = 0

    while True:

        step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)

    #for t in range(100): #seconde boucle pour les pas, range(100) pas obligatoire car l'ia n'ira pas jusque la pour le moment
        #action = env.action_space.sample() #création de notre action, dans ce cas 1 ou 0 (gauche ou droite)
        action = qnet_agent.select_action(state,epsilon)

        new_state, reward, done, info = env.step(action) #récupération de nos variable via env.step qui va lancer la simulation

        qnet_agent.optimizer(state,action,new_state, reward, done)
        #print(new_state)
        #print(info)

        #env.render() #pour voir le resultat de notre agent
        state = new_state
        if done: # si l'étape est réussie, on sort de la boucle
            steps_total.append(step)
            print("Episode finished after "+ str(step) +" steps")
            break


print("Average reward :" + str(sum(steps_total)/num_episode))
print("Average reward last 100 Episodes :" + str(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha = 0.6, color = 'red')
plt.show()

env.close()
env.env.close()

