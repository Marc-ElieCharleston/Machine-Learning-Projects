import gym
import torch
import random
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")
Tensor = torch.Tensor

env = gym.make('CartPole-v1')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

###### PARAMS ######
file_to_save = 'pong_save.hdf5' # ou test avec .hdf5

learning_rate = 0.02
num_episodes = 50
gamma = 1

hidden_layer = 64

egreedy = 0.9 # pourcentage exploration/exploitation
egreedy_final = 0 # finir l'algo avec ce pourcentage exploration/exploitation
egreedy_decay = 500 #modification du eggreey par episode

report_interval = 10
score_to_solve = 195
####################

number_of_inputs = env.observation_space.shape[0]
number_of_outputs = env.action_space.n


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay)
    return epsilon


class NeuralNetwork(nn.Module):
    def __init__(self):
        # super(NeuralNetwork, self).__init__()
        super().__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, number_of_outputs)

        self.activation = nn.Tanh()
        #self.activation = nn.ReLU()


    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        return output2


class QNet_Agent(object):
    def __init__(self):
        self.nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        # self.loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        # self.optimizer = optim.RMSprop(params=mynn.parameters(), lr=learning_rate)

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
            target_value = reward + gamma * max_new_state_values

        predicted_value = self.nn(state)[action]

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Q[state, action] = reward + gamma * torch.max(Q[new_state])


qnet_agent = QNet_Agent()

steps_total = []

frames_total = 0
solved_after = 0
solved = False

start_time = time.time()

for i_episode in range(num_episodes):

    state = env.reset()

    step = 0
    # for step in range(100):
    while True:

        step += 1
        frames_total += 1

        epsilon = calculate_epsilon(frames_total)

        # action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)

        new_state, reward, done, info = env.step(action)

        qnet_agent.optimize(state, action, new_state, reward, done)

        state = new_state

        if done:
            steps_total.append(step)
            mean_reward_100 = sum(steps_total[-100:])/100

            if mean_reward_100 > score_to_solve and solved == False:
                print("Solved after : "+str(i_episode))
                solved_after = i_episode
                solved = True

            if i_episode % report_interval == 0:
                print("*** Episode : " + str(i_episode)+" ***")
                print("Average reward : " + str())
                print("Sum total last 10 steps : " +str(sum(steps_total[-report_interval:])))
                print("Average of last 100 steps : " + str(mean_reward_100))
                print("Average of all steps : " + str(sum(steps_total)/len(steps_total)))
                print("Epsilon : " +str(epsilon))
                print("Frames total : " + str(frames_total))
                elapsed_time = time.time() - start_time
                print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                print()

            break

print("Average reward: %.2f" % (sum(steps_total) / num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:]) / 100))

if solved :
    print("Solved after : " + str(solved_after))
plt.figure(figsize=(12, 5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

torch.save(qnet_agent, file_to_save)
print("test 1")

torch.save(qnet_agent.state_dict, file_to_save)
env.close()
env.env.close()



