import gym
import torch
import matplotlib.pyplot as plt
env= gym.make('Taxi-v3')

number_of_states = env.observation_space.n #500
number_of_actions = env.action_space.n #6

Q = torch.zeros([number_of_states,number_of_actions])

num_episode = 500

gamma = 0.9 #learning rate



steps_total = []
rewards_total = []
for i_episode in range(num_episode):
    state = env.reset()
    step = 0
    while True:
        step += 1

        random_values = Q[state] + torch.rand(1, number_of_actions) /1000
        action = torch.max(random_values, 1)[1][0].item()

        new_state, reward, done, info = env.step(action)
        """print("new_state : " + str(new_state))
        print("Reward :" + str(reward))
        print("Done : "+ str(done))
        print("info : " +str(info))
        print(Q)"""
        Q[state,action] = reward + gamma *torch.max(Q[new_state])
        state = new_state
        if i_episode == num_episode-1:
            env.render()

        if i_episode < 5:
            env.render()
        if i_episode > num_episode -10:
            env.render()
        if done :
            steps_total.append(step)
            rewards_total.append(reward)
            print("Episode fini en " +str(step)+ " steps")
            break

print(Q)
print('Percent of episodes finished succesfully : '+str(sum(rewards_total)/num_episode))
print('Percent of last 100 episodes finished succesfully : '+str(sum(rewards_total[-100:])/100))

print("Average number of step :" + str(sum(steps_total)/num_episode))
print("Average number of last 100 episodes step :" + str(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha = 0.6, color = 'green')
plt.show()


plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha = 0.6, color = 'red')
plt.show()

