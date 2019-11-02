import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.9 # Measure of how future rewards are valued over current value
EPISODES = 1_000_000_000
SHOW_EVERY = 10000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

obs_high = env.observation_space.high
obs_high[1] = 100.0
obs_high[3] = 100.0

obs_low = env.observation_space.low
obs_low[1] = -100.0
obs_low[3] = -100.0

print(obs_high)

discrete_os_win_size = (obs_high - obs_low)/DISCRETE_OS_SIZE

epsilon = 0.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decaying_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=0, high=1, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discreet_state(state):
    discrete_state = (state - obs_low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

run = True

if run == True:
    for episode in range(EPISODES):
        episode_reward = 0
        render = episode % SHOW_EVERY == 0
        
        discrete_state = get_discreet_state(env.reset())
        done = False
        while not done:

            # Add randomness to explore
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discreet_state(new_state)
            
            if render:
                env.render()
            
            if not done:
                current_q = q_table[discrete_state + (action, )]  #wow
                max_future_q = np.max(q_table[new_discrete_state])      
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            else:
                print(f"We have learned by episode {episode}")
                q_table[discrete_state + (action, )] = 2

            discrete_state = new_discrete_state
        
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decaying_value
        
        ep_rewards.append(episode_reward)

        if episode % SHOW_EVERY == 0:
            average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
            aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
            print(f"Episde: {episode}, avg: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()