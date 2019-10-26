import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95 # Measure of how future rewards are valued over current value
EPISODES = 10000
SHOW_EVERY = 1000

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

print(env.observation_space.high)
print(env.observation_space.low)


epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decaying_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=0, high=200, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discreet_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

run = True

if run == True:
    for episode in range(EPISODES):
        if episode % SHOW_EVERY == 0:
            render = True
            print(episode)
        else:
            render = False
        
        discrete_state = get_discreet_state(env.reset())
        done = False
        while not done:

            # Add randomness to explore
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discreet_state(new_state)
            
            if render:
                env.render()
            
            if not done:
                current_q = q_table[discrete_state + (action, )]  #wow
                max_future_q = np.max(q_table[new_discrete_state])      
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            
            #elif new_state[0] >= env.goal_position:
            #    print(f"We have learned by episode {episode}")
            #    q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state
        
        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decaying_value

env.close()