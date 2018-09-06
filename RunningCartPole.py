import numpy as np
import time
from tqdm import tqdm

START = 150

def run(agent, env, episodes, discount, extra_reward=0):
    pb = tqdm(total=episodes)
    N = 0
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            action = agent.policy(state)[0]
            x, x_dot, theta, theta_dot = state
            obs, reward, done, inf = env.step(action)
            t *= discount
            length_episode += 1
            if not done:
                if ((x + 0.02 * x_dot > 2.4) or (x + 0.02 * x_dot < -2.4)):
                    cumulated_reward += (1 + extra_reward) * t
                else:
                    cumulated_reward += t
            if length_episode>500:
                break
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards

def run_value_agent_RBF(agent, env, episodes, discount, n_actions):
    pb = tqdm(total=episodes)
    N = 0
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            val = -np.inf
            action = 0
            for i in range(n_actions):
                obs1, reward1, done1, inf1 = env.step(i)
                if done1:
                    reward1 = 0
                else:
                    reward1 = 1
                val_temp = reward1 + np.dot(agent.phi(np.array(env.state)), agent.weights) * discount
                if val_temp > val:
                    val = val_temp
                    action = i
                env.reset()
                env.state = list(state)
            obs, reward, done, inf = env.step(action)
            t *= discount
            length_episode += 1
            if not done:
                cumulated_reward += t
            if length_episode>500:
                break
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards
    

def run_value_agent_threshold_RBF_no_robust(agent, env, episodes, discount, n_actions, p_, noise_):
    pb = tqdm(total=episodes)
    N = 0
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            val = -np.inf
            action = 0
            for i in range(n_actions):
                obs1, reward1, done1, inf1 = env.step(i)
                if done1:
                    reward1 = 0
                else:
                    reward1 = 1
                indices, distrib = agent.phi(np.array(env.state))
                val_temp = reward1 + np.dot(distrib, agent.weights[indices]) * discount
                if val_temp > val:
                    val = val_temp
                    action = i
                env.reset()
                env.state = list(state)
            obs, reward, done, inf = env.step(action)
            t *= discount
            length_episode += 1
            if not done:
                cumulated_reward += t
            if length_episode>500:
                break
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards

def run_value_agent_threshold_RBF_shift(agent, env, episodes, discount, n_actions, p_, noise_):
    pb = tqdm(total=episodes)
    N = 0
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            val = -np.inf
            action = 0
            for i in range(n_actions):
                obs1, reward1, done1, inf1 = env.step(i)
                if done1:
                    reward1 = 0
                else:
                    reward1 = 1
                indices, distrib = agent.phi(np.array(env.state))
                val_temp = reward1 + np.dot(distrib, agent.weights[indices]) * discount
                if val_temp > val:
                    val = val_temp
                    action = i
                env.reset()
                env.state = list(state)
            env.force_mag = 10.0 - np.random.random_sample() * noise_
            obs, reward, done, inf = env.step(action)
            env.force_mag = 10.0
            t *= discount
            length_episode += 1
            if not done:
                cumulated_reward += t
            if length_episode>500:
                break
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards

 
def run_value_agent_threshold_RBF(agent, env, episodes, discount, n_actions, robustness=(0, 0, 0)):
    Robustness_type = {0:run_value_agent_threshold_RBF_no_robust, 1:run_value_agent_threshold_RBF_no_robust, 2:run_value_agent_threshold_RBF_no_robust, 3:run_value_agent_threshold_RBF_shift}
    durations, cumulated_rewards = Robustness_type[robustness[0]](agent, env, episodes, discount, n_actions, robustness[1], robustness[2])
    return durations, cumulated_rewards

def run_value_agent_Tiles_no_robust(agent, env, episodes, discount, n_actions, nb_tilings, p_, noise_):
    pb = tqdm(total=episodes)
    N = 0
    k = agent.size
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            action = np.random.randint(n_actions)
            val = -np.inf
            for i in range(n_actions):
                obs1, reward1, done1, inf1 = env.step(i)
                indices = agent.phi(np.array(env.state))
                ok = True
                for j, x in enumerate(indices):
                    if x>=k:
                        ok = False
                        indices[j] = k-1
                        break
                if not ok:
                    print('explored outside of cells I really know...')
                    action = np.random.randint(n_actions)
                    env.reset()
                    env.state = list(state)
                    break

                val_temp = reward1 + discount * np.sum(agent.weights[indices]) / nb_tilings
                if val_temp > val:
                    val = val_temp
                    action = i
                env.reset()
                env.state = list(state)
            obs, reward, done, inf = env.step(action)
            length_episode += 1
            if not done:
                cumulated_reward += t
            if length_episode>500:
                break
            t *= discount
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards

def run_value_agent_Tiles_shift(agent, env, episodes, discount, n_actions, nb_tilings, p_, noise_):
    pb = tqdm(total=episodes)
    N = 0
    k = agent.size
    cumulated_rewards = []
    durations = []
    while N<episodes:
        done = False
        env.reset()
        N += 1
        cumulated_reward = 0
        t = 1
        length_episode = 0
        while not done:
            state = np.array(env.state)
            action = np.random.randint(n_actions)
            val = -np.inf
            for i in range(n_actions):
                obs1, reward1, done1, inf1 = env.step(i)
                indices = agent.phi(np.array(env.state))
                ok = True
                for j, x in enumerate(indices):
                    if x>=k:
                        ok = False
                        indices[j] = k-1
                        break
                if not ok:
                    print('explored outside of cells I really know...')
                    action = np.random.randint(n_actions)
                    env.reset()
                    env.state = list(state)
                    break

                val_temp = reward1 + discount * np.sum(agent.weights[indices]) / nb_tilings
                if val_temp > val:
                    val = val_temp
                    action = i
                env.reset()
                env.state = list(state)
            decision = np.random.binomial(1, p_)
            if decision:
                env.force_mag = 10.0 - noise_
            obs, reward, done, inf = env.step(action)
            env.force_mag = 10.0
            length_episode += 1
            if not done:
                cumulated_reward += t
            if length_episode>500:
                break
            t *= discount
        pb.update(1)
        cumulated_rewards.append(cumulated_reward)
        durations.append(length_episode)
    pb.close()
    return durations, cumulated_rewards
  
def run_value_agent_Tiles(agent, env, episodes, discount, n_actions, nb_tilings, robustness=(0, 0, 0)):
    Robustness_type = {0:run_value_agent_Tiles_no_robust, 1:run_value_agent_Tiles_no_robust, 2:run_value_agent_Tiles_no_robust, 3:run_value_agent_Tiles_shift}
    durations, cumulated_rewards = Robustness_type[robustness[0]](agent, env, episodes, discount, n_actions, nb_tilings, robustness[1], robustness[2])
    return durations, cumulated_rewards