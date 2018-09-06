import numpy as np
import time
from tqdm import tqdm
from TilesWrapperCartPole import MyCoding, MyCoding2, MyCoding3
import math
from rtree import index

Dic_dimensions = {4: MyCoding, 3: MyCoding3, 2: MyCoding2}

def Random_Full_Episodes(env, episodes, extra_reward=0):
    counter = episodes
    sampled_states = []
    sampled_rewards = []
    sampled_nstates = []
    sampled_actions = []
    while counter:
        done = False
        env.reset()
        while not done:
            sampled_states.append(np.array(env.state))
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            sampled_actions.append(action)
            state = env.state
            x, x_dot, theta, theta_dot = state
            if done:
                sampled_rewards.append(0)
            else:
                decision = (x + 0.02 * x_dot > 2.4) or (x + 0.02 * x_dot < -2.4) or (theta + 0.02 * theta_dot > math.pi / 15) or (theta + 0.02 * theta_dot < -math.pi / 15)
                if decision:
                    sampled_rewards.append(1 + extra_reward)
                else:
                    sampled_rewards.append(1)
            sampled_nstates.append(np.array(state))
        counter -= 1
    return sampled_states, sampled_actions, sampled_rewards, sampled_nstates

def gaussianKernelsCenters(env, mean, nb_actions, extra_reward=0):
    N = mean.shape[0]
    Kernels_rewards = np.zeros((N, nb_actions))
    Kernels_next_states = np.zeros((N, nb_actions, 4))
    env.reset()
    for i, kernel in enumerate(mean):
        for j in range(nb_actions):
            env.state = list(kernel)
            obs, reward, done, info = env.step(j)
            if done:
                reward = 0
            else:
                x, x_dot, theta, theta_dot = env.state
                decision = (x + 0.02 * x_dot > 2.4) or (x + 0.02 * x_dot < -2.4) or (theta + 0.02 * theta_dot > math.pi / 15) or (theta + 0.02 * theta_dot < -math.pi / 15)
                if decision:
                    reward = 1 + extra_reward
                else:
                    reward = 1
            Kernels_rewards[i, j] = reward
            Kernels_next_states[i, j, :] = np.array(env.state)
            env.reset()
    return Kernels_rewards, Kernels_next_states

def robustGaussianKernelsCenters(env, mean, sigma, n_per_kernel, nb_actions, extra_reward=0):
    N = mean.shape[0]
    Kernels_rewards = np.zeros((N, nb_actions, n_per_kernel))
    samples = sigma * np.random.randn(N, n_per_kernel, 4)
    Kernels_next_states = np.zeros((N, nb_actions, n_per_kernel, 4))
    env.reset()
    for i, kernel in enumerate(mean):
        for j in range(n_per_kernel):
            for k in range(nb_actions):
                env.state = list(kernel + samples[i, j,: ])
                obs, reward, done, info = env.step(k)
                if done:
                    reward = 0
                else:
                    x, x_dot, theta, theta_dot = env.state
                    decision = (x + 0.02 * x_dot > 2.4) or (x + 0.02 * x_dot < -2.4) or (theta + 0.02 * theta_dot > math.pi / 15) or (theta + 0.02 * theta_dot < -math.pi / 15)
                    if decision:
                        reward = 1 + extra_reward
                    else:
                        reward = 1
                Kernels_rewards[i, k, j] = reward
                Kernels_next_states[i, k, j, :] = np.array(env.state)
                env.reset()
    return Kernels_rewards, Kernels_next_states

def radialBasisFunctionsAveragersNoNoise(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, d_threshold, variances, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_):
    p = index.Property()
    p.dimension = 4
    idx = index.Index(properties=p)
    #np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
    N = n_x * n_x_dot * n_theta * n_theta_dot
    data_rewards = np.zeros((N, nb_actions))
    data_next_states_indices = [0] * N
    data_next_states_distrib = [0] * N
    from_indice_to_kernel = [0] * N
    lens = []

    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10

    print('Rtree construction')
    for i1, x in enumerate(tqdm(t_x)):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    idx.insert(i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, (x, x_dot, theta, theta_dot, x, x_dot, theta, theta_dot))
                    from_indice_to_kernel[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = np.array([x, x_dot, theta, theta_dot])

    print('Sampling')
    for i1, x in enumerate(tqdm(t_x)):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    next_states_indices = []
                    next_states_distrib = []
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        id_kernels_activated_nstate = list(idx.intersection((nx - d_threshold, nx_dot - d_threshold, ntheta - d_threshold / 10, ntheta_dot - d_threshold, nx + d_threshold, nx_dot + d_threshold, ntheta + d_threshold / 10, ntheta_dot + d_threshold)))
                        if not id_kernels_activated_nstate:
                            id_kernels_activated_nstate = list(idx.nearest((nx, nx_dot, ntheta, ntheta_dot, nx, nx_dot, ntheta, ntheta_dot), 1))
                        lens.append(len(id_kernels_activated_nstate))

                        states_kernels_activated_nstate = np.array([from_indice_to_kernel[indice] for indice in id_kernels_activated_nstate])
                        kernels_activation_nstate = np.exp(-np.sum((states_kernels_activated_nstate - np.array([nx, nx_dot, ntheta, ntheta_dot])) ** 2 / variances, axis=1))
                        kernels_activation_nstate = kernels_activation_nstate / np.sum(kernels_activation_nstate)
                        
                        next_states_indices.append(id_kernels_activated_nstate)
                        next_states_distrib.append(kernels_activation_nstate)
                        data_rewards[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, act] = reward
                        assert len(id_kernels_activated_nstate) == len(kernels_activation_nstate), 't\'as deconne Arnaud'
                    data_next_states_indices[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_indices
                    data_next_states_distrib[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_distrib

    print('robust neighbors kernels')
    lens = np.array(lens)
    print(np.mean(lens))
    print(np.min(lens))
    print(np.max(lens))
    return N, np.array(from_indice_to_kernel), data_rewards, data_next_states_indices, data_next_states_distrib, idx

def radialBasisFunctionsAveragersNoise1(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, d_threshold, variances, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_):
    p = index.Property()
    p.dimension = 4
    idx = index.Index(properties=p)
    #np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
    N = n_x * n_x_dot * n_theta * n_theta_dot
    data_rewards = np.zeros((N, nb_actions))
    data_next_states_indices = [0] * N
    data_next_states_distrib = [0] * N
    from_indice_to_kernel = [0] * N
    lens = []

    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10

    print('Rtree construction')
    for i1, x in enumerate(t_x):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    idx.insert(i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, (x, x_dot, theta, theta_dot, x, x_dot, theta, theta_dot))
                    from_indice_to_kernel[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = np.array([x, x_dot, theta, theta_dot])

    print('Sampling')
    for i1, x in enumerate(tqdm(t_x)):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    next_states_indices = []
                    next_states_distrib = []
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        decision = np.random.binomial(1, p_)
                        if decision:
                            reward += noise_# np.random.uniform(noise_)
                        else:
                            reward += 0
                        id_kernels_activated_nstate = list(idx.intersection((nx - d_threshold, nx_dot - d_threshold, ntheta - d_threshold / 10, ntheta_dot - d_threshold, nx + d_threshold, nx_dot + d_threshold, ntheta + d_threshold / 10, ntheta_dot + d_threshold)))
                        if not id_kernels_activated_nstate:
                            id_kernels_activated_nstate = list(idx.nearest((nx, nx_dot, ntheta, ntheta_dot, nx, nx_dot, ntheta, ntheta_dot), 1))
                        lens.append(len(id_kernels_activated_nstate))

                        states_kernels_activated_nstate = np.array([from_indice_to_kernel[indice] for indice in id_kernels_activated_nstate])
                        kernels_activation_nstate = np.exp(-np.sum((states_kernels_activated_nstate - np.array([nx, nx_dot, ntheta, ntheta_dot])) ** 2 / variances, axis=1))
                        kernels_activation_nstate = kernels_activation_nstate / np.sum(kernels_activation_nstate)
                        
                        next_states_indices.append(id_kernels_activated_nstate)
                        next_states_distrib.append(kernels_activation_nstate)
                        data_rewards[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, act] = reward
                        assert len(id_kernels_activated_nstate) == len(kernels_activation_nstate), 't\'as deconne Arnaud'
                    data_next_states_indices[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_indices
                    data_next_states_distrib[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_distrib

    print('robust neighbors kernels')
    lens = np.array(lens)
    print(np.mean(lens))
    print(np.min(lens))
    print(np.max(lens))
    return N, np.array(from_indice_to_kernel), data_rewards, data_next_states_indices, data_next_states_distrib, idx

def radialBasisFunctionsAveragersNoise2(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, d_threshold, variances, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_):
    p = index.Property()
    p.dimension = 4
    idx = index.Index(properties=p)

    #np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
    N = n_x * n_x_dot * n_theta * n_theta_dot
    data_rewards = np.zeros((N, nb_actions))
    data_next_states_indices = [0] * N
    data_next_states_distrib = [0] * N
    from_indice_to_kernel = [0] * N
    lens = []

    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10

    print('Rtree construction')
    for i1, x in enumerate(tqdm(t_x)):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    idx.insert(i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, (x, x_dot, theta, theta_dot, x, x_dot, theta, theta_dot))
                    from_indice_to_kernel[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = np.array([x, x_dot, theta, theta_dot])

    print('Sampling')
    for i1, x in enumerate(tqdm(t_x)):
        for i2, x_dot in enumerate(t_x_dot):
            for i3, theta in enumerate(t_theta):
                for i4, theta_dot in enumerate(t_theta_dot):
                    next_states_indices = []
                    next_states_distrib = []
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        decision = np.random.binomial(1, p_)
                        if decision:
                            nx -= noise_ * nx
                            nx_dot -= noise_ * nx_dot
                            ntheta -= noise_ * ntheta
                            ntheta_dot -= noise_ * ntheta_dot
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        id_kernels_activated_nstate = list(idx.intersection((nx - d_threshold, nx_dot - d_threshold, ntheta - d_threshold / 10, ntheta_dot - d_threshold, nx + d_threshold, nx_dot + d_threshold, ntheta + d_threshold / 10, ntheta_dot + d_threshold)))
                        if not id_kernels_activated_nstate:
                            id_kernels_activated_nstate = list(idx.nearest((nx, nx_dot, ntheta, ntheta_dot, nx, nx_dot, ntheta, ntheta_dot), 1))
                        lens.append(len(id_kernels_activated_nstate))

                        states_kernels_activated_nstate = np.array([from_indice_to_kernel[indice] for indice in id_kernels_activated_nstate])
                        kernels_activation_nstate = np.exp(-np.sum((states_kernels_activated_nstate - np.array([nx, nx_dot, ntheta, ntheta_dot])) ** 2 / variances, axis=1))
                        kernels_activation_nstate = kernels_activation_nstate / np.sum(kernels_activation_nstate)
                        
                        next_states_indices.append(id_kernels_activated_nstate)
                        next_states_distrib.append(kernels_activation_nstate)
                        data_rewards[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta, act] = reward
                        assert len(id_kernels_activated_nstate) == len(kernels_activation_nstate), 't\'as deconne Arnaud'
                    data_next_states_indices[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_indices
                    data_next_states_distrib[i1 + i2 * n_x + i3 * n_x * n_x_dot + i4 * n_x * n_x_dot * n_theta] = next_states_distrib

    print('robust neighbors kernels')
    lens = np.array(lens)
    print(np.mean(lens))
    print(np.min(lens))
    print(np.max(lens))
    return N, np.array(from_indice_to_kernel), data_rewards, data_next_states_indices, data_next_states_distrib, idx

def radialBasisFunctionsAveragers(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, d_threshold, variances, xmax=2.5, xdotmax=3, thetamax=0.3, thetadotmax=3.5, robustness=(0, 0, 0)):
    Robustness_type = {0:radialBasisFunctionsAveragersNoNoise, 1:radialBasisFunctionsAveragersNoise1, 2:radialBasisFunctionsAveragersNoise2}
    N, mean, data_rewards, data_next_states_indices, data_next_states_distrib, idx = Robustness_type[robustness[0]](env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, d_threshold, variances, xmax, xdotmax, thetamax, thetadotmax, robustness[1], robustness[2])
    return N, mean, data_rewards, data_next_states_indices, data_next_states_distrib, idx


def stateAggregationGridNoNoise(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, max_size, nb_tilings, width, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_, dim):
    N = n_x * n_x_dot * n_theta * n_theta_dot
    print(x_max, x_dot_max, theta_max, theta_dot_max)
    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10
    
    dict_abstract_states = {}
    scales = [2 * x_max, 2 * x_dot_max, 2 * theta_max, 2 * theta_dot_max]
    tile_coding = Dic_dimensions[dim](max_size, nb_tilings, width, scale=scales)
    
    print('start of actual sampling-------')
    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])

    number_of_abstract_states = tile_coding.iht.count()
    
    print('number of abstract states:')
    print(number_of_abstract_states)
    print('to compare with expected size:')
    print(((width + 1) ** dim) * nb_tilings)

    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        next_indices = tile_coding.mytiles([nx, nx_dot, ntheta, ntheta_dot])
                        for i, next_indice in enumerate(next_indices):
                            if next_indice > number_of_abstract_states:
                                next_indices[i] = number_of_abstract_states
                        for indice in indices:
                            key = (indice, act)
                            if key in dict_abstract_states:
                                dict_abstract_states[key].append((reward, next_indices))
                            else:
                                dict_abstract_states[key] = [(reward, next_indices)]
    for action in range(nb_actions):
        dict_abstract_states[(number_of_abstract_states, action)] = [(0, [number_of_abstract_states] * nb_tilings)]
    k = number_of_abstract_states + 1
    lengths = np.zeros((k, nb_actions))
    for i in range(k):
        for act in range(nb_actions):
            lengths[i, act] = len(dict_abstract_states[(i, act)])
    print('average number of transitions: ' + str(np.mean(lengths)))
    print('per action: ' + str(np.mean(lengths, axis=0)))
    
    print('shifting to lists')
    
    list_of_rewards = []
    list_of_nstates = []
    
    for j in range(k):
        rewards_for_each_action = []
        nstates_for_each_action = []
        for action in range(nb_actions):
            next_rewards, next_states = zip(*dict_abstract_states[(j, action)])
            #print(len(next_rewards))
            #print(len(next_states))
            #print(len(next_states[0]))
            #print('------')
            rewards_for_each_action.append(list(next_rewards))
            nstates_for_each_action.append(list(next_states))
        list_of_rewards.append(rewards_for_each_action)
        list_of_nstates.append(nstates_for_each_action)
    print('end shifting to list')
    print(k)
    print('vs ' + str(len(list_of_nstates)))
    print('vs ' + str(len(list_of_rewards)))
    print('average number of transitions action0: ' + str(sum([len(list_of_nstates[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_nstates[j][1]) for j in range(k)])/k))
    print('average number of transitions action0: ' + str(sum([len(list_of_rewards[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_rewards[j][1]) for j in range(k)])/k))
    return k, list_of_rewards, list_of_nstates, tile_coding

def stateAggregationGridNoise1(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, max_size, nb_tilings, width, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_, dim):
    N = n_x * n_x_dot * n_theta * n_theta_dot
    print(x_max, x_dot_max, theta_max, theta_dot_max)
    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10
    
    dict_abstract_states = {}
    scales = [2 * x_max, 2 * x_dot_max, 2 * theta_max, 2 * theta_dot_max]
    tile_coding = Dic_dimensions[dim](max_size, nb_tilings, width, scale=scales)
    
    print('start of actual sampling-------')
    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])

    number_of_abstract_states = tile_coding.iht.count()
    
    print('number of abstract states:')
    print(number_of_abstract_states)
    print('to compare with expected size:')
    print(((width + 1) **4) * nb_tilings)

    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        decision = np.random.binomial(1, p_)
                        if decision:
                            reward += noise_ #np.random.uniform(noise_)
                        next_indices = tile_coding.mytiles([nx, nx_dot, ntheta, ntheta_dot])
                        for i, next_indice in enumerate(next_indices):
                            if next_indice > number_of_abstract_states:
                                next_indices[i] = number_of_abstract_states
                        for indice in indices:
                            key = (indice, act)
                            if key in dict_abstract_states:
                                dict_abstract_states[key].append((reward, next_indices))
                            else:
                                dict_abstract_states[key] = [(reward, next_indices)]
    for action in range(nb_actions):
        dict_abstract_states[(number_of_abstract_states, action)] = [(0, [number_of_abstract_states] * nb_tilings)]
    k = number_of_abstract_states + 1
    lengths = np.zeros((k, nb_actions))
    for i in range(k):
        for act in range(nb_actions):
            lengths[i, act] = len(dict_abstract_states[(i, act)])
    print('average number of transitions: ' + str(np.mean(lengths)))
    print('per action: ' + str(np.mean(lengths, axis=0)))
    
    print('shifting to lists')
    
    list_of_rewards = []
    list_of_nstates = []
    
    for j in range(k):
        rewards_for_each_action = []
        nstates_for_each_action = []
        for action in range(nb_actions):
            next_rewards, next_states = zip(*dict_abstract_states[(j, action)])
            #print(len(next_rewards))
            #print(len(next_states))
            #print(len(next_states[0]))
            #print('------')
            rewards_for_each_action.append(list(next_rewards))
            nstates_for_each_action.append(list(next_states))
        list_of_rewards.append(rewards_for_each_action)
        list_of_nstates.append(nstates_for_each_action)
    print('end shifting to list')
    print(k)
    print('vs ' + str(len(list_of_nstates)))
    print('vs ' + str(len(list_of_rewards)))
    print('average number of transitions action0: ' + str(sum([len(list_of_nstates[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_nstates[j][1]) for j in range(k)])/k))
    print('average number of transitions action0: ' + str(sum([len(list_of_rewards[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_rewards[j][1]) for j in range(k)])/k))
    return k, list_of_rewards, list_of_nstates, tile_coding

def stateAggregationGridNoise2(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, max_size, nb_tilings, width, x_max, x_dot_max, theta_max, theta_dot_max, p_, noise_, dim):
    N = n_x * n_x_dot * n_theta * n_theta_dot
    print(x_max, x_dot_max, theta_max, theta_dot_max)
    t_x = np.linspace(-x_max, x_max, num=n_x)
    t_x_dot = np.linspace(-x_dot_max, x_dot_max, num=n_x_dot)
    t_theta = np.linspace(-theta_max, theta_max, num=n_theta)
    t_theta_dot = np.linspace(-theta_dot_max, theta_dot_max, num=n_theta_dot)
    
    grid_noise_x = np.random.random(size=n_x)
    h = 0.5 * x_max / (n_x -1)
    t_x = t_x - h / 10 + grid_noise_x * 2 * h / 10
    grid_noise_x_dot = np.random.random(size=n_x_dot)
    h = 0.5 * x_dot_max / (n_x_dot -1)
    t_x_dot = t_x_dot - h / 10 + grid_noise_x_dot * 2 * h / 10
    grid_noise_theta = np.random.random(size=n_theta)
    h = 0.5 * theta_max / (n_theta -1)
    t_theta = t_theta - h / 10 + grid_noise_theta * 2 * h / 10
    grid_noise_theta_dot = np.random.random(size=n_theta_dot)
    h = 0.5 * theta_dot_max / (n_theta_dot -1)
    t_theta_dot = t_theta_dot - h / 10 + grid_noise_theta_dot * 2 * h / 10
    
    dict_abstract_states = {}
    scales = [2 * x_max, 2 * x_dot_max, 2 * theta_max, 2 * theta_dot_max]
    tile_coding = Dic_dimensions[dim](max_size, nb_tilings, width, scale=scales)
    
    print('start of actual sampling-------')
    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])

    number_of_abstract_states = tile_coding.iht.count()
    
    print('number of abstract states:')
    print(number_of_abstract_states)
    print('to compare with expected size:')
    print(((width + 1) **4) * nb_tilings)

    for x in tqdm(t_x):
        for x_dot in t_x_dot:
            for theta in t_theta:
                for theta_dot in t_theta_dot:
                    indices = tile_coding.mytiles([x, x_dot, theta, theta_dot])
                    for act in range(nb_actions):
                        env.reset()
                        env.state = [x, x_dot, theta, theta_dot]
                        obs, reward, done, info = env.step(act)
                        nx, nx_dot, ntheta, ntheta_dot = env.state
                        decision = np.random.binomial(1, p_)
                        if decision:
                            nx -= noise_ * nx
                            nx_dot -= noise_ * nx_dot
                            ntheta -= noise_ * ntheta
                            ntheta_dot -= noise_ * ntheta_dot
                        if done:
                            reward = 0
                        else:
                            reward = 1
                        next_indices = tile_coding.mytiles([nx, nx_dot, ntheta, ntheta_dot])
                        for i, next_indice in enumerate(next_indices):
                            if next_indice > number_of_abstract_states:
                                next_indices[i] = number_of_abstract_states
                        for indice in indices:
                            key = (indice, act)
                            if key in dict_abstract_states:
                                dict_abstract_states[key].append((reward, next_indices))
                            else:
                                dict_abstract_states[key] = [(reward, next_indices)]
    for action in range(nb_actions):
        dict_abstract_states[(number_of_abstract_states, action)] = [(0, [number_of_abstract_states] * nb_tilings)]
    k = number_of_abstract_states + 1
    lengths = np.zeros((k, nb_actions))
    for i in range(k):
        for act in range(nb_actions):
            lengths[i, act] = len(dict_abstract_states[(i, act)])
    print('average number of transitions: ' + str(np.mean(lengths)))
    print('per action: ' + str(np.mean(lengths, axis=0)))
    
    print('shifting to lists')
    
    list_of_rewards = []
    list_of_nstates = []
    
    for j in range(k):
        rewards_for_each_action = []
        nstates_for_each_action = []
        for action in range(nb_actions):
            next_rewards, next_states = zip(*dict_abstract_states[(j, action)])
            #print(len(next_rewards))
            #print(len(next_states))
            #print(len(next_states[0]))
            #print('------')
            rewards_for_each_action.append(list(next_rewards))
            nstates_for_each_action.append(list(next_states))
        list_of_rewards.append(rewards_for_each_action)
        list_of_nstates.append(nstates_for_each_action)
    print('end shifting to list')
    print(k)
    print('vs ' + str(len(list_of_nstates)))
    print('vs ' + str(len(list_of_rewards)))
    print('average number of transitions action0: ' + str(sum([len(list_of_nstates[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_nstates[j][1]) for j in range(k)])/k))
    print('average number of transitions action0: ' + str(sum([len(list_of_rewards[j][0]) for j in range(k)])/k))
    print('average number of transitions action1: ' + str(sum([len(list_of_rewards[j][1]) for j in range(k)])/k))
    return k, list_of_rewards, list_of_nstates, tile_coding
    

def stateAggregationGrid(env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, max_size, nb_tilings, width, xmax=2.5, xdotmax=3, thetamax=0.3, thetadotmax=3.5, robustness=(0, 0, 0), dimension=4):
    Robustness_type = {0:stateAggregationGridNoNoise, 1:stateAggregationGridNoise1, 2:stateAggregationGridNoise2}
    k, list_of_rewards, list_of_nstates, tile_coding = Robustness_type[robustness[0]](env, n_x, n_x_dot, n_theta, n_theta_dot, nb_actions, max_size, nb_tilings, width, xmax, xdotmax, thetamax, thetadotmax, robustness[1], robustness[2], dimension)
    return k, list_of_rewards, list_of_nstates, tile_coding

def Random_plus_RBFs(env, episodes, number):
    counter = episodes
    sampled_states = []
    sampled_rewards = []
    sampled_nstates = []
    sampled_actions = []
    while counter:
        done = False
        env.reset()
        while not done:
            sampled_states.append(np.array(env.state))
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            sampled_actions.append(action)
            if done:
                sampled_rewards.append(0)
            else:
                sampled_rewards.append(1)
            sampled_nstates.append(np.array(env.state))
        counter -= 1
    RBF_indices = np.random.choice(len(sampled_states), size=number)
    return sampled_states, sampled_actions, sampled_rewards, sampled_nstates, RBF_indices

    '''
def Agent_Full_Episodes(agent, greed_level, env, episodes):
    counter = episodes
    sampled_states = []
    sampled_rewards = []
    sampled_nstates = []
    sampled_actions = []
    while counter:
        done = False
        env.reset()
        while not done:
            state = np.array(env.state)
            sampled_states.append(state)
            decision = np.random.binomial(1, greed_level)
            if decision:  # Go random
                arg = np.random.randint(agent.nb_actions)
            else:  # Go greedy
                arg = agent.policy(state)[0]
            obs, reward, done, info = env.step(arg)
            sampled_actions.append(arg)
            if done:
                sampled_rewards.append(0)
            else:
                sampled_rewards.append(1)
            sampled_nstates.append(np.array(env.state))
        counter -= 1
    return sampled_states, sampled_actions, sampled_rewards, sampled_nstates

def training_single_dataset(agent, env, episodes, discount, max_iter, stop, greed_level, delta_init=0.1):
    info = {}
    iterations = 0
    timer = time.time()
    agent.reset()
    D_states, D_actions, D_rewards, D_next_states = Random_Full_Episodes(env, episodes)
    N = len(D_states)
    k = agent.nb_actions * (agent.number + 1)

    # Computations of the b term
    b = np.zeros(k)
    for i in range(N):
        b += agent.phi(D_states[i], D_actions[i]) * D_rewards[i]

    # Initialization of weights
    w = np.copy(agent.weights)
    evolution = []

    while True:
        iterations += 1
        B = delta_init * np.identity(k)
        for i in tqdm(range(N)):
            s, a, r, s_prime = D_states[i], D_actions[i], D_rewards[i], D_next_states[i]

            V1 = agent.phi(s, a)

            # Epsilon-greedy policy with current weights to get the action taken in the next state s_prime
            decision = np.random.binomial(1, greed_level)
            if decision:  # Go random
                arg = np.random.randint(agent.nb_actions)
            else:  # Go greedy
                arg = agent.policy(s_prime)[0]

            A = V1 - discount * agent.phi(s_prime, arg)
            A = A.reshape(1,k)
            V1 = V1.reshape(k,1)
            B = B + V1 @ A
        w = np.linalg.lstsq(B,b,rcond=None)[0]

        # Gap between the new policy and the former
        gap = np.linalg.norm(w - agent.weights)
        evolution.append(gap)
        agent.weights = np.copy(w)
        if ((gap <= stop) or (iterations >= max_iter)):
            break

    info['time'] = time.time() - timer
    info['iterations'] = iterations
    info['deltas'] = evolution
    return info

def training_multiple_datasets(agent, env, episodes, discount, max_iter, stop, greed_level, delta_init=0.1):
    info = {}
    iterations = 0
    timer = time.time()
    agent.reset()
    D_states, D_actions, D_rewards, D_next_states = Random_Full_Episodes(env, episodes)
    N = len(D_states)
    k = agent.nb_actions * (agent.number + 1)

    # Computations of the b term
    b = np.zeros(k)
    for i in range(N):
        b += agent.phi(D_states[i], D_actions[i]) * D_rewards[i]

    # Initialization of weights
    w = np.copy(agent.weights)
    evolution = []

    while True:
        iterations += 1
        B = delta_init * np.identity(k)
        for i in tqdm(range(N)):
            s, a, r, s_prime = D_states[i], D_actions[i], D_rewards[i], D_next_states[i]

            V1 = agent.phi(s, a)

            # Epsilon-greedy policy with current weights to get the action taken in the next state s_prime
            decision = np.random.binomial(1, greed_level)
            if decision:  # Go random
                arg = np.random.randint(agent.nb_actions)
            else:  # Go greedy
                arg = agent.policy(s_prime)[0]

            A = V1 - discount * agent.phi(s_prime, arg)
            A = A.reshape(1,k)
            V1 = V1.reshape(k,1)
            B = B + V1 @ A
        w = np.linalg.lstsq(B,b,rcond=None)[0]

        # Gap between the new policy and the former
        gap = np.linalg.norm(w - agent.weights)
        evolution.append(gap)
        agent.weights = np.copy(w)
        
        D_states, D_actions, D_rewards, D_next_states = Agent_Full_Episodes(agent, greed_level, env, episodes)
        N = len(D_states)

        # Computations of the b term
        b = np.zeros(k)
        for i in range(N):
            b += agent.phi(D_states[i], D_actions[i]) * D_rewards[i]
        
        if ((gap <= stop) or (iterations >= max_iter)):
            break

    info['time'] = time.time() - timer
    info['iterations'] = iterations
    info['deltas'] = evolution
    return info
    '''