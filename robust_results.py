import sys, getopt
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
import ADP
from getopt import getopt
import SamplingCartPole
import RunningCartPole
import gym
import csv
import time
import multiprocessing as mp

x_max, x_dot_max, theta_max, theta_dot_max = 2.5, 3, 0.3, 3.5


def RVITilesParallel(sampling, ID, DISCOUNT, DELTA_INIT, GREED, STOP, DELTA_PHI, N_trainings, robust_levels, N_tests, MAX_ITER, INSIDE_LOOP_ITER, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, K_CLOSE, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, Nrbf, VAR, save, ROBUST):
    Lengths = np.zeros((len(robust_levels), N_trainings))
    Rewards = np.zeros((len(robust_levels), N_trainings))
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    sample_sizes = DIC_SAMPLES[N_RBF_X]
    if ROBUST[1]==-1:
        noise_axis = 1
    else:
        noise_axis = 2
    for i, robust_level in enumerate(robust_levels):

        def f(output, seed):
            np.random.seed(seed)
            robust_params = list(ROBUST)
            robust_params[noise_axis] = robust_level
            robust_params = tuple(robust_params)
            N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT = sample_sizes
            size, rewards, next_states, coding = SamplingCartPole.stateAggregationGrid(env, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, robustness=robust_params)
            D = rewards, next_states
            agent = ADP.RVI_EA_Tiles(DISCOUNT, D, MAX_ITER, NB_ACTIONS, size, coding, NB_TILINGS, WIDTH, DELTA_INIT)
            w, info = agent.solve_robust(GREED, STOP)
            durations, cumulated_rewards = RunningCartPole.run_value_agent_Tiles(agent, env, N_tests, DISCOUNT, NB_ACTIONS, NB_TILINGS, robustness=robust_params)
            output.put((np.mean(durations), np.mean(cumulated_rewards)))
            return

        output_queue = mp.Queue()
        
        for loop in range(N_trainings // 10):
            processes = [mp.Process(target=f, args=(output_queue,_ + loop * 10,)) for _ in range(10)]
            for p in processes:
                p.start()
            for j in range(10):
                results = output_queue.get()
                Lengths[i, loop * 10 + j] = results[0]
                Rewards[i, loop * 10 + j] = results[1]
            for p in processes:
                p.join()
        
        if N_trainings%10 > 0:
            processes = [mp.Process(target=f, args=(output_queue, N_trainings - N_trainings%10 + x)) for x in range(N_trainings%10)]
            for p in processes:
                p.start()
            for j in range(N_trainings%10):
                results = output_queue.get()
                Lengths[i, N_trainings - N_trainings%10 + j] = results[0]
                Rewards[i, N_trainings - N_trainings%10 + j] = results[1]
            for p in processes:
                p.join()

    if save:
        np.save('./Arrays/' + str(ID) + 'Lengths', np.mean(Lengths, axis=1))
        np.save('./Arrays/' + str(ID) + 'LengthsStd', np.std(Lengths, axis=1) * 1.96 / N_trainings)
        np.save('./Arrays/' + str(ID) + 'Rewards', np.mean(Rewards, axis=1))
        np.save('./Arrays/' + str(ID) + 'RewardsStd', np.std(Rewards, axis=1) * 1.96 / N_trainings)
    plt.figure(1)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average sum of Discounted Rewards')
    V1 = np.mean(Rewards, axis=1)
    eV1 = np.std(Rewards, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V1, eV1, ecolor='red', label='RVITiles', color='blue')
    plt.title('Discounted Rewards')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Rewards')
    plt.close()
    plt.figure(2)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average Length of runs')
    V2 = np.mean(Lengths, axis=1)
    eV2 = np.std(Lengths, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V2, eV2, ecolor='red', label='RVITiles', color='blue')
    plt.title('Lengths')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Lengths')
    plt.close()
    return

def RVIRBFParallel(sampling, ID, DISCOUNT, DELTA_INIT, GREED, STOP, DELTA_PHI, N_trainings, robust_levels, N_tests, MAX_ITER, INSIDE_LOOP_ITER, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, K_CLOSE, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, Nrbf, VAR, save, ROBUST):
    Lengths = np.zeros((len(robust_levels), N_trainings))
    Rewards = np.zeros((len(robust_levels), N_trainings))
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    sample_sizes = DIC_SAMPLES[N_RBF_X]
    if ROBUST[1]==-1:
        noise_axis = 1
    else:
        noise_axis = 2
    for i, robust_level in enumerate(robust_levels):
        def f(output, seed):
            np.random.seed(seed)
            robust_params = list(ROBUST)
            robust_params[noise_axis] = robust_level
            robust_params = tuple(robust_params)
            N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT = sample_sizes
            distance_threshold = K_CLOSE
            N, MEANS, Rewards_, Nstates_indices, Nstates_activation, my_Rtree = SamplingCartPole.radialBasisFunctionsAveragers(env, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, NB_ACTIONS, distance_threshold, VAR, robustness=robust_params)
            D = Rewards_, Nstates_indices, Nstates_activation
            agent = ADP.RVI_threshold_RBF(DISCOUNT, D, MAX_ITER, MEANS, VAR, NB_ACTIONS, N, my_Rtree, distance_threshold, DELTA_INIT)
            w, info = agent.solve_robust(GREED, STOP)
            durations, cumulated_rewards = RunningCartPole.run_value_agent_threshold_RBF(agent, env, N_tests, DISCOUNT, NB_ACTIONS, robustness=robust_params)
            output.put((np.mean(durations), np.mean(cumulated_rewards)))
            return
            
        output_queue = mp.Queue()
        
        for loop in range(N_trainings // 10):
            processes = [mp.Process(target=f, args=(output_queue,_ + loop * 10,)) for _ in range(10)]
            for p in processes:
                p.start()
            for j in range(10):
                results = output_queue.get()
                Lengths[i, loop * 10 + j] = results[0]
                Rewards[i, loop * 10 + j] = results[1]
            for p in processes:
                p.join()
        
        if N_trainings%10 > 0:
            processes = [mp.Process(target=f, args=(output_queue, N_trainings - N_trainings%10 + x)) for x in range(N_trainings%10)]
            for p in processes:
                p.start()
            for j in range(N_trainings%10):
                results = output_queue.get()
                Lengths[i, N_trainings - N_trainings%10 + j] = results[0]
                Rewards[i, N_trainings - N_trainings%10 + j] = results[1]
            for p in processes:
                p.join()
    if save:
        np.save('./Arrays/' + str(ID) + 'Lengths', np.mean(Lengths, axis=1))
        np.save('./Arrays/' + str(ID) + 'LengthsStd', np.std(Lengths, axis=1) * 1.96 / N_trainings)
        np.save('./Arrays/' + str(ID) + 'Rewards', np.mean(Rewards, axis=1))
        np.save('./Arrays/' + str(ID) + 'RewardsStd', np.std(Rewards, axis=1) * 1.96 / N_trainings)
    plt.figure(1)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average sum of Discounted Rewards')
    V1 = np.mean(Rewards, axis=1)
    eV1 = np.std(Rewards, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V1, eV1, ecolor='red', label='RVIRBF', color='blue')
    plt.title('Discounted Rewards')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Rewards')
    plt.close()
    plt.figure(2)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average Length of runs')
    V2 = np.mean(Lengths, axis=1)
    eV2 = np.std(Lengths, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V2, eV2, ecolor='red', label='RVIRBF', color='blue')
    plt.title('Lengths')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Lengths')
    plt.close()
    return

    
def VIRBFParallel(sampling, ID, DISCOUNT, DELTA_INIT, GREED, STOP, DELTA_PHI, N_trainings, robust_levels, N_tests, MAX_ITER, INSIDE_LOOP_ITER, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, K_CLOSE, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, Nrbf, VAR, save, ROBUST):
    Lengths = np.zeros((len(robust_levels), N_trainings))
    Rewards = np.zeros((len(robust_levels), N_trainings))
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    sample_sizes = DIC_SAMPLES[N_RBF_X]
    if ROBUST[1]==-1:
        noise_axis = 1
    else:
        noise_axis = 2
    for i, robust_level in enumerate(robust_levels):
        def f(output, seed):
            np.random.seed(seed)
            robust_params = list(ROBUST)
            robust_params[noise_axis] = robust_level
            robust_params = tuple(robust_params)
            N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT = sample_sizes
            distance_threshold = K_CLOSE
            N, MEANS, Rewards_, Nstates_indices, Nstates_activation, my_Rtree = SamplingCartPole.radialBasisFunctionsAveragers(env, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, NB_ACTIONS, distance_threshold, VAR, robustness=robust_params)
            D = Rewards_, Nstates_indices, Nstates_activation
            agent = ADP.RVI_threshold_RBF(DISCOUNT, D, MAX_ITER, MEANS, VAR, NB_ACTIONS, N, my_Rtree, distance_threshold, DELTA_INIT)
            w, info = agent.solve_nominal(GREED, STOP)
            durations, cumulated_rewards = RunningCartPole.run_value_agent_threshold_RBF(agent, env, N_tests, DISCOUNT, NB_ACTIONS, robustness=robust_params)
            output.put((np.mean(durations), np.mean(cumulated_rewards)))
        
        output_queue = mp.Queue()
        
        for loop in range(N_trainings // 10):
            processes = [mp.Process(target=f, args=(output_queue,_ + loop * 10,)) for _ in range(10)]
            for p in processes:
                p.start()
            for j in range(10):
                results = output_queue.get()
                Lengths[i, loop * 10 + j] = results[0]
                Rewards[i, loop * 10 + j] = results[1]
            for p in processes:
                p.join()
        
        if N_trainings%10 > 0:
            processes = [mp.Process(target=f, args=(output_queue, N_trainings - N_trainings%10 + x)) for x in range(N_trainings%10)]
            for p in processes:
                p.start()
            for j in range(N_trainings%10):
                results = output_queue.get()
                Lengths[i, N_trainings - N_trainings%10 + j] = results[0]
                Rewards[i, N_trainings - N_trainings%10 + j] = results[1]
            for p in processes:
                p.join()

    if save:
        np.save('./Arrays/' + str(ID) + 'Lengths', np.mean(Lengths, axis=1))
        np.save('./Arrays/' + str(ID) + 'LengthsStd', np.std(Lengths, axis=1) * 1.96 / N_trainings)
        np.save('./Arrays/' + str(ID) + 'Rewards', np.mean(Rewards, axis=1))
        np.save('./Arrays/' + str(ID) + 'RewardsStd', np.std(Rewards, axis=1) * 1.96 / N_trainings)
    plt.figure(1)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average sum of Discounted Rewards')
    V1 = np.mean(Rewards, axis=1)
    eV1 = np.std(Rewards, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V1, eV1, ecolor='red', label='VIRBF', color='blue')
    plt.title('Discounted Rewards')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Rewards')
    plt.close()
    plt.figure(2)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average Length of runs')
    V2 = np.mean(Lengths, axis=1)
    eV2 = np.std(Lengths, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V2, eV2, ecolor='red', label='VIRBF', color='blue')
    plt.title('Lengths')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Lengths')
    plt.close()
    return

    
def VITilesParallel(sampling, ID, DISCOUNT, DELTA_INIT, GREED, STOP, DELTA_PHI, N_trainings, robust_levels, N_tests, MAX_ITER, INSIDE_LOOP_ITER, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, K_CLOSE, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, Nrbf, VAR, save, ROBUST):
    Lengths = np.zeros((len(robust_levels), N_trainings))
    Rewards = np.zeros((len(robust_levels), N_trainings))
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    env.reset()
    sample_sizes = DIC_SAMPLES[N_RBF_X]
    if ROBUST[1]==-1:
        noise_axis = 1
    else:
        noise_axis = 2
    for i, robust_level in enumerate(robust_levels):
        def f(output, seed):
            np.random.seed(seed)
            robust_params = list(ROBUST)
            robust_params[noise_axis] = robust_level
            robust_params = tuple(robust_params)
            N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT = sample_sizes
            size, rewards, next_states, coding = SamplingCartPole.stateAggregationGrid(env, N_RBF_X, N_RBF_XDOT, N_RBF_THETA, N_RBF_THETADOT, NB_ACTIONS, MAX_SIZE, NB_TILINGS, WIDTH, robustness=robust_params)
            D = rewards, next_states
            agent = ADP.RVI_Tiles(DISCOUNT, D, MAX_ITER, NB_ACTIONS, size, coding, NB_TILINGS, WIDTH, DELTA_INIT)
            w, info = agent.solve_nominal(GREED, STOP)
            durations, cumulated_rewards = RunningCartPole.run_value_agent_Tiles(agent, env, N_tests, DISCOUNT, NB_ACTIONS, NB_TILINGS, robustness=robust_params)
            output.put((np.mean(durations), np.mean(cumulated_rewards)))
            return

        output_queue = mp.Queue()
        
        for loop in range(N_trainings // 10):
            processes = [mp.Process(target=f, args=(output_queue,_ + loop * 10,)) for _ in range(10)]
            for p in processes:
                p.start()
            for j in range(10):
                results = output_queue.get()
                Lengths[i, loop * 10 + j] = results[0]
                Rewards[i, loop * 10 + j] = results[1]
            for p in processes:
                p.join()
        
        if N_trainings%10 > 0:
            processes = [mp.Process(target=f, args=(output_queue, N_trainings - N_trainings%10 + x)) for x in range(N_trainings%10)]
            for p in processes:
                p.start()
            for j in range(N_trainings%10):
                results = output_queue.get()
                Lengths[i, N_trainings - N_trainings%10 + j] = results[0]
                Rewards[i, N_trainings - N_trainings%10 + j] = results[1]
            for p in processes:
                p.join()
    if save:
        np.save('./Arrays/' + str(ID) + 'Lengths', np.mean(Lengths, axis=1))
        np.save('./Arrays/' + str(ID) + 'LengthsStd', np.std(Lengths, axis=1) * 1.96 / N_trainings)
        np.save('./Arrays/' + str(ID) + 'Rewards', np.mean(Rewards, axis=1))
        np.save('./Arrays/' + str(ID) + 'RewardsStd', np.std(Rewards, axis=1) * 1.96 / N_trainings)
    plt.figure(1)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average sum of Discounted Rewards')
    V1 = np.mean(Rewards, axis=1)
    eV1 = np.std(Rewards, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V1, eV1, ecolor='red', label='VITiles', color='blue')
    plt.title('Discounted Rewards')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Rewards')
    plt.close()
    plt.figure(2)
    x_ax = robust_levels
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(robust_levels))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average Length of runs')
    V2 = np.mean(Lengths, axis=1)
    eV2 = np.std(Lengths, axis=1) * 1.96 / N_trainings
    plt.errorbar(x_ax, V2, eV2, ecolor='red', label='VITiles', color='blue')
    plt.title('Lengths')
    plt.legend()
    plt.savefig('./Plots/' + str(ID) + 'Lengths')
    plt.close()
    return
    
DIC_ALGO = {'RVITiles':RVITilesParallel, 'RVIRBF':RVIRBFParallel, 'VITiles':VITilesParallel, 'VIRBF':VIRBFParallel}
DIC_SAMPLES = {0: (5, 5, 5, 5), 1: (7, 7, 7, 7), 2: (10, 10, 10, 10), 3: (12, 12, 12, 12), 4: (15, 15, 15, 15), 5:(17, 17, 17, 17), 6:(20, 20, 20, 20), 7:(25, 25, 25, 25), 8:(30, 30, 30, 30), 9:(35, 35, 35, 35), 10:(40, 40, 40, 40)}

#args = -algo
def plotter(args):
    options, others = getopt(args, 'i:a:s:')
    filepath = others[0]
    id = options[0][1]
    algo = options[1][1]
    save_arrays = options[2][1]
    if save_arrays == 'y':
        save_arrays = True
    else:
        save_arrays = False
    times = []
    print(id)
    print(algo)
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        print(next(reader))
        for number, row in enumerate(reader):
            print(row)
            start = time.time()
            params = []
            vars = []
            robustness_parameters = []
            for i, string in enumerate(row):
                if i==0:
                    params.append(string)
                elif i <= 5:
                    params.append(float(string))
                elif i <= 20:
                    if i == 7:
                        liste = string.split(",")
                        sizes = []
                        for size in liste:
                            sizes.append(float(size))
                        params.append(sizes)
                    elif i == 15:
                        params.append(float(string))
                    else:
                        params.append(int(string))
                elif i <= 24:
                    vars.append(float(string))
                else:
                    if i == 25:
                        robustness_parameters.append(int(string))
                    else:
                        robustness_parameters.append(float(string))
            realid = id + str(number)
            VAR = 2 * np.array(vars) ** 2
            robustness_parameters = tuple(robustness_parameters)
            print(params)
            print(vars)
            DIC_ALGO[algo](params[0], realid, params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], VAR, save_arrays, robustness_parameters)
            times.append(time.time() - start)
    with open('./Plots/Timetaken'+str(id)+'.txt', 'w') as file:
        for x in times:
            file.write(str(x) + '\n')
        file.write(str(np.sum(times) % 60) + '\n')
    return
    
if __name__ == "__main__":
   plotter(sys.argv[1:])