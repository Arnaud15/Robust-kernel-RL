import sys, getopt
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
from getopt import getopt
import csv
import time

'''
Performances of an algorithm for comparison are stored under perfs
'''

def comparative_plot(ID, Lengths, LengthsStd, Rewards, RewardsStd, Labels, sample_sizes):
    N = len(Lengths)
    assert (N == len(Rewards) and N == len(LengthsStd) and N == len(RewardsStd)), 'len problem'
    plt.figure(1)
    x_ax = sample_sizes
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(sample_sizes))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average sum of Discounted Rewards')
    for i in range(N):
        V1 = Rewards[i]
        eV1 = RewardsStd[i]
        plt.errorbar(x_ax, V1, eV1, label=Labels[i])
    plt.title('Discounted Rewards')
    plt.legend()
    plt.savefig('./ComparativePlots/' + str(ID) + 'Rewards')
    plt.close()
    plt.figure(2)
    x_ax = sample_sizes
    axes = plt.gca()
    #axes.set_ylim([0,100])
    plt.xticks(list(sample_sizes))
    plt.xlabel('Increasing noise level')
    plt.ylabel('Average Length of runs')
    for i in range(N):
        V2 = Lengths[i]
        eV2 = LengthsStd[i]
        plt.errorbar(x_ax, V2, eV2, label=Labels[i])
    plt.title('Lengths')
    plt.legend()
    plt.savefig('./ComparativePlots/' + str(ID) + 'Lengths')
    plt.close()
    return

def main(sysargs):
    options, others = getopt(sysargs, 'i:')
    big_id = options[0][1]
    filepath = others[0]
    ids = []
    labels = []
    with open(filepath, 'r') as file:
        big_string = file.read()
        args = big_string.splitlines()
        sample_sizes = [ float(string) for string in args[0].split(',')]
        for i in range(0, len(args[1:]), 2):
            ids.append(args[1+i])
            labels.append(args[2+i])
        lengths = []
        rewards = []
        lengthsStd = []
        rewardsStd = []
        for id in ids:
            lengths.append(np.load('./Arrays/' + str(id) + 'Lengths.npy'))
            rewards.append(np.load('./Arrays/' + str(id) + 'Rewards.npy'))
            lengthsStd.append(np.load('./Arrays/' + str(id) + 'LengthsStd.npy'))
            rewardsStd.append(np.load('./Arrays/' + str(id) + 'RewardsStd.npy'))
        comparative_plot(big_id, lengths, lengthsStd, rewards, rewardsStd, labels, sample_sizes)
if __name__ == "__main__":
    main(sys.argv[1:])