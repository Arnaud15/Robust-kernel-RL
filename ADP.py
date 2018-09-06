#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 07:53:30 2018

@author: arnaudautef
"""
import numpy as np
import time
from tqdm import tqdm
from TilesWrapperCartPole import MyCoding
import scipy.sparse
import scipy.sparse.linalg
import queue as Q
from rtree import index
from multiprocessing.dummy import Pool as ThreadPool
import itertools

class ADP(object):
    def __init__(self, discount, D, max_iter, nb_actions, delta_init):
        self.discount = discount
        self.max_iter = max_iter
        self.iter = 0
        self.time = None
        self.delta_init = delta_init # to initialize each loop with a full-rank matrix
        self.nb_actions = nb_actions
        self.D = D


class ADP_Tiles(ADP):
    def __init__(self, discount, D, max_iter, nb_actions, max_size, nb_tilings, width, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)
        self.size = max_size # max_size * nb_tilings
        self.max_size = max_size
        self.nb_tilings = nb_tilings
        self.weights = np.zeros(self.size * self.nb_actions)
        self.tiles = MyCoding(max_size, nb_tilings, width)
    

    def phi(self, state, action):
        x, x_dot, theta, theta_dot = state
        return [idx + self.size * action for idx in self.tiles.mytiles([x, x_dot, theta, theta_dot])] # [idx  + (self.max_size * j) + self.size * action for j, idx in enumerate(self.tiles.mytiles(x, x_dot, theta, theta_dot))]
    
    def solve(self, greed_level, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []

        D_states, D_actions, D_rewards, D_nstates = self.D
        N = len(D_states)

        b = np.zeros(self.size * self.nb_actions)
        for i in range(N):
            for idx in self.phi(D_states[i], D_actions[i]):
                b[idx] += D_rewards[i] / self.nb_tilings

        self.weights = np.zeros(self.size * self.nb_actions)
        w = np.zeros(self.size * self.nb_actions)
        while True:
            self.iter += 1
            B = np.identity(self.size * self.nb_actions) * self.delta_init
            for i in range(N):
                s, a, r, s_prime = D_states[i], D_actions[i], D_rewards[i], D_nstates[i]
                V1 = set(self.phi(s, a))
                arg = np.random.randint(self.nb_actions)
                decision = np.random.binomial(1, greed_level)
                if not decision:
                    arg = self.policy(s_prime)[0]
                V_prime = set(self.phi(s_prime, arg))
                for idx in V1:
                    for id1 in V1 & V_prime:
                        B[idx, id1] += (1 - self.discount) / (self.nb_tilings ** 2)
                    for id2 in V1 - V_prime:
                        B[idx, id2] += 1 / (self.nb_tilings ** 2)
                    for id3 in V_prime - V1:
                        B[idx, id3] -= self.discount / (self.nb_tilings ** 2)
            B = scipy.sparse.csr_matrix(B)
            w = scipy.sparse.linalg.lsqr(B,b)[0]
            gap = np.linalg.norm(w - self.weights)
            evolution.append(gap)
            self.weights = np.copy(w)
            pb.update(1)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break
        pb.close()
        info['time'] = time.time() - self.time
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info

    def policy(self, state):
        action = np.random.randint(0, self.nb_actions)
        val = np.sum(self.weights[self.phi(state, action)]) / self.nb_tilings
        for act in range(0, self.nb_actions):
            val_new = np.sum(self.weights[self.phi(state, act)]) / self.nb_tilings
            if (val_new > val):
                val = val_new
                action = act
        return action, val


class ARDP_Tiles2(ADP):
    def __init__(self, discount, D, max_iter, inside_iter, nb_actions, max_size, nb_tilings, width, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)
        self.inside_iter = inside_iter

        self.size = max_size #* nb_tilings
        self.max_size = max_size
        self.nb_tilings = nb_tilings
        self.tiles = MyCoding(max_size, nb_tilings, width)

        self.weights = np.zeros(self.size * self.nb_actions)


    def phi(self, state, action):
        x, x_dot, theta, theta_dot = state
        return [idx  + self.size * action for idx in self.tiles.mytiles([x, x_dot, theta, theta_dot])]# [idx  + (self.max_size * j) + self.size * action for j, idx in enumerate(self.tiles.mytiles(x, x_dot, theta, theta_dot))]
    
    def aggregator(self, delta_phi):
        dict_delta = {}
        d = 1 / self.nb_tilings

        D_states, D_actions, D_rewards, D_nstates = self.D
        sample_size = len(D_states)

        monitoring = np.zeros(sample_size)
        for i in tqdm(range(sample_size)):
            s_i = D_states[i]
            a_i = D_actions[i]
            key_i = (s_i[0], s_i[1], s_i[2], s_i[3], a_i)
            if key_i not in dict_delta:
                dict_delta[key_i] = {i}
                monitoring[i] += 1
            V1 = self.phi(s_i, a_i)
            for j in range(sample_size):
                s_j = D_states[j]
                a_j = D_actions[j]
                if a_i != a_j:
                    continue
                V2 = self.phi(s_j, a_j)
                distance_i_j = d * sum([V1[t]!=V2[t] for t in range(self.nb_tilings)])
                if (distance_i_j<=delta_phi):
                    dict_delta[key_i].add(j)
                    monitoring[i] += 1
                else:
                    continue
        print('--------')
        print(np.mean(monitoring))
        return dict_delta
    
    '''
    def support_function(self, V, confidence):
        # Values of possible next states are sorted
        sorted_values = np.argsort(V)
        nb_small_states = sorted_values.shape[0]
        # The probability distribution to be shifted
        output = np.ones(nb_small_states) / nb_small_states
        # Size of the shift in L1 norm, nothing to do if the least favorable state transition is certain
        eps = np.minimum(1 - output[sorted_values[0]], confidence * 0.5)
        output[sorted_values[0]] = output[sorted_values[0]] + eps
        marker = nb_small_states - 1
        while eps > 0.001:
            temp = np.minimum(output[sorted_values[marker]], eps)
            output[sorted_values[marker]] = output[sorted_values[marker]] - temp
            eps = eps - temp
            marker -= 1
        return output
    '''

    def solve(self, greed, stopping_criterion, delta_phi):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        D_states, D_actions, D_rewards, D_nstates = self.D
        N = len(D_states)

        k = self.size * self.nb_actions
        self.weights = np.zeros(self.size * self.nb_actions)

        print('Beginning of aggregation--------')
        start = time.time()
        dict_delta = self.aggregator(delta_phi)
        print('aggregation took an extra: '+ str(time.time() - start) + ' seconds')

        M1 = np.identity(k) * self.delta_init
        for i in range(N):
            s, a, r = D_states[i], D_actions[i], D_rewards[i]
            V1 = self.phi(s, a)
            for idx in V1:
                for idy in V1:
                    M1[idx, idy] += 1 / (self.nb_tilings ** 2)
        M1 = scipy.sparse.csr_matrix(M1)

        while True:

            theta = np.zeros(k)
            self.iter += 1

            for j in range(self.inside_iter):
                c = np.zeros(k)
                dict_iteration = {}
                for i in range(N):
                    s, a = D_states[i], D_actions[i]
                    V1 = self.phi(s, a)
                    neighbors = dict_delta[(s[0], s[1], s[2], s[3], a)]
                    #values = []
                    value = np.inf
                    for indice in neighbors:
                        s_ = D_states[indice]
                        r = D_rewards[indice]
                        neighbor_key = (s_[0], s_[1], s_[2], s_[3], a)
                        if neighbor_key in dict_iteration:
                            dot_product = dict_iteration[neighbor_key]
                        else:
                            s_prime = D_nstates[indice]
                            arg = np.random.randint(self.nb_actions)
                            decision = np.random.binomial(1, greed)
                            if not decision:
                                arg = self.policy(s_prime)[0]                        
                            V_prime = self.phi(s_prime, arg)
                            dot_product = np.sum(theta[V_prime]) / self.nb_tilings
                            dict_iteration[neighbor_key] = dot_product
                        value_temp = r + self.discount * dot_product
                            #values.append(r + self.discount * np.sum(teta[V_prime]) / self.nb_tilings)
                        if value_temp < value:
                            value = value_temp
                    #value = np.dot(self.support_function(values, confidence), values)
                    for idx in V1:
                        c[idx] +=  value / self.nb_tilings
                ntheta = scipy.sparse.linalg.lsqr(M1, c)[0]
                print('inside gap at step ' + str(j+1) + ': ' + str(np.linalg.norm(ntheta - theta)))
                theta = np.copy(ntheta)

            gap = np.linalg.norm(theta - self.weights)
            print('gap global iteration: ' + str(gap))
            #print('gap threshold: ' + str(stopping_criterion))
            evolution.append(gap)
            self.weights = np.copy(theta)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
    

    def policy(self, state):
        action = np.random.randint(0, self.nb_actions)
        val = np.sum(self.weights[self.phi(state, action)]) / self.nb_tilings
        for act in range(0, self.nb_actions):
            val_new = np.sum(self.weights[self.phi(state, act)]) / self.nb_tilings
            if (val_new > val):
                val = val_new
                action = act
        return action, val


class API_RBF(ADP):
    def __init__(self, discount, D, max_iter, mean, var, nb_actions, num_RBF, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.mean = mean
        self.var = var
        self.number = num_RBF

        self.weights = np.zeros(self.number * self.nb_actions)

    def phi(self, state, action):
        result = np.zeros(self.number * self.nb_actions)
        result[action * (self.number) : (action + 1) * self.number] = np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
        #result = result / np.sum(result)
        return result

    def solve(self, greed_level, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        D_states, D_actions, D_rewards, D_nstates = self.D

        N = len(D_states)
        k = self.nb_actions * self.number

        b = np.zeros(k)
        for i in range(N):
            b += self.phi(D_states[i], D_actions[i]) * D_rewards[i]

        self.weights = np.zeros(self.number * self.nb_actions)
        w = np.zeros(self.number * self.nb_actions)

        while True:
            self.iter += 1
            B = self.delta_init * np.identity(k)

            for i in tqdm(range(N)):
                s, a, r, s_prime = D_states[i], D_actions[i], D_rewards[i], D_nstates[i]
                V1 = self.phi(s, a)
                arg = np.random.randint(self.nb_actions)
                decision = np.random.binomial(1, greed_level)
                if not decision:
                    arg = self.policy(s_prime)[0]
                V_prime = self.phi(s_prime, arg)
                A = V1 - self.discount * V_prime
                A = A.reshape(1,k)
                V1 = V1.reshape(k,1)
                B = B + V1 @ A

            w = np.linalg.lstsq(B,b,rcond=None)[0]

            gap = np.linalg.norm(w - self.weights)
            evolution.append(gap)
            self.weights = np.copy(w)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        info['time'] = time.time() - self.time
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info        


    def policy(self, state):
        action = np.random.randint(self.nb_actions)
        val = self.phi(state, action).dot(self.weights)
        for act in range(self.nb_actions):
            val_new = self.phi(state, act).dot(self.weights)
            if (val_new > val):
                val = val_new
                action = act
        return action, val


class ARPI_RBF(ADP):
    def __init__(self, discount, D, max_iter, inside_iter, mean, var, nb_actions, num_RBF, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)
        self.inside_iter = inside_iter

        self.var = var
        self.mean = mean
        self.number = num_RBF

        self.weights = np.zeros(self.number * self.nb_actions)
    
    def phi(self, state, action):
        result = np.zeros(self.number * self.nb_actions)
        result[action * self.number :(action + 1) * self.number] = np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
        result = result / np.sum(result)
        return result

    def aggregator(self, delta_phi):
        dict_delta = {}
        dict_phi = {}

        D_states, D_actions, D_rewards, D_nstates = self.D
        sample_size = len(D_states)

        monitoring = np.zeros(sample_size)

        for i in tqdm(range(sample_size)):
            s_i = D_states[i]
            a_i = D_actions[i]
            key_i = (s_i[0], s_i[1], s_i[2], s_i[3], a_i)

            if key_i not in dict_delta:
                dict_delta[key_i] = {i}
                monitoring[i] += 1

            if key_i not in dict_phi:
                V1 = self.phi(s_i, a_i)
                dict_phi[key_i] = V1
            else:
                V1 = dict_phi[key_i]

            for j in range(sample_size):
                s_j = D_states[j]
                a_j = D_actions[j]
                if a_i != a_j:
                    continue
                key_j = (s_j[0], s_j[1], s_j[2], s_j[3], a_j)

                if key_j not in dict_phi:
                    V2 = self.phi(s_j, a_j)
                    dict_phi[key_j] = V2
                else:
                    V2 = dict_phi[key_j]

                distance_i_j = np.linalg.norm(V1 - V2)
                if (distance_i_j<=delta_phi):
                    dict_delta[key_i].add(j)
                    monitoring[i] += 1
                else:
                    continue

        print('--------')
        print(np.mean(monitoring))
        return dict_delta
    
    '''
    def support_function(self, V, confidence):
        # Values of possible next states are sorted
        sorted_values = np.argsort(V)
        nb_small_states = sorted_values.shape[0]
        # The probability distribution to be shifted
        output = np.ones(nb_small_states) / nb_small_states
        # Size of the shift in L1 norm, nothing to do if the least favorable state transition is certain
        eps = np.minimum(1 - output[sorted_values[0]], confidence * 0.5)
        output[sorted_values[0]] = output[sorted_values[0]] + eps
        marker = nb_small_states - 1
        while eps > 0.001:
            temp = np.minimum(output[sorted_values[marker]], eps)
            output[sorted_values[marker]] = output[sorted_values[marker]] - temp
            eps = eps - temp
            marker -= 1
        return output
    '''

    def solve(self, greed, stopping_criterion, delta_phi):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        D_states, D_actions, D_rewards, D_nstates = self.D
        N = len(D_states)

        k = self.nb_actions * self.number
        self.weights = np.zeros(k)

        print('Beginning of aggregation--------')
        start = time.time()
        dict_delta = self.aggregator(delta_phi)
        print('aggregation took an extra: '+ str(time.time() - start) + ' seconds')

        M1 = np.identity(k) * self.delta_init
        for i in range(N):
            s, a, r = D_states[i], D_actions[i], D_rewards[i]
            V1 = self.phi(s, a)
            M1 = M1 + (V1.reshape(k,1) @ V1.reshape(1,k))

        while True:
            teta = np.zeros(k)
            self.iter += 1

            for j in range(self.inside_iter):
                c = np.zeros(k)
                dict_iteration = {}
                for i in range(N):
                    s, a = D_states[i], D_actions[i]
                    V1 = self.phi(s, a)
                    neighbors = dict_delta[(s[0], s[1], s[2], s[3], a)]
                    #values = []
                    value = np.inf
                    for indice in neighbors:
                        s_ = D_states[indice]
                        r = D_rewards[indice]
                        neighbor_key = (s_[0], s_[1], s_[2], s_[3], a)
                        if neighbor_key in dict_iteration:
                            dot_product = dict_iteration[neighbor_key]
                        else:
                            s_prime = D_nstates[indice]
                            arg = np.random.randint(self.nb_actions)
                            decision = np.random.binomial(1, greed)
                            if not decision:
                                arg = self.policy(s_prime)[0]                        
                            V_prime = self.phi(s_prime, arg)
                            dot_product = np.dot(V_prime, teta)
                            dict_iteration[neighbor_key] = dot_product
                        #values.append(r + self.discount * dot_product)
                        value_temp = r + self.discount * dot_product
                        if value_temp < value:
                            value = value_temp
                    #value = np.dot(self.support_function(values, confidence), values)
                    c += value * V1

                nteta = np.linalg.lstsq(M1, c, rcond=None)[0]
                print('inside gap at step ' + str(j+1) + ': ' + str(np.linalg.norm(nteta - teta)))
                teta = np.copy(nteta)

            gap = np.linalg.norm(teta - self.weights)
            print('gap global iteration: ' + str(gap))
            print('gap threshold: ' + str(stopping_criterion))
            evolution.append(gap)
            self.weights = np.copy(teta)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
        
    def policy(self, state):
        action = np.random.randint(self.nb_actions)
        val = self.phi(state, action).dot(self.weights)
        for i in range(self.nb_actions):
            val_new = self.phi(state, i).dot(self.weights)
            if (val_new > val):
                val = val_new
                action = i
        return action, val


class RVI_RBF(ADP):
    def __init__(self, discount, D, max_iter, mean, var, nb_actions, num_RBF, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.var = var
        self.mean = mean
        self.number = num_RBF

        self.weights = np.zeros(self.number)
    
    def phi(self, state):
        result = np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
        result = result / np.sum(result)
        return result

    '''
    def create_memberships(self, k_close, assess_delta=True):
        print('Beginning of aggregation--------')
        start = time.time()
        dict_memberships = {}
        dict_phi = {}
        D_states, D_actions, D_rewards, D_nstates = self.D
        abstract_states = self.mean
        k = abstract_states.shape[0]
        sample_size = len(D_states)
        for j in range(k):
            V_j = self.phi(abstract_states[j])
            dict_phi[j] = V_j
            for act in range(self.nb_actions):
                dict_memberships[(j, act)] = Q.PriorityQueue()
        for i in tqdm(range(sample_size)):
            s_i = D_states[i]
            a_i = D_actions[i]
            V_i = self.phi(s_i)
            for j in range(k):
                distance_i_j = np.linalg.norm(V_i - dict_phi[j])
                dict_memberships[(j, a_i)].put((distance_i_j, i))
        for j in range(k):
            for act in range(self.nb_actions):
                q = dict_memberships[(j, act)]
                table = []
                for m in range(k_close):
                    table.append(q.get()[1])
                dict_memberships[(j, act)] = table
        if assess_delta:
            membership_sizes = []
            for j in tqdm(range(k)):
                for act in range(self.nb_actions):
                    membership_sizes.append(len(dict_memberships[(j, act)]))
            print(np.min(membership_sizes))
            print(np.max(membership_sizes))
            print(np.std(membership_sizes))
        self.dict_memberships = dict_memberships
        print('aggregation took an extra: '+ str(time.time() - start) + ' seconds')
        return dict_memberships

    def solve(self, greed, stopping_criterion, k_close):
        """
        Implementation of the ARPI algorithm of Aviv Tamar, Shie Mannor and Huan Xu
        """
        info = {}
        self.iter = 0
        self.time = time.time()
        D_states, D_actions, D_rewards, D_nstates = self.D
        N = len(D_states)
        k = self.number
        pb = tqdm(total=self.max_iter)
        evolution = []
        w = np.zeros(k)
        while True:
            self.iter += 1
            w_temp = np.zeros(k)
            dict_phi = {}
            for j in range(k):
                value_max = - np.inf
                test2 = True
                for act in range(self.nb_actions):
                    value_min = np.inf
                    test = True
                    for i in self.dict_memberships[(j, act)]:
                        test = False
                        if i in dict_phi.keys():
                            value_temp = dict_phi[i]
                        else:
                            value_temp = D_rewards[i] + self.discount * np.dot(w, self.phi(D_nstates[i]))
                            dict_phi[i] = value_temp
                        if value_temp < value_min:
                            value_min = value_temp
                        
                    if test:
                        print('there is an issue with ' + str((j, act)))
                    if value_min > value_max:
                        test2 = False
                        value_max = value_min
                if test2:
                    print('there is an issue with ' + str(j) + ' at iter ' + str(self.iter))
                w_temp[j] = value_max
            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            #print('gap iteration: ' + str(gap))
            #print('gap threshold: ' + str(stopping_criterion))
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break
        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
    '''
    
    def solve_robust(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []

        kernels_rewards, kernels_next_states = self.D
        size_robustness = kernels_rewards.shape[2]

        w = np.zeros(self.number)

        while True:
            self.iter += 1
            w_temp = np.zeros(self.number)
            dict_phi = {}

            for j in range(self.number):
                value_max = - np.inf
                for act in range(self.nb_actions):
                    value_min = np.inf
                    for i in range(size_robustness):
                        value_temp = kernels_rewards[j, act, i]+ self.discount * np.dot(w, self.phi(kernels_next_states[j, act, i, :]))
                        if value_temp < value_min:
                            value_min = value_temp
                    if value_min > value_max:
                        value_max = value_min
                w_temp[j] = value_max

            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
        
        
    def solve_nominal(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []

        kernels_rewards, kernels_next_states = self.D

        w = np.zeros(self.number)
        while True:
            self.iter += 1
            w_temp = np.zeros(self.number)
            dict_phi = {}
            for j in range(self.number):
                value_max = -np.inf
                for act in range(self.nb_actions):
                    value_temp = kernels_rewards[j, act] + self.discount * np.dot(w, self.phi(kernels_next_states[j, act, :]))
                    if value_temp > value_max:
                        value_max = value_temp
                w_temp[j] = value_max
            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info

    
class RVI_threshold_RBF(ADP):
    def __init__(self, discount, D, max_iter, mean, var, nb_actions, num_RBF, myrtree, d, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.var = var
        self.mean = mean
        self.number = num_RBF

        self.my_rtree = myrtree
        self.d_threshold = d
        
        self.kernels_rewards = self.D[0]
        self.kernels_next_states_indices = self.D[1]
        self.kernels_next_states_distrib = self.D[2]

        self.weights = np.zeros(self.number)
    
    def phi(self, state):
        x, xdot, theta, thetadot = state
        id_kernels_activated_state = list(self.my_rtree.intersection((x - self.d_threshold, xdot - self.d_threshold, theta - self.d_threshold / 10, thetadot - self.d_threshold, x + self.d_threshold, xdot + self.d_threshold, theta + self.d_threshold / 10, thetadot + self.d_threshold)))
        if not id_kernels_activated_state:
            id_kernels_activated_state = list(self.my_rtree.nearest((x, xdot, theta, thetadot, x, xdot, theta, thetadot), 1))
        states_kernels_activated_state = np.array([self.mean[indice] for indice in id_kernels_activated_state])
        kernels_activation_nstate = np.exp(-np.sum((states_kernels_activated_state - state) ** 2 / self.var, axis=1))
        kernels_activation_nstate = kernels_activation_nstate / np.sum(kernels_activation_nstate)
        return id_kernels_activated_state, kernels_activation_nstate

    def nominalBellmanUpdate(self, j):
        value_min = np.inf
        x, xdot, theta, thetadot = self.mean[j]
        value_max = - np.inf
        for act in range(self.nb_actions):
            value = self.kernels_rewards[j, act] + self.discount * np.dot(self.kernels_next_states_distrib[j][act], self.weights[self.kernels_next_states_indices[j][act]])
            if value > value_max:
                value_max = value
        return value_max
    
    def solve_nominal_parallel(self, greed, stopping_criterion, nb_processes=20):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []


        kernels_rewards, kernels_next_states_indices, kernels_next_states_distrib = self.D

        self.weights = np.zeros(self.number)
        with Pool(processes=nb_processes) as p:
            while True:
                self.iter += 1
                results = list(p.map(self.nominalBellmanUpdate, range(self.number)))
                w_temp = np.array(results)
                pb.update(1)
                gap = np.linalg.norm(self.weights - w_temp)
                evolution.append(gap)
                self.weights = np.copy(w_temp)
                if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                    break

        pb.close()
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
    
    def solve_robust(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []
        kernels_rewards, kernels_next_states_indices, kernels_next_states_distrib = self.D

        w = np.zeros(self.number)

        while True:
            self.iter += 1
            w_temp = np.zeros(self.number)
            dict_iteration = {}

            for j in range(self.number):
                value_min = np.inf
                x, xdot, theta, thetadot = self.mean[j]
                neighbours = list(self.my_rtree.intersection((x - self.d_threshold, xdot - self.d_threshold, theta - self.d_threshold / 10, thetadot - self.d_threshold, x + self.d_threshold, xdot + self.d_threshold, theta + self.d_threshold / 10, thetadot + self.d_threshold)))
                for kernel_indice in neighbours:
                    value_max = - np.inf
                    for act in range(self.nb_actions):
                        my_key = (kernel_indice, act)
                        if my_key in dict_iteration:
                            dot_product = dict_iteration[my_key]
                        else:
                            dot_product = self.discount * np.dot(kernels_next_states_distrib[kernel_indice][act], w[kernels_next_states_indices[kernel_indice][act]])
                            dict_iteration[my_key] = dot_product
                        value_temp = kernels_rewards[kernel_indice, act] + dot_product
                        if value_temp > value_max:
                            value_max = value_temp
                    if value_max < value_min:
                        value_min = value_max
                w_temp[j] = value_min
            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
        
        
    def solve_nominal(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []


        kernels_rewards, kernels_next_states_indices, kernels_next_states_distrib = self.D

        w = np.zeros(self.number)

        while True:
            self.iter += 1
            w_temp = np.zeros(self.number)

            for j in range(self.number):
                value_min = np.inf
                x, xdot, theta, thetadot = self.mean[j]
                value_max = - np.inf
                for act in range(self.nb_actions):
                    value = kernels_rewards[j, act] + self.discount * np.dot(kernels_next_states_distrib[j][act], w[kernels_next_states_indices[j][act]])
                    if value > value_max:
                        value_max = value
                w_temp[j] = value_max

            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info

        
class RVI_Tiles(ADP):
    def __init__(self, discount, D, max_iter, nb_actions, size, tile_coding, nb_tilings, width, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.size = size
        self.tiles = tile_coding
        self.nb_tilings = nb_tilings
        self.width = width

        self.weights = np.zeros(self.size)



    def phi(self, state):
        x, x_dot, theta, theta_dot = state
        indices = self.tiles.mytiles([x, x_dot, theta, theta_dot])
        return indices

    def solve_robust(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        Rewards, New_states = self.D
        k = self.size

        w = np.zeros(k)

        pb = tqdm(total=self.max_iter)
        while True:
            self.iter += 1
            w_temp = np.zeros(k)
            dict_iter = {}
            for j in range(k):
                val_max = -np.inf
                for act in range(self.nb_actions):
                    rew_list = Rewards[j][act]
                    n_states_list = New_states[j][act]
                    val_min = np.inf
                    for i, r in enumerate(rew_list):
                        my_key = tuple(n_states_list[i])
                        if my_key in dict_iter:
                            val_temp = r + dict_iter[my_key]
                        else:
                            dot_val = self.discount * np.sum(w[n_states_list[i]])/ self.nb_tilings
                            val_temp = r + dot_val
                            dict_iter[my_key] = dot_val
                        if val_temp < val_min:
                            val_min = val_temp
                    if val_min > val_max:
                        val_max = val_min
                w_temp[j] = val_max

            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
    
    def solve_nominal(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        Rewards, New_states = self.D
        k = self.size

        w = np.zeros(k)

        pb = tqdm(total=self.max_iter)
        while True:
            self.iter += 1
            w_temp = np.zeros(k)
            dict_iter = {}
            for j in range(k):
                val_max = - np.inf
                for act in range(self.nb_actions):
                    rew_list = Rewards[j][act]
                    n_states_list = New_states[j][act]
                    val_mean = 0
                    cpt = 0
                    for i, r in enumerate(rew_list):
                        my_key = tuple(n_states_list[i])
                        if my_key in dict_iter:
                            val_mean += r + dict_iter[my_key]
                        else:
                            dot_val = self.discount * np.sum(w[n_states_list[i]])/ self.nb_tilings
                            val_mean += r + dot_val
                            dict_iter[my_key] = dot_val
                        cpt += 1
                    val_mean /= cpt
                    if val_mean > val_max:
                        val_max = val_mean
                w_temp[j] = val_max
            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info        


class RVI_EA_Tiles(ADP):
    def __init__(self, discount, D, max_iter, nb_actions, size, tile_coding, nb_tilings, width, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.size = size
        self.tiles = tile_coding
        self.nb_tilings = nb_tilings
        self.width = width

        self.weights = np.zeros(self.size)
        return

    def phi(self, state):
        x, x_dot, theta, theta_dot = state
        indices = self.tiles.mytiles([x, x_dot, theta, theta_dot])
        return indices

    def solve_robust(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        evolution = []

        Rewards, New_states = self.D
        k = self.size

        w = np.zeros(k)

        pb = tqdm(total=self.max_iter)
        while True:
            self.iter += 1
            w_temp = np.zeros(k)
            dict_iter = {}
            for j in range(k):
                val_min = np.inf
                n_j = len(Rewards[j][0])
                for i in range(n_j):
                    val_max = -np.inf
                    for act in range(self.nb_actions):
                        my_key = tuple(New_states[j][act][i])
                        if my_key in dict_iter:
                            val_temp = Rewards[j][act][i] + dict_iter[my_key]
                        else:
                            dot_val = self.discount * np.sum(w[New_states[j][act][i]])/ self.nb_tilings
                            val_temp = Rewards[j][act][i] + dot_val
                            dict_iter[my_key] = dot_val
                        if val_temp > val_max:
                            val_max = val_temp
                    if val_max < val_min:
                        val_min = val_max
                w_temp[j] = val_min

            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info


class RVI_EA_RBFs(ADP):
    def __init__(self, discount, D, max_iter, mean, var, nb_actions, num_RBF, delta_init=0.1):
        super().__init__(discount, D, max_iter, nb_actions, delta_init)

        self.var = var
        self.mean = mean
        self.number = num_RBF

        self.weights = np.zeros(self.number)
        return
    
    def phi(self, state):
        result = np.exp(-np.sum((self.mean - state) ** 2 / self.var, axis=1))
        result = result / np.sum(result)
        return result

    def solve_robust(self, greed, stopping_criterion):
        info = {}
        self.iter = 0
        self.time = time.time()
        pb = tqdm(total=self.max_iter)
        evolution = []

        kernels_rewards, kernels_next_states = self.D
        size_robustness = kernels_rewards.shape[2]

        w = np.zeros(self.number)

        while True:
            self.iter += 1
            w_temp = np.zeros(self.number)
            dict_phi = {}

            for j in range(self.number):
                value_min = np.inf
                for i in range(size_robustness):
                    value_max = max([kernels_rewards[j, act, i]+ self.discount * np.dot(w, self.phi(kernels_next_states[j, act, i, :])) for act in range(self.nb_actions)])
                    if value_max < value_min:
                        value_min = value_max
                w_temp[j] = value_min

            pb.update(1)
            gap = np.linalg.norm(w - w_temp)
            evolution.append(gap)
            w = np.copy(w_temp)
            if ((gap <= stopping_criterion) or (self.iter >= self.max_iter)):
                break

        pb.close()
        self.weights = np.copy(w)
        info['time'] = time.time() - self.time
        print(info['time'])
        print('-----end of training-----')
        info['iterations'] = self.iter
        info['deltas'] = evolution
        return self.weights, info
    