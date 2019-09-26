import numpy as np
from copy import deepcopy
from multiprocessing import Process
import gc
import time
import random
import os
import torch

from Node.UCTnode import UCTnode

from Env.EnvWrapper import EnvWrapper

from Mem.CheckpointManager import CheckpointManager

from Policy.PolicyWrapper import PolicyWrapper


class UCT():
    def __init__(self, env_params, max_steps = 1000, max_depth = 20, max_width = 5,
                 gamma = 1.0, policy = "Random", seed = 123, device = torch.device("cpu")):
        self.env_params = env_params
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.max_width = max_width
        self.gamma = gamma
        self.policy = policy
        self.seed = seed
        self.device = device

        self.policy_wrapper = None

        # Environment
        self.wrapped_env = EnvWrapper(**env_params)

        # Environment properties
        self.action_n = self.wrapped_env.get_action_n()
        self.max_width = min(self.action_n, self.max_width)

        assert self.max_depth > 0 and 0 < self.max_width <= self.action_n

        # Checkpoint data manager
        self.checkpoint_data_manager = CheckpointManager()
        self.checkpoint_data_manager.hock_env("main", self.wrapped_env)

        # For MCTS tree
        self.root_node = None
        self.global_saving_idx = 0

        self.init_policy()

    def init_policy(self):
        self.policy_wrapper = PolicyWrapper(
            self.policy,
            self.env_params["env_name"],
            self.action_n,
            self.device
        )

    # Entrance of the P-UCT algorithm
    def simulate_trajectory(self, max_episode_length = -1):
        state = self.wrapped_env.reset()
        accu_reward = 0.0
        done = False
        step_count = 0
        rewards = []
        times = []

        game_start_time = time.time()

        while not done and (max_episode_length == -1 or step_count < max_episode_length):
            simulation_start_time = time.time()
            action = self.simulate_single_move(state)
            simulation_end_time = time.time()

            next_state, reward, done = self.wrapped_env.step(action)
            rewards.append(reward)
            times.append(simulation_end_time - simulation_start_time)

            print("> Time step {}, take action {}, instance reward {}, cumulative reward {}, used {} seconds".format(
                step_count, action, reward, accu_reward + reward, simulation_end_time - simulation_start_time))

            accu_reward += reward
            state = next_state
            step_count += 1

        game_end_time = time.time()
        print("> game ended. total reward: {}, used time {} s".format(accu_reward, game_end_time - game_start_time))

        return accu_reward, np.array(rewards, dtype = np.float32), np.array(times, dtype = np.float32)

    def simulate_single_move(self, state):
        # Clear cache
        self.root_node = None
        self.global_saving_idx = 0
        self.checkpoint_data_manager.clear()

        gc.collect()

        # Construct root node
        self.checkpoint_data_manager.checkpoint_env("main", self.global_saving_idx)

        self.root_node = UCTnode(
            action_n = self.action_n,
            state = state,
            checkpoint_idx = self.global_saving_idx,
            parent = None,
            tree = self,
            is_head = True
        )

        self.global_saving_idx += 1

        for _ in range(self.max_steps):
            self.simulate_single_step()

        best_action = self.root_node.max_utility_action()

        self.checkpoint_data_manager.load_checkpoint_env("main", self.root_node.checkpoint_idx)

        return best_action

    def simulate_single_step(self):
        # Go into root node
        curr_node = self.root_node

        # Selection
        curr_depth = 1
        while True:
            if curr_node.no_child_available() or (not curr_node.all_child_visited() and
                    curr_node != self.root_node and np.random.random() < 0.5) or \
                    (not curr_node.all_child_visited() and curr_node == self.root_node):
                # If no child node has been updated, we have to perform expansion anyway.
                # Or if root node is not fully visited.
                # Or if non-root node is not fully visited and {with prob 1/2}.

                need_expansion = True
                break

            else:
                action = curr_node.select_action()

            curr_node.update_history(action, curr_node.rewards[action])

            if curr_node.dones[action] or curr_depth >= self.max_depth:
                need_expansion = False
                break

            next_node = curr_node.children[action]

            curr_depth += 1
            curr_node = next_node

        # Expansion
        if need_expansion:
            expand_action = curr_node.select_expand_action()

            self.checkpoint_data_manager.load_checkpoint_env("main", curr_node.checkpoint_idx)
            next_state, reward, done = self.wrapped_env.step(expand_action)
            self.checkpoint_data_manager.checkpoint_env("main", self.global_saving_idx)

            curr_node.rewards[expand_action] = reward
            curr_node.dones[expand_action] = done

            curr_node.update_history(
                action_taken = expand_action,
                reward = reward
            )

            curr_node.add_child(
                expand_action,
                next_state,
                self.global_saving_idx,
                prior_prob = self.get_prior_prob(next_state)
            )
            self.global_saving_idx += 1
        else:
            self.checkpoint_data_manager.load_checkpoint_env("main", curr_node.checkpoint_idx)
            next_state, reward, done = self.wrapped_env.step(action)

            curr_node.rewards[action] = reward
            curr_node.dones[action] = done

        # Simulation
        done = False
        accu_reward = 0.0
        accu_gamma = 1.0

        while not done:
            action = self.get_action(next_state)

            next_state, reward, done = self.wrapped_env.step(action)

            accu_reward += reward * accu_gamma

            accu_gamma *= self.gamma

        # Complete Update
        self.complete_update(curr_node, self.root_node, accu_reward)

    def get_action(self, state):
        return self.policy_wrapper.get_action(state)

    def get_prior_prob(self, state):
        return self.policy_wrapper.get_prior_prob(state)

    def close(self):
        pass

    @staticmethod
    def complete_update(curr_node, curr_node_head, accu_reward):
        while curr_node != curr_node_head:
            accu_reward = curr_node.update(accu_reward)
            curr_node = curr_node.parent

        curr_node_head.update(accu_reward)
