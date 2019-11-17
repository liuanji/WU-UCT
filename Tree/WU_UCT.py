import numpy as np
from copy import deepcopy
import gc
import time
import logging

from Node.WU_UCTnode import WU_UCTnode

from Env.EnvWrapper import EnvWrapper

from ParallelPool.PoolManager import PoolManager

from Mem.CheckpointManager import CheckpointManager


class WU_UCT():
    def __init__(self, env_params, max_steps = 1000, max_depth = 20, max_width = 5,
                 gamma = 1.0, expansion_worker_num = 16, simulation_worker_num = 16,
                 policy = "Random", seed = 123, device = "cpu", record_video = False):
        self.env_params = env_params
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.max_width = max_width
        self.gamma = gamma
        self.expansion_worker_num = expansion_worker_num
        self.simulation_worker_num = simulation_worker_num
        self.policy = policy
        self.device = device
        self.record_video = record_video

        # Environment
        record_path = "Records/P-UCT_" + env_params["env_name"] + ".mp4"
        self.wrapped_env = EnvWrapper(**env_params, enable_record = record_video,
                                      record_path = record_path)

        # Environment properties
        self.action_n = self.wrapped_env.get_action_n()
        self.max_width = min(self.action_n, self.max_width)

        assert self.max_depth > 0 and 0 < self.max_width <= self.action_n

        # Expansion worker pool
        self.expansion_worker_pool = PoolManager(
            worker_num = expansion_worker_num,
            env_params = env_params,
            policy = policy,
            gamma = gamma,
            seed = seed,
            device = device,
            need_policy = False
        )

        # Simulation worker pool
        self.simulation_worker_pool = PoolManager(
            worker_num = simulation_worker_num,
            env_params = env_params,
            policy = policy,
            gamma = gamma,
            seed = seed,
            device = device,
            need_policy = True
        )

        # Checkpoint data manager
        self.checkpoint_data_manager = CheckpointManager()
        self.checkpoint_data_manager.hock_env("main", self.wrapped_env)

        # For MCTS tree
        self.root_node = None
        self.global_saving_idx = 0

        # Task recorder
        self.expansion_task_recorder = dict()
        self.unscheduled_expansion_tasks = list()
        self.simulation_task_recorder = dict()
        self.unscheduled_simulation_tasks = list()

        # Simulation count
        self.simulation_count = 0

        # Logging
        logging.basicConfig(filename = "Logs/P-UCT_" + self.env_params["env_name"] + "_" +
                                       str(self.simulation_worker_num) + ".log", level = logging.INFO)

    # Entrance of the P-UCT algorithm
    # This is the outer loop of P-UCT simulation, where the P-UCT agent consecutively plan a best action and
    # interact with the environment.
    def simulate_trajectory(self, max_episode_length = -1):
        state = self.wrapped_env.reset()
        accu_reward = 0.0
        done = False
        step_count = 0
        rewards = []
        times = []

        game_start_time = time.clock()

        logging.info("Start simulation")

        while not done and (max_episode_length == -1 or step_count < max_episode_length):
            # Plan a best action under the current state
            simulation_start_time = time.clock()
            action = self.simulate_single_move(state)
            simulation_end_time = time.clock()

            # Interact with the environment
            next_state, reward, done = self.wrapped_env.step(action)
            rewards.append(reward)
            times.append(simulation_end_time - simulation_start_time)

            print("> Time step {}, take action {}, instance reward {}, cumulative reward {}, used {} seconds".format(
                step_count, action, reward, accu_reward + reward, simulation_end_time - simulation_start_time))
            logging.info("> Time step {}, take action {}, instance reward {}, cumulative reward {}, used {} seconds".format(
                step_count, action, reward, accu_reward + reward, simulation_end_time - simulation_start_time))

            # Record video
            if self.record_video:
                self.wrapped_env.capture_frame()
                self.wrapped_env.store_video_files()

            # update game status
            accu_reward += reward
            state = next_state
            step_count += 1

        game_end_time = time.clock()
        print("> game ended. total reward: {}, used time {} s".format(accu_reward, game_end_time - game_start_time))
        logging.info("> game ended. total reward: {}, used time {} s".format(accu_reward,
            game_end_time - game_start_time))

        return accu_reward, np.array(rewards, dtype = np.float32), np.array(times, dtype = np.float32)

    # This is the planning process of P-UCT. Starts from a tree with a root node only,
    # P-UCT performs selection, expansion, simulation, and backpropagation on it.
    def simulate_single_move(self, state):
        # Clear cache
        self.root_node = None
        self.global_saving_idx = 0
        self.checkpoint_data_manager.clear()

        # Clear recorders
        self.expansion_task_recorder.clear()
        self.unscheduled_expansion_tasks.clear()
        self.simulation_task_recorder.clear()
        self.unscheduled_simulation_tasks.clear()

        gc.collect()

        # Free all workers
        self.expansion_worker_pool.wait_until_all_envs_idle()
        self.simulation_worker_pool.wait_until_all_envs_idle()

        # Construct root node
        self.checkpoint_data_manager.checkpoint_env("main", self.global_saving_idx)
        self.root_node = WU_UCTnode(
            action_n = self.action_n,
            state = state,
            checkpoint_idx = self.global_saving_idx,
            parent = None,
            tree = self,
            is_head = True
        )

        # An index used to retrieve game-states
        self.global_saving_idx += 1

        # t_complete in the origin paper, measures the completed number of simulations
        self.simulation_count = 0

        # Repeatedly invoke the master loop (Figure 2 of the paper)
        sim_idx = 0
        while self.simulation_count < self.max_steps:
            self.simulate_single_step(sim_idx)

            sim_idx += 1

        # Select the best root action
        best_action = self.root_node.max_utility_action()

        # Retrieve the game-state before simulation begins
        self.checkpoint_data_manager.load_checkpoint_env("main", self.root_node.checkpoint_idx)

        return best_action

    def simulate_single_step(self, sim_idx):
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

                cloned_curr_node = curr_node.shallow_clone()
                checkpoint_data = self.checkpoint_data_manager.retrieve(curr_node.checkpoint_idx)

                # Record the task
                self.expansion_task_recorder[sim_idx] = (checkpoint_data, cloned_curr_node, curr_node)
                self.unscheduled_expansion_tasks.append(sim_idx)

                need_expansion = True
                break

            else:
                action = curr_node.select_action()

            curr_node.update_history(sim_idx, action, curr_node.rewards[action])

            if curr_node.dones[action] or curr_depth >= self.max_depth:
                # Exceed maximum depth
                need_expansion = False
                break

            if curr_node.children[action] is None:
                need_expansion = False
                break

            next_node = curr_node.children[action]

            curr_depth += 1
            curr_node = next_node

        # Expansion
        if not need_expansion:
            if not curr_node.dones[action]:
                # Reach maximum depth but have not terminate.
                # Record simulation task.

                self.simulation_task_recorder[sim_idx] = (
                    action,
                    curr_node,
                    curr_node.checkpoint_idx,
                    None
                )
                self.unscheduled_simulation_tasks.append(sim_idx)
            else:
                # Reach terminal node.
                # In this case, update directly.

                self.incomplete_update(curr_node, self.root_node, sim_idx)
                self.complete_update(curr_node, self.root_node, 0.0, sim_idx)

                self.simulation_count += 1

        else:
            # Assign tasks to idle server
            while len(self.unscheduled_expansion_tasks) > 0 and self.expansion_worker_pool.has_idle_server():
                # Get a task
                curr_idx = np.random.randint(0, len(self.unscheduled_expansion_tasks))
                task_idx = self.unscheduled_expansion_tasks.pop(curr_idx)

                # Assign the task to server
                checkpoint_data, cloned_curr_node, _ = self.expansion_task_recorder[task_idx]
                self.expansion_worker_pool.assign_expansion_task(
                    checkpoint_data,
                    cloned_curr_node,
                    self.global_saving_idx,
                    task_idx
                )
                self.global_saving_idx += 1

            # Wait for an expansion task to complete
            if self.expansion_worker_pool.server_occupied_rate() >= 0.99:
                expand_action, next_state, reward, done, checkpoint_data, \
                saving_idx, task_idx = self.expansion_worker_pool.get_complete_expansion_task()

                curr_node = self.expansion_task_recorder.pop(task_idx)[2]
                curr_node.update_history(task_idx, expand_action, reward)

                # Record info
                curr_node.dones[expand_action] = done
                curr_node.rewards[expand_action] = reward

                if done:
                    # If this expansion result in a terminal node, perform update directly.
                    # (simulation is not needed)

                    self.incomplete_update(curr_node, self.root_node, task_idx)
                    self.complete_update(curr_node, self.root_node, 0.0, task_idx)

                    self.simulation_count += 1

                else:
                    # Schedule the task to the simulation task buffer.

                    self.checkpoint_data_manager.store(saving_idx, checkpoint_data)

                    self.simulation_task_recorder[task_idx] = (
                        expand_action,
                        curr_node,
                        saving_idx,
                        deepcopy(next_state)
                    )
                    self.unscheduled_simulation_tasks.append(task_idx)

        # Assign simulation tasks to idle environment server
        while len(self.unscheduled_simulation_tasks) > 0 and self.simulation_worker_pool.has_idle_server():
            # Get a task
            idx = np.random.randint(0, len(self.unscheduled_simulation_tasks))
            task_idx = self.unscheduled_simulation_tasks.pop(idx)

            checkpoint_data = self.checkpoint_data_manager.retrieve(self.simulation_task_recorder[task_idx][2])

            first_aciton = None if self.simulation_task_recorder[task_idx][3] \
                is not None else self.simulation_task_recorder[task_idx][0]

            # Assign the task to server
            self.simulation_worker_pool.assign_simulation_task(
                task_idx,
                checkpoint_data,
                first_action = first_aciton
            )

            # Perform incomplete update
            self.incomplete_update(
                self.simulation_task_recorder[task_idx][1], # This is the corresponding node
                self.root_node,
                task_idx
            )

        # Wait for a simulation task to complete
        if self.simulation_worker_pool.server_occupied_rate() >= 0.99:
            args = self.simulation_worker_pool.get_complete_simulation_task()
            if len(args) == 3:
                task_idx, accu_reward, prior_prob = args
            else:
                task_idx, accu_reward, reward, done = args
            expand_action, curr_node, saving_idx, next_state = self.simulation_task_recorder.pop(task_idx)

            if len(args) == 4:
                curr_node.rewards[expand_action] = reward
                curr_node.dones[expand_action] = done

            # Add node
            if next_state is not None:
                curr_node.add_child(
                    expand_action,
                    next_state,
                    saving_idx,
                    prior_prob = prior_prob
                )

            # Complete Update
            self.complete_update(curr_node, self.root_node, accu_reward, task_idx)

            self.simulation_count += 1

    def close(self):
        # Free sub-processes
        self.expansion_worker_pool.close_pool()
        self.simulation_worker_pool.close_pool()

    # Incomplete update allows to track unobserved samples (Algorithm 2 in the paper)
    @staticmethod
    def incomplete_update(curr_node, curr_node_head, idx):
        while curr_node != curr_node_head:
            curr_node.update_incomplete(idx)
            curr_node = curr_node.parent

        curr_node_head.update_incomplete(idx)

    # Complete update tracks the observed samples (Algorithm 3 in the paper)
    @staticmethod
    def complete_update(curr_node, curr_node_head, accu_reward, idx):
        while curr_node != curr_node_head:
            accu_reward = curr_node.update_complete(idx, accu_reward)
            curr_node = curr_node.parent

        curr_node_head.update_complete(idx, accu_reward)
