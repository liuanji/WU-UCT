from multiprocessing import Process
from copy import deepcopy
import random
import numpy as np

from Env.EnvWrapper import EnvWrapper

from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN

from Policy.PolicyWrapper import PolicyWrapper


# Slave workers
class Worker(Process):
    def __init__(self, pipe, env_params, policy = "Random", gamma = 1.0, seed = 123,
                 device = "cpu", need_policy = True):
        super(Worker, self).__init__()

        self.pipe = pipe
        self.env_params = deepcopy(env_params)
        self.gamma = gamma
        self.seed = seed
        self.policy = deepcopy(policy)
        self.device = deepcopy(device)
        self.need_policy = need_policy

        self.wrapped_env = None
        self.action_n = None
        self.max_episode_length = None

        self.policy_wrapper = None

    # Initialize the environment
    def init_process(self):
        self.wrapped_env = EnvWrapper(**self.env_params)

        self.wrapped_env.seed(self.seed)

        self.action_n = self.wrapped_env.get_action_n()
        self.max_episode_length = self.wrapped_env.get_max_episode_length()

    # Initialize the default policy
    def init_policy(self):
        self.policy_wrapper = PolicyWrapper(
            self.policy,
            self.env_params["env_name"],
            self.action_n,
            self.device
        )

    def run(self):
        self.init_process()
        self.init_policy()

        print("> Worker ready.")

        while True:
            # Wait for tasks
            command, args = self.receive_safe_protocol()

            if command == "KillProc":
                return
            elif command == "Expansion":
                checkpoint_data, curr_node, saving_idx, task_idx = args

                # Select expand action, and do expansion
                expand_action, next_state, reward, done, \
                    checkpoint_data = self.expand_node(checkpoint_data, curr_node)

                item = (expand_action, next_state, reward, done, checkpoint_data,
                        saving_idx, task_idx)

                self.send_safe_protocol("ReturnExpansion", item)
            elif command == "Simulation":
                if args is None:
                    raise RuntimeError
                else:
                    task_idx, checkpoint_data, first_action = args

                    state = self.wrapped_env.restore(checkpoint_data)

                    # Prior probability is calculated for the new node
                    prior_prob = self.get_prior_prob(state)

                    # When simulation invoked because of reaching maximum search depth,
                    # an action was actually selected. Therefore, we need to execute it
                    # first anyway.
                    if first_action is not None:
                        state, reward, done = self.wrapped_env.step(first_action)

                if first_action is not None and done:
                    accu_reward = reward
                else:
                    # Simulate until termination condition satisfied
                    accu_reward = self.simulate(state)

                if first_action is not None:
                    self.send_safe_protocol("ReturnSimulation", (task_idx, accu_reward, reward, done))
                else:
                    self.send_safe_protocol("ReturnSimulation", (task_idx, accu_reward, prior_prob))

    def expand_node(self, checkpoint_data, curr_node):
        self.wrapped_env.restore(checkpoint_data)

        # Choose action to expand, according to the shallow copy node
        expand_action = curr_node.select_expand_action()

        # Execute the action, and observe new state, etc.
        next_state, reward, done = self.wrapped_env.step(expand_action)

        if not done:
            checkpoint_data = self.wrapped_env.checkpoint()
        else:
            checkpoint_data = None

        return expand_action, next_state, reward, done, checkpoint_data

    def simulate(self, state, max_simulation_step = 100, lamda = 0.5):
        step_count = 0
        accu_reward = 0.0
        accu_gamma = 1.0

        start_state_value = self.get_value(state)

        done = False
        # A strict upper bound for simulation count
        while not done and step_count < max_simulation_step:
            action = self.get_action(state)

            next_state, reward, done = self.wrapped_env.step(action)

            accu_reward += reward * accu_gamma
            accu_gamma *= self.gamma

            state = deepcopy(next_state)

            step_count += 1

        if not done:
            accu_reward += self.get_value(state) * accu_gamma

        # Use V(s) to stabilize simulation return
        accu_reward = accu_reward * lamda + start_state_value * (1.0 - lamda)

        return accu_reward

    def get_action(self, state):
        return self.policy_wrapper.get_action(state)

    def get_value(self, state):
        return self.policy_wrapper.get_value(state)

    def get_prior_prob(self, state):
        return self.policy_wrapper.get_prior_prob(state)

    # Send message through pipe
    def send_safe_protocol(self, command, args):
        success = False

        count = 0
        while not success:
            self.pipe.send((command, args))

            ret = self.pipe.recv()
            if ret == command or count >= 10:
                success = True
                
            count += 1

    # Receive message from pipe
    def receive_safe_protocol(self):
        self.pipe.poll(None)

        command, args = self.pipe.recv()

        self.pipe.send(command)

        return deepcopy(command), deepcopy(args)
