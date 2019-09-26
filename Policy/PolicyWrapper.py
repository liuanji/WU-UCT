import numpy as np
import random
import os

from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN


class PolicyWrapper():
    def __init__(self, policy_name, env_name, action_n, device):
        self.policy_name = policy_name
        self.env_name = env_name
        self.action_n = action_n
        self.device = device

        self.policy_func = None

        self.init_policy()

    def init_policy(self):
        if self.policy_name == "Random":
            self.policy_func = None

        elif self.policy_name == "PPO":
            assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"), "Policy file not found"

            self.policy_func = PPOAtariCNN(
                self.action_n,
                device = self.device,
                checkpoint_dir = "./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"
            )

        elif self.policy_name == "DistillPPO":
            assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"), "Policy file not found"
            assert os.path.exists("./Policy/PPO/PolicyFiles/SmallPPO_" + self.env_name + ".pt"), "Policy file not found"

            full_policy = PPOAtariCNN(
                self.action_n,
                device = "cpu", # To save memory
                checkpoint_dir = "./Policy/PPO/PolicyFiles/PPO_" + self.env_name + ".pt"
            )

            small_policy = PPOSmallAtariCNN(
                self.action_n,
                device = self.device,
                checkpoint_dir = "./Policy/PPO/PolicyFiles/SmallPPO_" + self.env_name + ".pt"
            )

            self.policy_func = [full_policy, small_policy]
        else:
            raise NotImplementedError()

    def get_action(self, state):
        if self.policy_name == "Random":
            return random.randint(0, self.action_n - 1)
        elif self.policy_name == "PPO":
            return self.categorical(self.policy_func.get_action(state))
        elif self.policy_name == "DistillPPO":
            return self.categorical(self.policy_func[1].get_action(state))
        else:
            raise NotImplementedError()

    def get_value(self, state):
        if self.policy_name == "Random":
            return 0.0
        elif self.policy_name == "PPO":
            return self.policy_func.get_value(state)
        elif self.policy_name == "DistillPPO":
            return self.policy_func[0].get_value(state)
        else:
            raise NotImplementedError()

    def get_prior_prob(self, state):
        if self.policy_name == "Random":
            return np.ones([self.action_n], dtype = np.float32) / self.action_n
        elif self.policy_name == "PPO":
            return self.policy_func.get_action(state)
        elif self.policy_name == "DistillPPO":
            return self.policy_func[0].get_action(state)
        else:
            raise NotImplementedError()

    @staticmethod
    def categorical(probs):
        val = random.random()
        chosen_idx = 0

        for prob in probs:
            val -= prob

            if val < 0.0:
                break

            chosen_idx += 1

        if chosen_idx >= len(probs):
            chosen_idx = len(probs) - 1

        return chosen_idx
