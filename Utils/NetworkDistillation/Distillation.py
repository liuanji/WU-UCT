import torch
import random
import numpy as np
import sys
sys.path.append("../Env")

from Env.EnvWrapper import EnvWrapper

from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN

from .ReplayBuffer import ReplayBuffer


class Distillation():
    def __init__(self, wrapped_env, teacher_network, student_network,
                 temperature = 2.5, buffer_size = 1e5, batch_size = 32,
                 device = torch.device("cpu")):
        self.wrapped_env = wrapped_env
        self.teacher_network = teacher_network
        self.student_network = student_network
        self.temperature = temperature
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size = buffer_size, device = self.device)

    def train_step(self):
        state_batch, policy_batch, value_batch = self.replay_buffer.sample(self.batch_size)

        loss = self.student_network.train_step(state_batch, policy_batch, value_batch, temperature = self.temperature)

        return loss

    def gather_samples(self, max_step_count = 10000):
        state = self.wrapped_env.reset()

        step_count = 0

        done = False
        while not done:
            if np.random.random() < 0.9:
                action = self.categorical(self.student_network.get_action(state))
            else:
                action = np.random.randint(0, self.wrapped_env.action_n)
            target_policy = self.teacher_network.get_action(state, logit = True)
            target_value = self.teacher_network.get_value(state)

            self.replay_buffer.add((np.array(state), target_policy, target_value))

            state, _, done = self.wrapped_env.step(action)

            step_count += 1

            if step_count > max_step_count:
                return

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


def train_distillation(env_name, device):
    device = torch.device("cuda:0" if device == "cuda" else device)

    wrapped_env = EnvWrapper(env_name = env_name, max_episode_length = 100000)

    teacher_network = PPOAtariCNN(wrapped_env.action_n, device,
                                  checkpoint_dir = "./Policy/PPO/PolicyFiles/PPO_" + env_name + ".pt")
    student_network = PPOSmallAtariCNN(wrapped_env.action_n, device,
                                       checkpoint_dir = "")

    distillation = Distillation(wrapped_env, teacher_network, student_network, device = device)

    for _ in range(1000):
        for _ in range(10):
            distillation.gather_samples()

        for _ in range(1000):
            loss = distillation.train_step()
            print(loss)

        student_network.save("./Policy/PPO/PolicyFiles/SmallPPO_" + env_name + ".pt")
