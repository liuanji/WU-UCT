import numpy as np
from copy import deepcopy
import math
import random

from Utils.MovingAvegCalculator import MovingAvegCalculator


class UCTnode():
    def __init__(self, action_n, state, checkpoint_idx, parent, tree,
                 prior_prob = None, is_head = False, allowed_actions = None):
        self.action_n = action_n
        self.state = state
        self.checkpoint_idx = checkpoint_idx
        self.parent = parent
        self.tree = tree
        self.is_head = is_head
        self.allowed_actions = allowed_actions

        if tree is not None:
            self.max_width = tree.max_width
        else:
            self.max_width = 0

        self.children = [None for _ in range(self.action_n)]
        self.rewards = [0.0 for _ in range(self.action_n)]
        self.dones = [False for _ in range(self.action_n)]
        self.children_visit_count = [0 for _ in range(self.action_n)]
        self.Q_values = [0 for _ in range(self.action_n)]
        self.visit_count = 0

        if prior_prob is not None:
            self.prior_prob = prior_prob
        else:
            self.prior_prob = np.ones([self.action_n], dtype = np.float32) / self.action_n

        # Record traverse history
        self.traverse_history = list()

        # Updated node count
        self.updated_node_count = 0

        # Moving average calculator
        self.moving_aveg_calculator = MovingAvegCalculator(window_length = 500)

    def no_child_available(self):
        # All child nodes have not been expanded.
        return self.updated_node_count == 0

    def all_child_visited(self):
        # All child nodes have been visited and updated.
        if self.is_head:
            if self.allowed_actions is None:
                return self.updated_node_count == self.action_n
            else:
                return self.updated_node_count == len(self.allowed_actions)
        else:
            return self.updated_node_count == self.max_width

    def select_action(self):
        best_score = -10000.0
        best_action = 0

        for action in range(self.action_n):
            if self.children[action] is None:
                continue

            if self.allowed_actions is not None and action not in self.allowed_actions:
                continue

            exploit_score = self.Q_values[action] / self.children_visit_count[action]
            explore_score = math.sqrt(1.0 * math.log(self.visit_count) / self.children_visit_count[action])
            score_std = self.moving_aveg_calculator.get_standard_deviation()
            score = exploit_score + score_std * explore_score

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def max_utility_action(self):
        best_score = -10000.0
        best_action = 0

        for action in range(self.action_n):
            if self.children[action] is None:
                continue

            score = self.Q_values[action] / self.children_visit_count[action]

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def select_expand_action(self):
        count = 0

        while True:
            if self.allowed_actions is None:
                if count < 20:
                    action = self.categorical(self.prior_prob)
                else:
                    action = np.random.randint(0, self.action_n)
            else:
                action = random.choice(self.allowed_actions)

            if count > 100:
                return action

            if self.children_visit_count[action] > 0 and count < 10:
                count += 1
                continue

            if self.children[action] is None:
                return action

            count += 1

    def update_history(self, action_taken, reward):
        self.traverse_history = (action_taken, reward)

    def update(self, accu_reward):
        action_taken = self.traverse_history[0]
        reward = self.traverse_history[1]

        accu_reward = reward + self.tree.gamma * accu_reward

        if self.children_visit_count[action_taken] == 0:
            self.updated_node_count += 1

        self.children_visit_count[action_taken] += 1
        self.Q_values[action_taken] += accu_reward

        self.visit_count += 1

        self.moving_aveg_calculator.add_number(accu_reward)

        return accu_reward

    def add_child(self, action, child_state, checkpoint_idx, prior_prob):
        if self.children[action] is not None:
            node = self.children[action]
        else:
            node = UCTnode(
                action_n = self.action_n,
                state = child_state,
                checkpoint_idx = checkpoint_idx,
                parent = self,
                tree = self.tree,
                prior_prob = prior_prob
            )

            self.children[action] = node

        return node

    @staticmethod
    def categorical(pvals):
        num = np.random.random()
        for i in range(pvals.size):
            if num < pvals[i]:
                return i
            else:
                num -= pvals[i]

        return pvals.size - 1
