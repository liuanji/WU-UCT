import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from Env.AtariEnv.atari_wrappers import LazyFrames


def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models

    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()


class PPOAtariCNN():
    def __init__(self, num_actions, device = "cpu", checkpoint_dir = ""):
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        self.model = AtariCNN(num_actions, self.device)

        if checkpoint_dir != "" and os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
            self.model.load_state_dict(checkpoint["policy"])

        self.model.to(device)

    def get_action(self, state, logit = False):
        return self.model.get_action(state, logit = logit)

    def get_value(self, state):
        return self.model.get_value(state)


class PPOSmallAtariCNN():
    def __init__(self, num_actions, device = "cpu", checkpoint_dir = ""):
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        self.model = SmallPolicyAtariCNN(num_actions, self.device)

        if checkpoint_dir != "" and os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir, map_location = "cpu")
            # self.model.load_state_dict(checkpoint["policy"])

        self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3)

        self.mseLoss = nn.MSELoss()

    def get_action(self, state):
        return self.model.get_action(state)

    def get_value(self, state):
        # raise RuntimeError("Small policy net does not support value evaluation.")
        return self.model.get_value(state)

    def train_step(self, state_batch, policy_batch, value_batch, temperature = 2.5):
        self.optimizer.zero_grad()

        # policy_batch = F.softmax(policy_batch / temperature, dim = 1)

        out_policy, out_value = self.model(state_batch)
        # out_policy = F.softmax(out_policy / temperature, dim = 1)

        # loss = -(policy_batch * torch.log(out_policy + 1e-8)).sum(dim = 1).mean()
        loss = self.mseLoss(policy_batch, out_policy) + self.mseLoss(value_batch, out_value)
        loss.backward()

        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def save(self, path):
        torch.save({"policy": self.model.state_dict()}, path)


class AtariCNN(nn.Module):
    def __init__(self, num_actions, device):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU(inplace=True))

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                                nn.ReLU(inplace=True))

        self.pi = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

        self.num_actions = num_actions

        self.device = device

    def forward(self, conv_in):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return pi_out, v_out

    def get_action(self, conv_in, logit = False):
        if isinstance(conv_in, LazyFrames):
            conv_in = torch.from_numpy(np.array(conv_in)).type(torch.float32).to(self.device).unsqueeze(0)
        elif isinstance(conv_in, np.ndarray):
            conv_in = torch.from_numpy(conv_in).to(self.device)

        N = conv_in.size(0)
        s = time.time()
        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)
        aa = time.time()
        fc_out = self.fc(conv_out)

        if logit:
            pi_out = self.pi(fc_out)
        else:
            pi_out = F.softmax(self.pi(fc_out), dim = 1)

        if N == 1:
            pi_out = pi_out.view(-1)
        e = time.time()
        # print("large", e - s, aa - s)
        return pi_out.detach().cpu().numpy()

    def get_value(self, conv_in):
        if isinstance(conv_in, LazyFrames):
            conv_in = torch.from_numpy(np.array(conv_in)).type(torch.float32).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()

        N = conv_in.size(0)

        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        v_out = self.v(fc_out)

        if N == 1:
            v_out = v_out.sum()

        return v_out.detach().cpu().numpy()


class SmallPolicyAtariCNN(nn.Module):
    def __init__(self, num_actions, device):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 16, 3, stride = 4),
                                  nn.ReLU(inplace = True),
                                  nn.Conv2d(16, 16, 3, stride = 4),
                                  nn.ReLU(inplace = True))

        self.fc = nn.Sequential(nn.Linear(16 * 5 * 5, 64),
                                nn.ReLU(inplace = True))

        self.pi = nn.Linear(64, num_actions)
        self.v = nn.Linear(64, 1)

        self.num_actions = num_actions

        self.device = device

    def forward(self, conv_in):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).view(N, 16 * 5 * 5)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return pi_out, v_out

    def get_action(self, conv_in, logit = False, get_tensor = False):
        if isinstance(conv_in, LazyFrames):
            conv_in = torch.from_numpy(np.array(conv_in)).type(torch.float32).to(self.device).unsqueeze(0)
        elif isinstance(conv_in, np.ndarray):
            conv_in = torch.from_numpy(conv_in).to(self.device)

        N = conv_in.size(0)
        s = time.time()
        # with torch.no_grad(): 
        conv_out = self.conv(conv_in).view(N, 16 * 5 * 5)
        aa = time.time()
        fc_out = self.fc(conv_out)
        bb = time.time()
        if logit:
            pi_out = self.pi(fc_out)
        else:
            pi_out = F.softmax(self.pi(fc_out), dim = 1)
        cc = time.time()
        if N == 1:
            pi_out = pi_out.view(-1)
        e = time.time()
        print("small", e - s, aa - s, bb - aa, cc - bb)
        if get_tensor:
            return pi_out
        else:
            return pi_out.detach().cpu().numpy()

    def get_value(self, conv_in):
        if isinstance(conv_in, LazyFrames):
            conv_in = torch.from_numpy(np.array(conv_in)).type(torch.float32).to(self.device).unsqueeze(0)
        else:
            raise NotImplementedError()

        N = conv_in.size(0)

        conv_out = self.conv(conv_in).view(N, 16 * 5 * 5)

        fc_out = self.fc(conv_out)

        v_out = self.v(fc_out)
        # print("a")
        if N == 1:
            v_out = v_out.sum()

        return v_out.detach().cpu().numpy()
