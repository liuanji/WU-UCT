import argparse
import torch
import torch.optim as optim
from atari_wrappers import FrameStack
from subproc_vec_env import SubprocVecEnv
from vec_frame_stack import VecFrameStack
import multiprocessing

from envs import make_env, RenderSubprocVecEnv
from models import AtariCNN
from ppo import PPO
from utils import set_seed, cuda_if

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    
    parser = argparse.ArgumentParser(description='PPO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('env_id', type=str, help='Gym environment id')
    parser.add_argument('--arch', type=str, default='cnn', help='policy architecture, {lstm, cnn}')
    parser.add_argument('--num-workers', type=int, default=8, help='number of parallel actors')
    parser.add_argument('--opt-epochs', type=int, default=3, help='optimization epochs between environment interaction')
    parser.add_argument('--total-steps', type=int, default=int(10e6), help='total number of environment steps to take')
    parser.add_argument('--worker-steps', type=int, default=128, help='steps per worker between optimization rounds')
    parser.add_argument('--sequence-steps', type=int, default=32, help='steps per sequence (for backprop through time)')
    parser.add_argument('--minibatch-steps', type=int, default=256, help='steps per optimization minibatch')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='initial learning rate')
    parser.add_argument('--lr-func', type=str, default='linear', help='learning rate schedule function, {linear, constant}')
    parser.add_argument('--clip', type=float, default=.1, help='initial probability ratio clipping range')
    parser.add_argument('--clip-func', type=str, default='linear', help='clip range schedule function, {linear, constant}')
    parser.add_argument('--gamma', type=float, default=.99, help='discount factor')
    parser.add_argument('--lambd', type=float, default=.95, help='GAE lambda parameter')
    parser.add_argument('--value-coef', type=float, default=1., help='value loss coeffecient')
    parser.add_argument('--entropy-coef', type=float, default=.01, help='entropy loss coeffecient')
    parser.add_argument('--max-grad-norm', type=float, default=.5, help='grad norm to clip at')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA acceleration')
    parser.add_argument('--render', action='store_true', help='render training environments')
    parser.add_argument('--render-interval', type=int, default=4, help='steps between environment renders')
    parser.add_argument('--plot-reward', action='store_true', help='plot episode reward')
    parser.add_argument('--plot-points', type=int, default=20, help='number of plot points (groups with mean, std)')
    parser.add_argument('--plot-path', type=str, default='ep_reward.png', help='path to save reward plot to')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    cuda = torch.cuda.is_available() and not args.no_cuda

    env_fns = []
    for rank in range(args.num_workers):
        env_fns.append(lambda: make_env(args.env_id, rank, args.seed + rank))
    if args.render:
        venv = RenderSubprocVecEnv(env_fns, args.render_interval)
    else:
        venv = SubprocVecEnv(env_fns)
    venv = VecFrameStack(venv, 4)

    test_env = make_env(args.env_id, 0, args.seed)
    test_env = FrameStack(test_env, 4)

    policy = {'cnn': AtariCNN}[args.arch](venv.action_space.n)
    policy = cuda_if(policy, cuda)

    optimizer = optim.Adam(policy.parameters())

    if args.lr_func == 'linear':
        lr_func = lambda a: args.lr * (1. - a)
    elif args.lr_func == 'constant':
        lr_func = lambda a: args.lr

    if args.clip_func == 'linear':
        clip_func = lambda a: args.clip * (1. - a)
    elif args.clip_func == 'constant':
        clip_func = lambda a: args.clip

    algorithm = PPO(policy, venv, test_env, optimizer,
                    lr_func=lr_func, clip_func=clip_func, gamma=args.gamma, lambd=args.lambd,
                    worker_steps=args.worker_steps, sequence_steps=args.sequence_steps,
                    minibatch_steps=args.minibatch_steps,
                    value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                    max_grad_norm=args.max_grad_norm,
                    cuda=cuda,
                    plot_reward=args.plot_reward, plot_points=args.plot_points, plot_path=args.plot_path, env_name = args.env_id)
    algorithm.run(args.total_steps)
