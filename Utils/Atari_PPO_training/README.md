# PPO
PyTorch implementation of Proximal Policy Optimization

![live agents](assets/agents.gif)

## Usage

Example command line usage:
````
python main.py BreakoutNoFrameskip-v0 --num-workers 8 --render
````

This will run PPO with 8 parallel training environments, which will be rendered on the screen. Run with `-h` for usage information.

## Performance

Results are comparable to those of the original PPO paper. The horizontal axis here is labeled by environment steps, whereas the graphs in the paper label it with frames, with 4 frames per step.

Training episode reward versus environment steps for `BreakoutNoFrameskip-v3`:

![Breakout training curve](assets/breakout_reward.png)

## References

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

[OpenAI Baselines](https://github.com/openai/baselines)

This code uses some environment utilities such as `SubprocVecEnv` and `VecFrameStack` from OpenAI's Baselines.
