import gym
from copy import deepcopy

from Env.AtariEnv.AtariEnvWrapper import make_atari_env


# To allow easily extending to other tasks, we built a wrapper on top of the 'real' environment.
class EnvWrapper():
    def __init__(self, env_name, max_episode_length = 0, enable_record = False, record_path = "1.mp4"):
        self.env_name = env_name

        self.env_type = None

        try:
            self.env, self.recorder = make_atari_env(env_name, 0, 0, enable_record = enable_record,
                                                     record_path = record_path)

            # Call reset to avoid gym bugs.
            self.env.reset()

            self.env_type = "Atari"
        except gym.error.Error:
            exit(1)

        assert isinstance(self.env.action_space, gym.spaces.Discrete), "Should be discrete action space."
        self.action_n = self.env.action_space.n

        self.max_episode_length = self.env._max_episode_steps if max_episode_length == 0 else max_episode_length

        self.current_step_count = 0

        self.since_last_reset = 0

    def reset(self):
        state = self.env.reset()

        self.current_step_count = 0
        self.since_last_reset = 0

        return state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)

        self.current_step_count += 1
        if self.current_step_count >= self.max_episode_length:
            done = True

        self.since_last_reset += 1

        return next_state, reward, done

    def checkpoint(self):
        return deepcopy(self.env.clone_full_state()), self.current_step_count

    def restore(self, checkpoint):
        if self.since_last_reset > 20000:
            self.reset()
            self.since_last_reset = 0

        self.env.restore_full_state(checkpoint[0])

        self.current_step_count = checkpoint[1]

        return self.env.get_state()

    def render(self):
        self.env.render()

    def capture_frame(self):
        self.recorder.capture_frame()

    def store_video_files(self):
        self.recorder.write_metadata()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self.env.seed(seed)

    def get_action_n(self):
        return self.action_n

    def get_max_episode_length(self):
        return self.max_episode_length
