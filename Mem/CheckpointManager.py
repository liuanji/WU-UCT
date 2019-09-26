# This is the centralized game-state storage.
class CheckpointManager():
    def __init__(self):
        self.buffer = dict()

        self.envs = dict()

    def hock_env(self, name, env):
        self.envs[name] = env

    def checkpoint_env(self, name, idx):
        self.store(idx, self.envs[name].checkpoint())

    def load_checkpoint_env(self, name, idx):
        self.envs[name].restore(self.retrieve(idx))

    def store(self, idx, checkpoint_data):
        assert idx not in self.buffer

        self.buffer[idx] = checkpoint_data

    def retrieve(self, idx):
        assert idx in self.buffer

        return self.buffer[idx]

    def length(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
