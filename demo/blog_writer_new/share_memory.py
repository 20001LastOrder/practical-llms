from sherpa_ai.memory import SharedMemory


class DictSharedMemory(SharedMemory):
    def __init__(self, data: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.data = data
