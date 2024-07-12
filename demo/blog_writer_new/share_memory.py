from sherpa_ai.events import Event, EventType
from sherpa_ai.memory import SharedMemory


class DictSharedMemory(SharedMemory):
    fixed_keys: list[str] = []

    def __init__(self, data: dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.fixed_keys = list(data.keys())

    def update(self, name: str, data: dict):
        self.data.update(data)
        self.add_event(Event(event_type=EventType.result, agent=name, content=data))

    def get_last(self) -> str:
        last_event = self.events[-1]
        content = "\n".join([f"{k}: {v}" for k, v in last_event.content.items()])
        
        return content