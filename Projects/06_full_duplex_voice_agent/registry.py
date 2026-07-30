import asyncio

class GenerationRegistry:
    def __init__(self):
        self.tasks = {}
        self.audio_queues = {}

    def register(self, generation_id, coroutine):
        self.audio_queues[generation_id] = asyncio.Queue()
        task = asyncio.create_task(coroutine)
        self.tasks[generation_id] = task
        return task

    async def cancel(self, generation_id):
        task = self.tasks.pop(generation_id, None)
        self.audio_queues.pop(generation_id, None)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True
