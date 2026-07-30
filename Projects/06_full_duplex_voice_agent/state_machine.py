from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CANCELLING = "cancelling"

@dataclass
class DuplexSession:
    state: State = State.IDLE
    generation_id: str | None = None

    def on_user_audio(self, speech: bool):
        if speech and self.state == State.SPEAKING:
            self.state = State.CANCELLING
            return "barge_in"
        if speech and self.state in {State.IDLE, State.LISTENING}:
            self.state = State.LISTENING
            return "listening"
        return "ignored"

    def on_endpoint(self):
        if self.state != State.LISTENING:
            raise ValueError("endpoint outside listening state")
        self.state = State.THINKING

    def on_generation_started(self, generation_id: str):
        if self.state != State.THINKING:
            raise ValueError("generation can only start after thinking")
        self.generation_id = generation_id
        self.state = State.SPEAKING

    def on_cancelled(self, generation_id: str):
        if self.state != State.CANCELLING or generation_id != self.generation_id:
            raise ValueError("stale cancellation")
        self.generation_id = None
        self.state = State.LISTENING

    def on_finished(self, generation_id: str):
        if generation_id != self.generation_id or self.state == State.CANCELLING:
            return False
        self.generation_id = None
        self.state = State.IDLE
        return True
