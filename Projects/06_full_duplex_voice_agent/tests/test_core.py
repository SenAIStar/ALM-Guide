import unittest
from events import paired_latency
from state_machine import DuplexSession, State

class CoreTest(unittest.TestCase):
    def test_barge_in(self):
        session = DuplexSession(State.THINKING)
        session.on_generation_started("g1")
        self.assertEqual(session.on_user_audio(True), "barge_in")
        self.assertFalse(session.on_finished("g1"))
        self.assertEqual(session.state, State.CANCELLING)
        session.on_cancelled("g1")
        self.assertEqual(session.state, State.LISTENING)

    def test_latency(self):
        rows = [{"session_id": "s", "generation_id": "g", "event": "barge_in_detected", "timestamp_ms": 10}, {"session_id": "s", "generation_id": "g", "event": "generation_cancelled", "timestamp_ms": 42}]
        self.assertEqual(paired_latency(rows, "barge_in_detected", "generation_cancelled"), [32])

if __name__ == "__main__":
    unittest.main()
