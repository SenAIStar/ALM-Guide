import unittest
from s2st_core import latency_budget, make_chunks, validate_pair

class CoreTest(unittest.TestCase):
    def test_chunking(self):
        chunks = make_chunks(45.0, 20.0, 2.0)
        self.assertEqual([(x.start, x.end) for x in chunks], [(0.0, 20.0), (18.0, 38.0), (36.0, 45.0)])

    def test_protocol(self):
        self.assertEqual(latency_budget(10, 20, 30)["total_ms"], 60)
        with self.assertRaises(ValueError):
            validate_pair({"audio_path": "a", "source_language": "eng", "target_language": "eng", "source_text": "a", "target_text": "a"})

if __name__ == "__main__":
    unittest.main()
