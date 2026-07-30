import unittest
from audio_llm_core import downsample_mask, task_macro_average, validate_instruction_sample

class CoreTest(unittest.TestCase):
    def test_mask(self):
        self.assertEqual(downsample_mask([1, 1, 0, 0], 2), [1, 0])

    def test_task_macro(self):
        report = task_macro_average([{"task": "asr", "score": 1}, {"task": "asr", "score": 0}, {"task": "audio_qa", "score": 1}])
        self.assertEqual(report["macro"], 0.75)
        with self.assertRaises(ValueError):
            validate_instruction_sample({"task": "unknown"})

if __name__ == "__main__":
    unittest.main()
