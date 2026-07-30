import unittest
from omni_core import conflict_preference, counterfactual_report, validate_sample

class CoreTest(unittest.TestCase):
    def test_protocol(self):
        sample = {"sample_id": "1", "modalities": ["audio", "image"], "instruction": "q", "answer": "a", "conflict_type": "audio_visual"}
        self.assertEqual(validate_sample(sample), sample)
        with self.assertRaises(ValueError):
            validate_sample({**sample, "modalities": ["audio"]})

    def test_counterfactual(self):
        report = counterfactual_report(1.0, {"audio": 0.2, "image": 0.9}, {"audio"})
        self.assertAlmostEqual(report["drop_without_audio"], 0.8)
        self.assertEqual(conflict_preference(["audio", "vision", "x"])["other"], 1)

if __name__ == "__main__":
    unittest.main()
