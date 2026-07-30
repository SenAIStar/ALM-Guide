import unittest
from tts_core import cosine_similarity, normalize_embedding, real_time_factor, validate_sample

class CoreTest(unittest.TestCase):
    def test_embedding(self):
        values = normalize_embedding([3.0, 4.0], expected_dim=2)
        self.assertAlmostEqual(sum(x * x for x in values), 1.0)
        self.assertAlmostEqual(cosine_similarity(values, values), 1.0)

    def test_metrics_and_schema(self):
        self.assertEqual(real_time_factor(2.0, 1.0), 0.5)
        with self.assertRaises(ValueError):
            validate_sample({"text": "missing fields"})

if __name__ == "__main__":
    unittest.main()
