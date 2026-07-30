import unittest
from asr_core import ManifestRow, assert_speaker_disjoint, error_rate, normalize_text


class CoreTest(unittest.TestCase):
    def test_normalize_and_cer(self):
        self.assertEqual(normalize_text(" 你好，WORLD! "), "你好world")
        self.assertAlmostEqual(error_rate("你好", "你号", "char"), 0.5)

    def test_speaker_leakage(self):
        a = ManifestRow("a.wav", "a", "zh", "s1", 1.0, "train")
        b = ManifestRow("b.wav", "b", "zh", "s1", 1.0, "test")
        with self.assertRaises(ValueError):
            assert_speaker_disjoint([a, b])


if __name__ == "__main__":
    unittest.main()
