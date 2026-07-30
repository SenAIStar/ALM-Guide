import unittest
from grounding_core import Interval, best_span, grounded_correct, temporal_iou, temporal_nms, window_starts

class CoreTest(unittest.TestCase):
    def test_iou_and_nms(self):
        a = Interval(0, 10, 0.9, "s1")
        b = Interval(5, 15, 0.8, "s1")
        self.assertAlmostEqual(temporal_iou(a, b), 1 / 3)
        self.assertEqual(len(temporal_nms([a, b], 0.3)), 1)
        self.assertEqual(len(temporal_nms([a, Interval(5, 15, 0.8, "s2")], 0.3)), 2)
        self.assertTrue(grounded_correct(a, a, True))

    def test_windows(self):
        self.assertEqual(window_starts(65, 30, 5), [0.0, 25.0, 50.0])

    def test_constrained_span(self):
        self.assertEqual(best_span([0.0, 10.0, 0.0], [9.0, 0.0, 8.0]), (1, 2))

if __name__ == "__main__":
    unittest.main()
