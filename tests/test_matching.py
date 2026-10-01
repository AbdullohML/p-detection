"""Check duplicate protection, empty sets, and the exact IoU threshold."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_failures import match


class MatchingTests(unittest.TestCase):
    def test_duplicate_is_false_positive(self):
        result = match([[0,0,10,10,.8],[0,0,10,10,.9]], [[0,0,10,10]])
        self.assertEqual(result['fp'], [0])
        self.assertEqual(result['tp'][0]['prediction'], 1)
        self.assertEqual(result['fn'], [])

    def test_empty_predictions(self):
        self.assertEqual(match([], [[0,0,10,10]])['fn'], [0])

    def test_empty_ground_truth(self):
        self.assertEqual(match([[0,0,10,10,.9]], [])['fp'], [0])

    def test_disjoint_boxes(self):
        result = match([[20,20,30,30,.9]], [[0,0,10,10]])
        self.assertEqual((result['fp'], result['fn']), ([0], [0]))

    def test_boundary_is_inclusive(self):
        result = match([[0,0,5,10,.9]], [[0,0,10,10]])
        self.assertEqual(result['tp'][0]['iou'], .5)

    def test_prediction_cannot_match_two_ground_truths(self):
        result = match([[0,0,10,10,.9]], [[0,0,10,10], [0,0,10,10]])
        self.assertEqual(len(result['tp']), 1)
        self.assertEqual(len(result['fn']), 1)


if __name__ == '__main__':
    unittest.main()
