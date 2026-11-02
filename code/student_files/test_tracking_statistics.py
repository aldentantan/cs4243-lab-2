"""Tracking-report regressions for failed registrations and lost intervals. [AI-CODE]"""
import unittest

import numpy as np

from overlaylab.report import summarise_tracking


class TestTrackingStatistics(unittest.TestCase):
    def test_failed_localisations_count_as_calls_and_break_valid_runs(self):
        quad = [[10., 10.], [30., 10.], [30., 30.], [10., 30.]]
        status = ['ok', 'ok', 'lost', 'lost', 'lost', 'ok', 'ok']
        saved = {
            'resize_width': None,
            'corners': [quad if s == 'ok' else np.full((4, 2), np.nan).tolist() for s in status],
            'status': status,
            'flags': [['OK', 'none'] if s == 'ok' else ['HIDE', 'no_model'] for s in status],
            'reinit_frames': [0, 5],
            'stats': [{'n_matches': 20}, {'n_matches': None}, {'n_matches': 0},
                      {'n_matches': 2}, {'n_matches': 0}, {'n_matches': 20}, {'n_matches': None}],
        }
        result = summarise_tracking(saved, {0: np.array(quad), 6: np.array(quad)}, .01, 100)
        with self.subTest(metric='localiser calls'):
            self.assertEqual(result['localisations'], 5)
        with self.subTest(metric='longest valid run'):
            self.assertEqual(result['longest_run'], 2)

    def test_longest_valid_run_includes_the_initial_registration(self):
        quad = [[10., 10.], [30., 10.], [30., 30.], [10., 30.]]
        saved = {'resize_width': None, 'corners': [quad] * 7, 'status': ['ok'] * 7,
                 'flags': [['OK', 'none']] * 7, 'reinit_frames': [0, 3],
                 'stats': [{'n_matches': 20 if i in (0, 3) else None} for i in range(7)]}
        result = summarise_tracking(saved, {0: np.array(quad), 6: np.array(quad)}, .01, 100)
        self.assertEqual(result['localisations'], 2)
        self.assertEqual(result['longest_run'], 4)


if __name__ == '__main__':
    unittest.main()
