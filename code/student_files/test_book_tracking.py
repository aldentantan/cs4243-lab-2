"""Behaviour checks for the separate VACE book-mask preprocessor."""
import unittest
import json
import shutil
import tempfile
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from student_files.book_tracking import BookTracker, TrackerConfig, editable_mask


class TestHelperEntrypoint(unittest.TestCase):
    def test_help_works_outside_the_code_directory(self):
        helper = Path(__file__).resolve().parents[1] / 'student_files/prepare_vace_inputs.py'
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(helper), '--help'], cwd=temporary,
                                    env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--repair-missing-board', result.stdout)


def scene(shift=0, book=True):
    frame = np.full((180, 320, 3), 210, np.uint8)
    board = np.zeros(frame.shape[:2], np.uint8)
    board[30:160, 120:290] = 1
    frame[board.astype(bool)] = (35, 65, 140)
    # Blue artwork must not become a foreground object.
    frame[45:80, 145:175] = (105, 70, 45)
    frame[110:145, 245:275] = (105, 70, 45)
    truth = np.zeros_like(board)
    if book:
        polygon = np.array([[0, 85 + shift], [205, 105 + shift],
                            [195, 179], [0, 179]], np.int32)
        cv2.fillPoly(truth, [polygon], 1)
        frame[truth.astype(bool)] = (100, 70, 45)
    return frame, board.astype(bool), truth.astype(bool)


class TestBookTracker(unittest.TestCase):
    def tracker(self):
        return BookTracker(TrackerConfig(working_width=320))

    def test_blue_artwork_does_not_occlude_a_clean_board(self):
        frame, board, _ = scene(book=False)
        mask, _ = self.tracker().update(frame, board)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(editable_mask(board, mask, 0), board)

    def test_solid_book_is_protected_and_visible_artwork_remains_editable(self):
        frame, board, truth = scene()
        mask, _ = self.tracker().update(frame, board)
        editable = editable_mask(board, mask, 1)
        occluded = truth & board
        visible = board & ~truth
        self.assertGreaterEqual((occluded & ~editable).sum() / occluded.sum(), .98)
        self.assertGreaterEqual((visible & editable).sum() / visible.sum(), .95)
        self.assertFalse((editable & ~board).any())

    def test_motion_does_not_leave_a_mask_trail(self):
        tracker = self.tracker()
        first, board, old = scene()
        tracker.update(first, board)
        second, board, current = scene(shift=35)
        mask, _ = tracker.update(second, board)
        newly_visible = old & ~current & board
        self.assertGreater((newly_visible & ~mask).sum() / newly_visible.sum(), .95)

    def test_disappearance_clears_previous_foreground(self):
        tracker = self.tracker()
        frame, board, _ = scene()
        tracker.update(frame, board)
        clean, board, _ = scene(book=False)
        mask, _ = tracker.update(clean, board)
        self.assertFalse(mask.any())

    def test_small_blue_border_noise_is_rejected(self):
        frame, board, _ = scene(book=False)
        frame[-4:, :4] = (100, 70, 45)
        mask, _ = self.tracker().update(frame, board)
        self.assertFalse(mask.any())

    def test_mismatched_input_shapes_are_rejected(self):
        frame, _, _ = scene()
        with self.assertRaises(ValueError):
            self.tracker().update(frame, np.zeros((90, 160), bool))

    def test_audit_conventions_describe_the_same_physical_regions(self):
        from student_files.prepare_vace_inputs import mask_scores
        _, board, book = scene()
        editable = board & ~book
        occluder_white = mask_scores(editable, board, book & board, 'occluder')
        visible_white = mask_scores(editable, board, board & ~book, 'visible')
        self.assertEqual(occluder_white, visible_white)
        self.assertEqual(visible_white['protected_book_recall'], 1.)
        self.assertEqual(visible_white['visible_poster_editable_fraction'], 1.)

    def test_missing_board_coverage_never_creates_an_edit_region(self):
        frame, board, _ = scene()
        board[:] = False
        book, _ = self.tracker().update(frame, board)
        self.assertFalse(editable_mask(board, book, 2).any())


class TestBoardGapRepair(unittest.TestCase):
    def test_interpolates_camera_translation_without_mutating_predictions(self):
        from student_files.prepare_vace_inputs import interpolate_missing_geometry
        first = np.array([[20., 20.], [100., 20.], [100., 80.], [20., 80.]])
        last = first + [40., 12.]
        predictions = {10: first.copy(), 14: last.copy()}
        repairs = interpolate_missing_geometry(predictions, [11, 12, 13])
        np.testing.assert_allclose(repairs[12]['corners'], first + [20., 6.])
        self.assertEqual(repairs[12]['previous_frame'], 10)
        self.assertEqual(repairs[12]['next_frame'], 14)
        np.testing.assert_array_equal(predictions[10], first)
        np.testing.assert_array_equal(predictions[14], last)

    def test_unbracketed_gap_is_rejected_instead_of_extrapolated(self):
        from student_files.prepare_vace_inputs import interpolate_missing_geometry
        square = np.array([[20., 20.], [100., 20.], [100., 80.], [20., 80.]])
        with self.assertRaises(ValueError):
            interpolate_missing_geometry({14: square}, [10])


@unittest.skipUnless(shutil.which('ffmpeg'), 'Video export requires ffmpeg')
class TestPairedExport(unittest.TestCase):
    def test_gap_repair_preserves_original_masks_and_valid_predictions(self):
        from student_files.prepare_vace_inputs import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, output = root / 'inputs', root / 'prepared'
            (inputs / 'masks').mkdir(parents=True)
            writer = cv2.VideoWriter(str(inputs / 'interval.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 180))
            self.assertTrue(writer.isOpened())
            for index in range(10, 14):
                frame, board, _ = scene(book=False)
                writer.write(frame)
                mask = np.zeros_like(board) if index == 11 else board
                cv2.imwrite(str(inputs / 'masks' / f'mask_{index:06d}.png'),
                            mask.astype(np.uint8) * 255)
            writer.release()
            corners = [None] * 14
            for index in [10, 12, 13]:
                corners[index] = [[120, 30], [289, 30], [289, 159], [120, 159]]
            stats = root / 'stats.json'
            stats.write_text(json.dumps({'source_width': 320, 'resize_width': None,
                                          'corners': corners}))
            metadata = prepare(inputs, output, 10, 13, TrackerConfig(working_width=320),
                               repair_missing=True, stats_path=stats)
            self.assertEqual([r['source_frame'] for r in metadata['board_repairs']], [11])
            self.assertFalse(cv2.imread(str(inputs / 'masks/mask_000011.png'), 0).any())
            repaired = cv2.imread(str(output / 'board_masks/mask_000011.png'), 0) > 127
            np.testing.assert_array_equal(repaired, board)
            for index in [10, 12, 13]:
                name = f'mask_{index:06d}.png'
                np.testing.assert_array_equal(cv2.imread(str(inputs / 'masks' / name), 0),
                                              cv2.imread(str(output / 'board_masks' / name), 0))

    def test_source_indices_and_encoded_masks_stay_aligned(self):
        from student_files.prepare_vace_inputs import prepare
        with tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary) / 'inputs'
            output = Path(temporary) / 'prepared'
            (inputs / 'masks').mkdir(parents=True)
            writer = cv2.VideoWriter(str(inputs / 'interval.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 180))
            self.assertTrue(writer.isOpened())
            for index in range(120, 124):
                frame, board, _ = scene(book=index >= 122)
                writer.write(frame)
                cv2.imwrite(str(inputs / 'masks' / f'mask_{index:06d}.png'),
                            board.astype(np.uint8) * 255)
            writer.release()
            metadata = prepare(inputs, output, 120, 123,
                               TrackerConfig(working_width=320))
            self.assertFalse(metadata['audit_annotations_used_for_preparation'])
            self.assertEqual([r['source_frame'] for r in metadata['frames']], [120, 121, 122, 123])
            capture = cv2.VideoCapture(str(output / 'src_mask.mp4'))
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 30)
            for index in range(120, 124):
                ok, frame = capture.read()
                self.assertTrue(ok)
                decoded = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127
                png = cv2.imread(str(output / 'editable_masks' / f'mask_{index:06d}.png'), 0) > 127
                np.testing.assert_array_equal(decoded, png)
            self.assertFalse(capture.read()[0])
            capture.release()
            with self.assertRaises(FileExistsError):
                prepare(inputs, output, 120, 123)
            with self.assertRaises(ValueError):
                prepare(inputs, Path(temporary) / 'wrong_interval', 120, 124)


if __name__ == '__main__':
    unittest.main()
