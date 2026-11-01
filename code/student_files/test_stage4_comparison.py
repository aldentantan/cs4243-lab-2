"""Exercise Notebook 4's actual comparison helpers and label convention."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

from overlaylab import supplied as sp


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/04_personal_clips_and_genai_analysis.ipynb'


def load_helper(name, namespace):
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell.get('source', [])))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
                return namespace[name]
    raise AssertionError(f'Notebook helper {name} is missing')


class TestVisiblePosterAnnotations(unittest.TestCase):
    def test_conversion_preserves_book_region_and_complete_dictionary(self):
        convert = load_helper('occluder_masks_from_visible', {'sp': sp})
        corners = np.array([[2, 2], [7, 2], [7, 7], [2, 7]], float)
        board = sp.quad_mask((10, 10), corners)
        book = np.zeros_like(board)
        book[2:8, 2:5] = True
        visible = {0: board.copy(), 257: board & ~book, 385: board.copy()}
        originals = {i: m.copy() for i, m in visible.items()}
        converted = convert(visible, {i: corners for i in visible})
        self.assertEqual(set(converted), {0, 257, 385})
        self.assertFalse(converted[0].any())
        self.assertFalse(converted[385].any())
        np.testing.assert_array_equal(converted[257], book)
        self.assertFalse(converted[257][~board].any())
        for i in visible:
            np.testing.assert_array_equal(visible[i], originals[i])

    def test_missing_board_annotation_is_rejected(self):
        convert = load_helper('occluder_masks_from_visible', {'sp': sp})
        with self.assertRaises(ValueError):
            convert({257: np.zeros((10, 10), bool)}, {})


class TestDeclaredTemporalPair(unittest.TestCase):
    def namespace(self):
        calls = []

        def evaluate(paths, *args, **kwargs):
            indices = [int(Path(path).stem.rsplit('_', 1)[1]) for path in paths]
            pairs = [[i, i + 1] for i in indices if i + 1 in indices]
            calls.append((indices, kwargs))
            return {'frames': indices, 'fidelity': {'ncc': .234},
                    'temporal': {'pairs': len(pairs), 'pair_frames': pairs,
                                 'flicker': .15, 'recovered_corner_jitter_px': float('nan')}}

        ns = {'Path': Path, 'evaluate_frames': evaluate, 'video': 'source.mp4',
              'art': np.zeros((4, 4, 3)), 'audit_c': {0: 'complete', 411: 'dictionary'},
              'audit_m': {0: 'complete', 257: 'dictionary'}, 'SUCCESS_THRESHOLD': .01,
              'cond': {'frames': {'pair': [229, 230]}}}
        return ns, calls

    def test_only_true_pair_contributes_and_all_audit_labels_are_shared(self):
        namespace, calls = self.namespace()
        evaluate = load_helper('evaluate_comparison_frames', namespace)
        paths = [Path(f'frame_{i:06d}.png') for i in [137, 228, 229, 230, 257]]
        result = evaluate(paths)
        self.assertEqual([indices for indices, _ in calls], [[137, 228, 229, 230, 257], [229, 230]])
        self.assertEqual(result['frames'], [137, 228, 229, 230, 257])
        self.assertEqual(result['temporal']['pair_frames'], [[229, 230]])
        self.assertEqual(result['temporal']['pairs'], 1)
        self.assertTrue(np.isnan(result['temporal']['recovered_corner_jitter_px']))
        self.assertEqual(result['fidelity']['ncc'], .234)
        for _, kwargs in calls:
            self.assertIs(kwargs['gt_corners'], namespace['audit_c'])
            self.assertIs(kwargs['gt_masks'], namespace['audit_m'])

    def test_missing_pair_frame_is_rejected_before_evaluation(self):
        namespace, calls = self.namespace()
        evaluate = load_helper('evaluate_comparison_frames', namespace)
        with self.assertRaises(ValueError):
            evaluate([Path('frame_000228.png'), Path('frame_000229.png')])
        self.assertEqual(calls, [])

    def test_nonconsecutive_declared_pair_is_rejected(self):
        namespace, calls = self.namespace()
        namespace['cond']['frames']['pair'] = [228, 230]
        evaluate = load_helper('evaluate_comparison_frames', namespace)
        with self.assertRaises(ValueError):
            evaluate([Path('frame_000228.png'), Path('frame_000230.png')])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
