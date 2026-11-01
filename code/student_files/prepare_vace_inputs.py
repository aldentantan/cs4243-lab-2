"""Prepare paired VACE inputs with a dedicated navy-book tracker.

From code/:
    python student_files/prepare_vace_inputs.py prepare
    python student_files/prepare_vace_inputs.py prepare --repair-missing-board
    python student_files/prepare_vace_inputs.py evaluate
    python student_files/prepare_vace_inputs.py evaluate --audit-mask-kind visible

Preparation reads the video and predicted board masks only. Evaluation is a
separate invocation that loads audit labels after all predictions are fixed.
All frame filenames retain the original source indices.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from overlaylab import supplied as sp
from student_files.book_tracking import BookTracker, TrackerConfig, editable_mask
DEFAULT_INPUTS = CODE / 'outputs/genai/inputs'
DEFAULT_OUTPUT = DEFAULT_INPUTS / 'book_tracking_v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate_missing_geometry(predictions, missing):
    """Bracket each gap with valid predictions; never extrapolate or read labels."""
    missing = set(missing)
    valid = {int(i): np.asarray(q, dtype=float) for i, q in predictions.items()
             if i not in missing and np.asarray(q).shape == (4, 2) and np.isfinite(q).all()}
    indices = sorted(valid)
    repairs = {}
    for index in sorted(missing):
        position = bisect_left(indices, index)
        if position == 0 or position == len(indices):
            raise ValueError(f'Cannot bracket source frame {index} with valid predictions')
        previous, following = indices[position - 1], indices[position]
        weight = (index - previous) / (following - previous)
        corners = (1 - weight) * valid[previous] + weight * valid[following]
        polygon = corners.astype(np.float32)
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 1:
            raise ValueError(f'Interpolated board at frame {index} is degenerate')
        repairs[index] = {'previous_frame': previous, 'next_frame': following,
                          'weight': weight, 'gap_length': following - previous - 1,
                          'corners': corners.tolist()}
    return repairs


class VideoEncoder:
    def __init__(self, path, width, height, fps, grayscale=False, lossless=False):
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('ffmpeg is required for portable H.264 video export')
        self.process = subprocess.Popen([
            executable, '-v', 'error', '-n', '-f', 'rawvideo',
            '-pix_fmt', 'gray' if grayscale else 'bgr24',
            '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '0' if lossless else '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
        ], stdin=subprocess.PIPE)

    def write(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError('ffmpeg video encoding failed')


def _preview(frame, prepared, board, book, index):
    scale = min(1., 640 / frame.shape[1])
    size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
    left = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(prepared, size, interpolation=cv2.INTER_AREA)
    for mask, colour in [(board, (255, 255, 0)), (book, (0, 0, 255))]:
        small = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, contours, -1, colour, 1)
    for panel, text in [(left, f'Source {index}: book=red, board=cyan'),
                        (right, 'Prepared input: gray = editable')]:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), cv2.FILLED)
        cv2.putText(panel, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((left, right), axis=1)


def prepare(inputs: Path, output: Path, start: int, end: int,
            config: TrackerConfig = TrackerConfig(), repair_missing=False, stats_path=None):
    if output.exists():
        raise FileExistsError(f'Preserving existing output: {output}. Choose a new --out directory.')
    if start < 0 or end < start:
        raise ValueError('Expected a non-negative inclusive source interval')
    info = sp.video_info(inputs / 'interval.mp4')
    if info['frames'] != end - start + 1:
        raise ValueError('Source interval length differs from input video frame count')
    if info['width'] % 2 or info['height'] % 2:
        raise ValueError('H.264 yuv420p export requires even video dimensions')
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required')
    mask_paths = [inputs / 'masks' / f'mask_{i:06d}.png' for i in range(start, end + 1)]
    original_board_digest = hashlib.sha256()
    missing = []
    for path in mask_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        mask = sp.read_mask(path)
        if mask.shape != (info['height'], info['width']):
            raise ValueError(f'Mask dimensions differ from video: {path}')
        if not mask.any():
            missing.append(int(path.stem.rsplit('_', 1)[1]))
        original_board_digest.update(path.name.encode())
        original_board_digest.update(path.read_bytes())
    repairs, stats_hash = {}, None
    if repair_missing and missing:
        if stats_path is None:
            raise ValueError('Board repair requires saved frozen tracking statistics')
        stats = json.loads(stats_path.read_text())
        processing_width = stats.get('resize_width') or stats['source_width']
        factor = info['width'] / processing_width
        predictions = {i: np.asarray(q, dtype=float) * factor
                       for i, q in enumerate(stats['corners'])}
        repairs = interpolate_missing_geometry(predictions, missing)
        stats_hash = sha256(stats_path)
        print(f'Repairing {len(repairs)} empty board masks from neighbouring predicted corners', flush=True)
    metadata = {
        'source_interval': {'start': start, 'end': end}, 'video_info': info,
        'input_video': str((inputs / 'interval.mp4').resolve()),
        'input_video_sha256': sha256(inputs / 'interval.mp4'),
        'predicted_board_masks': str((output / 'board_masks').resolve()),
        'original_board_masks': str((inputs / 'masks').resolve()),
        'original_board_masks_sha256': original_board_digest.hexdigest(),
        'original_empty_board_frames': missing,
        'board_repairs': [{'source_frame': i, **repair} for i, repair in repairs.items()],
        'tracking_stats_sha256': stats_hash,
        'tracker_config': asdict(config), 'audit_annotations_used_for_preparation': False,
        'versions': {'opencv': cv2.__version__, 'numpy': np.__version__},
        'implementation_sha256': {
            'prepare_vace_inputs.py': sha256(__file__),
            'book_tracking.py': sha256(CODE / 'student_files/book_tracking.py'),
        },
        'formula': 'editable = predicted_board & ~dilate(predicted_book); gray-fill editable with 127',
        'note': 'Dedicated extra preprocessing; the classical pipeline and its frozen thresholds are unchanged.',
    }
    output.mkdir(parents=True)
    for folder in ['book_masks', 'editable_masks', 'board_masks']:
        (output / folder).mkdir()
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    tracker = BookTracker(config)
    padding = math.ceil(config.padding_at_model_height * info['height'] / config.model_height)
    metadata['padding_source_pixels'] = padding
    capture = cv2.VideoCapture(str(inputs / 'interval.mp4'))
    if not capture.isOpened():
        raise RuntimeError('Cannot decode input video')
    encoders, partials, records = [], [], []
    board_digest = hashlib.sha256()
    try:
        for index, path in enumerate(mask_paths, start=start):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f'Input video ended before source frame {index}')
            board = sp.read_mask(path)
            if index in repairs:
                board = sp.quad_mask(board.shape, np.asarray(repairs[index]['corners']))
                if not board.any():
                    raise ValueError(f'Interpolated board falls outside the frame: {index}')
            board_path = output / 'board_masks' / path.name
            sp.write_mask(board_path, board)
            board_digest.update(board_path.name.encode()); board_digest.update(board_path.read_bytes())
            book, diagnostic = tracker.update(frame, board)
            editable = editable_mask(board, book, padding)
            prepared = frame.copy()
            prepared[editable] = 127
            sp.write_mask(output / 'book_masks' / path.name, book)
            sp.write_mask(output / 'editable_masks' / path.name, editable)
            preview = _preview(frame, prepared, board, book, index)
            if not encoders:
                for name, image, gray, lossless in [
                    ('interval_inpaint', prepared, False, False),
                    ('src_mask', editable, True, True),
                    ('preview', preview, False, False),
                ]:
                    partial = output / f'{name}.partial.mp4'
                    partials.append((partial, output / f'{name}.mp4'))
                    encoders.append(VideoEncoder(partial, image.shape[1], image.shape[0],
                                                  info['fps'], gray, lossless))
            encoders[0].write(prepared)
            encoders[1].write(editable.astype(np.uint8) * 255)
            encoders[2].write(preview)
            records.append({'source_frame': index, 'input_frame': index - start,
                            'board_pixels': int(board.sum()), 'editable_pixels': int(editable.sum()),
                            'board_repaired': index in repairs,
                            **diagnostic})
            if (index - start + 1) % 50 == 0:
                print(f'Prepared {index - start + 1}/{len(mask_paths)} frames', flush=True)
        ok, _ = capture.read()
        if ok:
            raise RuntimeError('Input video has extra decoded frames')
    finally:
        capture.release()
        for encoder in encoders:
            encoder.close()
    for partial, final in partials:
        partial.rename(final)
    metadata['frames'] = records
    metadata['board_masks_sha256'] = board_digest.hexdigest()
    metadata['outputs_sha256'] = {path.name: sha256(path) for _, path in partials}
    metadata['complete'] = True
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Prepared inputs: {output.resolve()}', flush=True)
    return metadata


def mask_scores(editable, board, annotation, annotation_kind):
    """Interpret label polarity explicitly; never change source annotations."""
    if annotation_kind == 'occluder':
        occluded = board & annotation
        visible = board & ~annotation
    elif annotation_kind == 'visible':
        occluded = board & ~annotation
        visible = board & annotation
    else:
        raise ValueError('Expected occluder or visible audit-mask convention')
    recall = (float(np.count_nonzero(occluded & ~editable) / occluded.sum())
              if occluded.any() else None)
    coverage = float(np.count_nonzero(visible & editable) / max(1, visible.sum()))
    warning = None
    if occluded.any() and occluded.sum() / max(1, board.sum()) < .05:
        distance = cv2.distanceTransform(board.astype(np.uint8), cv2.DIST_L2, 5)
        if np.mean(distance[occluded] <= 5) > .9:
            warning = 'Derived occlusion is mostly a thin board-border band; verify annotation outline/polarity.'
    return {'occluded_pixels': int(occluded.sum()), 'protected_book_recall': recall,
            'visible_poster_editable_fraction': coverage,
            'book_recall_pass': recall is None or recall >= .98,
            'visible_coverage_pass': coverage >= .95,
            'annotation_warning': warning}


def evaluate(output: Path, clip: str, conditions: Path, audit_mask_kind='occluder'):
    # Audit annotations are intentionally imported and read only by this command.
    from overlaylab import data
    metadata = json.loads((output / 'preparation.json').read_text())
    if not metadata.get('complete'):
        raise ValueError('Preparation did not complete')
    for name, expected in metadata['outputs_sha256'].items():
        if sha256(output / name) != expected:
            raise ValueError(f'Generated input changed since preparation: {name}')
    start, end = [metadata['source_interval'][k] for k in ['start', 'end']]
    info = metadata['video_info']
    exact_video_info = all(sp.video_info(output / name) == info
                           for name in ['interval_inpaint.mp4', 'src_mask.mp4'])
    capture = cv2.VideoCapture(str(output / 'src_mask.mp4'))
    disagreement, mask_digest, board_digest = 0, hashlib.sha256(), hashlib.sha256()
    try:
        for i in range(start, end + 1):
            path = output / 'editable_masks' / f'mask_{i:06d}.png'
            editable = sp.read_mask(path)
            mask_digest.update(path.name.encode()); mask_digest.update(path.read_bytes())
            book = sp.read_mask(output / 'book_masks' / path.name)
            board_path = Path(metadata['predicted_board_masks']) / path.name
            board = sp.read_mask(board_path)
            board_digest.update(board_path.name.encode()); board_digest.update(board_path.read_bytes())
            if not np.array_equal(editable, editable_mask(board, book, metadata['padding_source_pixels'])):
                raise ValueError(f'PNG mask does not match preparation formula: {i}')
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Encoded mask video ended early')
            decoded = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127
            disagreement += int(np.count_nonzero(decoded != editable))
    finally:
        capture.release()
    if board_digest.hexdigest() != metadata['board_masks_sha256']:
        raise ValueError('Predicted board masks changed since preparation')
    audit_masks, audit_corners = data.load_audit_masks(clip), data.load_audit_corners(clip)
    cond = json.loads(conditions.read_text())
    selected = {i: (mask, audit_mask_kind) for i, mask in audit_masks.items() if start <= i <= end}
    clean = cond['frames']['clean']
    if start <= clean <= end and clean not in selected:
        selected[clean] = (np.zeros((info['height'], info['width']), bool), 'occluder')
    rows = []
    for index, (annotation, kind) in sorted(selected.items()):
        board = sp.quad_mask(annotation.shape, audit_corners[index])
        editable = sp.read_mask(output / 'editable_masks' / f'mask_{index:06d}.png')
        rows.append({'source_frame': index, 'annotation_kind': kind,
                     **mask_scores(editable, board, annotation, kind)})
    empty_board = [r['source_frame'] for r in metadata['frames'] if r['board_pixels'] == 0]
    annotation_review = any(row['annotation_warning'] for row in rows)
    passed = (exact_video_info and disagreement == 0 and not empty_board and
              all(row['book_recall_pass'] and row['visible_coverage_pass'] for row in rows))
    report = {
        'evaluation_only': True, 'audit_labels_used_after_prediction': True,
        'audit_mask_kind': audit_mask_kind, 'audit_files_modified': False,
        'targets': {'protected_book_recall': .98, 'visible_poster_editable_fraction': .95},
        'matching_video_info': exact_video_info, 'encoded_mask_disagreement_pixels': disagreement,
        'empty_board_source_frames': empty_board,
        'editable_png_sha256': mask_digest.hexdigest(), 'audit_frames': rows,
        'all_criteria_pass': passed,
        'status': 'requires_annotation_review' if annotation_review else ('passed' if passed else 'failed'),
        'limitation': 'Sparse audit frames do not certify every boundary on all 150 frames or VACE output preservation.',
    }
    (output / f'acceptance_report_{audit_mask_kind}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'evaluate'])
    parser.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--conditions', type=Path, default=CODE / 'manifests/genai_conditions.json')
    parser.add_argument('--clip', default='clip')
    parser.add_argument('--repair-missing-board', action='store_true',
                        help='Interpolate empty board masks from neighbouring frozen predictions')
    parser.add_argument('--stats', type=Path, default=CODE / 'outputs/videos/clip_flow_assisted_stats.json')
    parser.add_argument('--audit-mask-kind', choices=['occluder', 'visible'], default='occluder',
                        help='Explicit audit PNG convention for evaluation only; lab default is occluder-white')
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_INPUTS / ('book_tracking_repaired_v1' if args.repair_missing_board
                                    else 'book_tracking_v1')
    if args.action == 'prepare':
        conditions = json.loads(args.conditions.read_text())
        interval = conditions['model']['interval']
        prepare(args.inputs, args.out, interval['start'], interval['end'],
                repair_missing=args.repair_missing_board, stats_path=args.stats)
    else:
        evaluate(args.out, args.clip, args.conditions, args.audit_mask_kind)


if __name__ == '__main__':
    main()
