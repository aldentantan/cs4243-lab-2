"""Full-frame navy-book tracking for VACE input preparation.

This is a separate, clip-specific preprocessor, not a change to the frozen
classical pipeline. Inputs are BGR uint8 video pixels and predicted board
coverage. No audit annotation, artwork, or generated output is used. [AI-CODE]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from overlaylab.config import DetectorConfig, FlowConfig
from overlaylab.keypoints import detect_corners
from overlaylab.temporal import lk_flow


@dataclass(frozen=True)
class TrackerConfig:
    working_width: int = 960
    minimum_area_fraction: float = .003
    minimum_outside_fraction: float = .10
    blue_green_ratio: float = 1.10
    green_red_ratio: float = 1.10
    blue_red_difference: float = 15.
    maximum_blue: float = 160.
    line_tolerance: float = 2.5  # pixels at working resolution
    line_support: float = .65
    geometry_iou: float = .90
    padding_at_model_height: float = 2.
    model_height: int = 480


def editable_mask(board: np.ndarray, book: np.ndarray, padding: int) -> np.ndarray:
    board, book = np.asarray(board, bool), np.asarray(book, bool)
    if board.ndim != 2 or board.shape != book.shape or padding < 0:
        raise ValueError('Board and book masks must have matching 2-D shapes; padding must be non-negative')
    if padding and book.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * padding + 1,) * 2)
        book = cv2.dilate(book.astype(np.uint8), kernel).astype(bool)
    return board & ~book


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled.astype(bool)


def _robust_line(x: np.ndarray, y: np.ndarray, tolerance: float,
                 minimum_support: float) -> Optional[Tuple[float, float, float]]:
    """Fit y=a*x+b with deterministic RANSAC, rejecting non-straight boundaries."""
    if len(x) < 20 or np.ptp(x) < 20:
        return None
    stride = max(1, len(x) // 300)
    x, y = x[::stride].astype(float), y[::stride].astype(float)
    rng = np.random.default_rng(4243)
    best = np.zeros(len(x), bool)
    for _ in range(100):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 10:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        support = np.abs(y - (a * x + b)) <= tolerance
        if support.sum() > best.sum():
            best = support
    fraction = float(best.mean())
    if fraction < minimum_support:
        return None
    a, b = np.linalg.lstsq(np.column_stack((x[best], np.ones(best.sum()))),
                           y[best], rcond=None)[0]
    return float(a), float(b), fraction


class BookTracker:
    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        if self.config.working_width < 32 or not 0 < self.config.minimum_area_fraction < 1:
            raise ValueError('Invalid tracker configuration')
        self.previous_gray = None
        self.previous_mask = None

    def _motion_prior(self, gray: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        if self.previous_mask is None or self.previous_gray is None:
            return None, 0
        if self.previous_gray.shape != gray.shape:
            return None, 0
        old = self.previous_mask.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        boundary = (cv2.dilate(old, kernel) - cv2.erode(old, kernel)) * 255
        previous = self.previous_gray.astype(np.float32) / 255.
        current = gray.astype(np.float32) / 255.
        points, _ = detect_corners(previous, DetectorConfig(
            sigma_w=1.5, kappa=.04, response_fraction=.01,
            nms_radius=8, max_corners=1000, border=13))
        xy = np.round(points).astype(int)
        points = points[boundary[xy[:, 1], xy[:, 0]] > 0][:100]
        if len(points) < 4:
            return self.previous_mask, 0
        flow_config = FlowConfig(window=12, iterations=30)
        vectors, _, _, status = lk_flow(previous, current, points, flow_config)
        moved = points + vectors
        reverse_vectors, _, _, reverse = lk_flow(current, previous, moved, flow_config)
        good = status & reverse
        good &= np.linalg.norm(vectors + reverse_vectors, axis=1) < 1.5
        if good.sum() < 4:
            return self.previous_mask, int(good.sum())
        affine, inliers = cv2.estimateAffinePartial2D(points[good], moved[good],
                                                     method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0)
        if affine is None:
            return self.previous_mask, int(good.sum())
        prior = cv2.warpAffine(old, affine, (gray.shape[1], gray.shape[0]),
                               flags=cv2.INTER_NEAREST).astype(bool)
        return prior, int(inliers.sum())

    def _select_component(self, candidate: np.ndarray, board: np.ndarray,
                          prior: Optional[np.ndarray], anchor=None) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
        minimum = max(32, int(candidate.size * self.config.minimum_area_fraction))
        border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        best_score, selected = -1., 0
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum:
                continue
            component = labels == label
            if np.count_nonzero(component & ~board) < max(32, area * self.config.minimum_outside_fraction):
                continue
            overlap = 0. if prior is None else np.count_nonzero(component & prior) / area
            if label not in border and overlap < .5:
                continue
            if anchor is not None and np.count_nonzero(component & anchor) < .5 * anchor.sum():
                continue
            score = area * (1 + overlap)
            if score > best_score:
                best_score, selected = score, label
        return labels == selected if selected else np.zeros(candidate.shape, bool)

    def _straight_edges(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        # This particular foreground is a rectangular cover entering from the
        # lower left. Use that geometry only when colour evidence supports it.
        diagnostic = {'geometry_used': False, 'top_support': 0., 'right_support': 0.}
        if not mask[:, 0].any() or not mask[-1].any():
            return mask, diagnostic
        columns = np.flatnonzero(mask.any(axis=0))
        rows = np.flatnonzero(mask.any(axis=1))
        top = mask[:, columns].argmax(axis=0)
        right = mask.shape[1] - 1 - mask[rows, ::-1].argmax(axis=1)
        cfg = self.config
        top_line = _robust_line(columns, top, cfg.line_tolerance, cfg.line_support)
        right_line = _robust_line(rows, right, cfg.line_tolerance, cfg.line_support)
        if top_line is None or right_line is None:
            return mask, diagnostic
        a, b, top_support = top_line
        c, d, right_support = right_line
        yy, xx = np.indices(mask.shape)
        straight = (yy >= a * xx + b - .5) & (xx <= c * yy + d + .5)
        union = np.count_nonzero(straight | mask)
        agreement = np.count_nonzero(straight & mask) / max(1, union)
        diagnostic.update(top_support=top_support, right_support=right_support,
                          geometry_iou=float(agreement))
        if agreement >= cfg.geometry_iou:
            diagnostic['geometry_used'] = True
            return straight, diagnostic
        return mask, diagnostic

    def update(self, frame_bgr: np.ndarray, board: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if (frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or board.ndim != 2 or frame_bgr.shape[:2] != board.shape):
            raise ValueError('Expected BGR uint8 image and matching 2-D board mask')
        height, width = board.shape
        work_w = min(width, self.config.working_width)
        work_h = max(1, round(height * work_w / width))
        small = cv2.resize(frame_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(board.astype(np.uint8), (work_w, work_h),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prior, motion_points = self._motion_prior(gray)
        blue, green, red = [channel.astype(np.float32) for channel in cv2.split(small)]
        cfg = self.config
        strong = ((blue > cfg.blue_green_ratio * green) &
                  (green > cfg.green_red_ratio * red) &
                  (blue - red > cfg.blue_red_difference) & (blue < cfg.maximum_blue))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        strong = cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        seed = self._select_component(strong, coverage, prior)
        diagnostic = {'book_present': bool(seed.any()), 'motion_points': motion_points,
                      'geometry_used': False, 'seed_area_working': int(seed.sum())}
        if not seed.any():
            self.previous_gray, self.previous_mask = gray, None
            return np.zeros(board.shape, bool), diagnostic

        # Learn colour variation from the foreground outside the board so the
        # original artwork cannot drive the colour model.
        interior = cv2.erode(seed.astype(np.uint8), kernel).astype(bool) & ~coverage
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        samples = lab[interior]
        if len(samples) >= 32:
            samples = samples[::max(1, len(samples) // 20000)]
            centre = np.median(samples, axis=0)
            spread = np.maximum(1.4826 * np.median(np.abs(samples - centre), axis=0), [6., 3., 3.])
            distance = np.sum(((lab - centre) / spread) ** 2, axis=2)
            permissive_blue = ((blue > 1.03 * green) & (green > 1.02 * red) &
                               (blue - red > 8) & (blue < 190))
            candidate = (permissive_blue & (distance < 16)) | seed
            candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                                          kernel).astype(bool)
            selected = self._select_component(candidate, coverage, prior, anchor=seed)
            if selected.any():
                seed = selected
            diagnostic.update(colour_centre_lab=centre.tolist(), colour_spread_lab=spread.tolist())
        filled = _fill_holes(seed)
        refined, geometry = self._straight_edges(filled)
        diagnostic.update(geometry)
        self.previous_gray, self.previous_mask = gray, refined
        full = cv2.resize(refined.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        diagnostic['book_pixels'] = int(full.sum())
        return full, diagnostic


__all__ = ['BookTracker', 'TrackerConfig', 'editable_mask']
