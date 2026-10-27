"""Stage 3 Composite: occlusion handling and compositing.

Implement the functions marked ``YOUR CODE HERE``. Motion-compensated
differencing is the required occlusion signal; ``mode="colour"`` is an
optional ablation. Permitted library calls: ``cv2.morphologyEx``,
``cv2.connectedComponentsWithStats``, ``cv2.distanceTransform`` (feathering),
and your Lab 1 ``hysteresis``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import OcclusionConfig, CompositeConfig
from .supplied import to_gray, to_float
from .geometry import warp_artwork
from . import metrics


def hysteresis(strong: np.ndarray, weak: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Keep weak pixels reachable from strong pixels (reuse your Lab 1 implementation).

    ``strong`` and ``weak`` are Boolean masks of the same shape (weak includes
    strong). Traverse with the 4- or 8-connected neighbourhood from every
    strong pixel, through chains of weak pixels, and return the Boolean mask
    of reached pixels. Validate shapes and the connectivity value.
    """
    strong = np.asarray(strong, dtype=bool)
    weak = np.asarray(weak, dtype=bool)
    if strong.ndim != 2 or weak.ndim != 2 or strong.shape != weak.shape:
        raise ValueError("strong and weak must be 2-D masks with the same shape")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if np.any(strong & ~weak):
        raise ValueError("weak must include every strong pixel")
    if not strong.any():
        return np.zeros_like(strong)

    # Connected components of the weak mask are precisely the regions
    # reachable from one another through weak pixels. Keep only regions
    # containing at least one strong seed.
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=connectivity)
    seeded = np.unique(labels[strong])
    seeded = seeded[seeded != 0]
    if not len(seeded):
        return np.zeros_like(strong)
    return np.isin(labels, seeded) & (labels != 0)


def compensated_difference(frame: np.ndarray, warped_reference: np.ndarray, coverage: np.ndarray,
                           config: OcclusionConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Per-pixel difference after a global gain/offset fit, zero outside ``coverage``.

    Fit I_frame ~ a * I_ref + b on the ``config.gain_quantile`` fraction of
    covered pixels whose raw difference is smallest (so that illumination
    change alone does not become an occluder), then return the mean absolute
    difference over the colour channels between the frame and the compensated
    reference, together with (a, b). Use colour, not grayscale alone: a
    coloured occluder over a coloured board can have almost the same luminance
    (the ball over the blue board differs by only ~0.13 in grayscale but ~0.3
    in mean channel difference), and the default thresholds assume colour.
    """
    frame_f = to_float(frame)
    ref_f = to_float(warped_reference)
    coverage = np.asarray(coverage, dtype=bool)
    if frame_f.shape != ref_f.shape or frame_f.ndim not in (2, 3):
        raise ValueError("frame and warped_reference must have the same 2-D or 3-D shape")
    if coverage.shape != frame_f.shape[:2]:
        raise ValueError("coverage must match the image height and width")
    if not 0.0 < config.gain_quantile <= 1.0:
        raise ValueError("gain_quantile must be in (0, 1]")

    channel_difference = np.abs(frame_f - ref_f)
    raw_difference = channel_difference if frame_f.ndim == 2 else channel_difference.mean(axis=2)
    covered = np.flatnonzero(coverage.ravel())
    diff = np.zeros(coverage.shape, dtype=np.float32)
    if not len(covered):
        return diff, (1.0, 0.0)

    # Choose the most similar covered pixels before fitting the global
    # photometric transform, so foreground objects do not drive the fit.
    order = np.argsort(raw_difference.ravel()[covered], kind="stable")
    n_fit = max(1, int(np.ceil(config.gain_quantile * len(covered))))
    selected = covered[order[:n_fit]]
    if frame_f.ndim == 2:
        x = ref_f.ravel()[selected].astype(np.float64)
        y = frame_f.ravel()[selected].astype(np.float64)
    else:
        x = ref_f.reshape(-1, ref_f.shape[2])[selected].astype(np.float64).ravel()
        y = frame_f.reshape(-1, frame_f.shape[2])[selected].astype(np.float64).ravel()
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_centered = x - x_mean
    variance = float(np.dot(x_centered, x_centered))
    if variance <= np.finfo(np.float64).eps:
        gain = 1.0
        offset = y_mean - x_mean
    else:
        gain = float(np.dot(x_centered, y - y_mean) / variance)
        offset = float(y_mean - gain * x_mean)

    compensated = gain * ref_f.astype(np.float64) + offset
    absolute = np.abs(frame_f.astype(np.float64) - compensated)
    if absolute.ndim == 3:
        absolute = absolute.mean(axis=2)
    diff[coverage] = absolute[coverage].astype(np.float32)
    return diff, (gain, offset)


def occlusion_mask(frame: np.ndarray, H: np.ndarray, reference: np.ndarray, config: OcclusionConfig
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Foreground mask inside the board by motion-compensated differencing.

    1. Warp ``reference`` into the frame with ``H`` (``warp_artwork``) and
       compute the ``compensated_difference`` inside the coverage mask.
    2. Threshold at ``config.strong`` and ``config.weak`` and apply
       ``hysteresis`` with ``config.connectivity``.
    3. Open with a disc of radius ``config.open_radius`` (``cv2.morphologyEx``:
       erosion then dilation, which deletes anything thinner than 2r+1 pixels;
       0 disables) and keep connected components with at least
       ``config.min_area`` pixels.
    4. Return (mask bool H x W, difference map float H x W, component stats).

    ``component stats`` is an array with one row per kept component:
    [label, x, y, width, height, area]. With ``config.mode == "colour"`` an
    appearance model may be used instead (optional ablation).
    """
    frame_f = to_float(frame)
    reference_f = to_float(reference)
    if config.mode == "colour":
        raise NotImplementedError('mode="colour" is an optional ablation')
    if config.mode != "difference":
        raise ValueError("mode must be 'difference' or 'colour'")
    warped_reference, coverage = warp_artwork(reference_f, H, frame_f.shape)
    diff, _ = compensated_difference(frame_f, warped_reference, coverage, config)
    candidate = hysteresis(diff >= config.strong, diff >= config.weak, config.connectivity)

    if config.open_radius > 0 and candidate.any():
        radius = int(config.open_radius)
        size = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    if config.min_area < 0:
        raise ValueError("min_area must be non-negative")
    if not candidate.any():
        return candidate, diff, np.empty((0, 6), dtype=np.int32)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=config.connectivity
    )
    kept_labels = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= config.min_area]
    mask = np.isin(labels, kept_labels) if kept_labels else np.zeros_like(candidate)
    components = np.asarray([
        [label, stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP],
         stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT],
         stats[label, cv2.CC_STAT_AREA]]
        for label in kept_labels
    ], dtype=np.int32).reshape(-1, 6)
    return mask, diff, components


def select_difference_thresholds(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                 masks: Sequence[np.ndarray], config: OcclusionConfig
                                 ) -> Tuple[OcclusionConfig, List[Dict]]:
    """Choose strong/weak thresholds and the minimum area on validation masks only.

    ``frames`` is a sequence of (frame, H, reference) triples and ``masks`` the
    matching ground-truth occlusion masks. Grid-search the thresholds and the
    minimum area, score each setting by mean ``metrics.mask_iou`` over the
    validation frames, and return (frozen config, search log) where the log
    holds one dict per setting with its parameters and score.

    Several settings usually tie at the best score: the few validation masks
    do not separate the strong threshold. Break ties towards the middle of the
    tied range of ``strong`` values (not the first or the largest grid point),
    so that the frozen value sits inside the plateau rather than on its edge,
    and state the rule in the report.
    """
    if len(frames) != len(masks) or not frames:
        raise ValueError("frames and masks must be non-empty sequences of equal length")
    strong_values = (0.15, 0.20, 0.25, 0.30, 0.35)
    weak_ratios = (0.4, 0.6, 0.8)
    area_values = sorted({max(1, int(round(config.min_area * scale))) for scale in (0.5, 1.0, 2.0)})
    log: List[Dict] = []
    for strong in strong_values:
        for ratio in weak_ratios:
            weak = strong * ratio
            for area in area_values:
                candidate_config = replace(config, strong=strong, weak=weak, min_area=area)
                scores = []
                for (frame, H, reference), gt in zip(frames, masks):
                    predicted, _, _ = occlusion_mask(frame, H, reference, candidate_config)
                    scores.append(metrics.mask_iou(predicted, gt))
                finite_scores = [score for score in scores if np.isfinite(score)]
                score = float(np.mean(finite_scores)) if finite_scores else float("nan")
                log.append({"strong": float(strong), "weak": float(weak),
                            "min_area": int(area), "mean_iou": score})

    finite = [row["mean_iou"] for row in log if np.isfinite(row["mean_iou"])]
    if not finite:
        raise ValueError("threshold search produced no finite IoU scores")
    best_score = max(finite)
    tied = [row for row in log if np.isclose(row["mean_iou"], best_score, rtol=0.0, atol=1e-12)]
    tied_strong = sorted({row["strong"] for row in tied})
    middle_strong = float(np.median(tied_strong))
    # If tied strong values straddle the median, prefer the lower value for
    # deterministic selection; then keep the original weak/area settings as
    # close as possible among the remaining ties.
    chosen_strong = min(tied_strong, key=lambda value: (abs(value - middle_strong), value))
    same_strong = [row for row in tied if row["strong"] == chosen_strong]
    chosen = min(same_strong, key=lambda row: (
        abs(row["weak"] - config.weak), abs(row["min_area"] - config.min_area),
        row["weak"], row["min_area"]
    ))
    frozen = replace(config, strong=chosen["strong"], weak=chosen["weak"], min_area=chosen["min_area"])
    return frozen, log


def composite(frame: np.ndarray, warped: np.ndarray, coverage: np.ndarray, occlusion: np.ndarray,
              config: CompositeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-composite the warped artwork into the frame.

    alpha = C * (1 - O) * F, where C is the coverage mask, O the occlusion
    mask and F an optional feathering term that softens the quadrilateral and
    occluder boundaries over ``config.feather`` pixels (0 disables).
    I_out = alpha * W + (1 - alpha) * I. Optional brightness matching of the
    artwork to the board is controlled by ``config.match_gain``.
    Returns (composited RGB float32 frame, alpha map float32 in [0, 1]).
    """
    frame_f = to_float(frame)
    warped_f = to_float(warped)
    coverage = np.asarray(coverage, dtype=bool)
    occlusion = np.asarray(occlusion, dtype=bool)
    if frame_f.shape != warped_f.shape or frame_f.ndim not in (2, 3):
        raise ValueError("frame and warped artwork must have the same 2-D or 3-D shape")
    if coverage.shape != frame_f.shape[:2] or occlusion.shape != coverage.shape:
        raise ValueError("coverage and occlusion must match the image height and width")

    visible = coverage & ~occlusion
    alpha = visible.astype(np.float32)
    feather = int(config.feather)
    if feather < 0:
        raise ValueError("feather must be non-negative")
    if feather > 0 and visible.any():
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 3)
        alpha *= np.clip(distance / float(feather), 0.0, 1.0).astype(np.float32)

    if config.match_gain and visible.any():
        source_luma = to_gray(warped_f)[visible]
        target_luma = to_gray(frame_f)[visible]
        denom = float(np.dot(source_luma, source_luma))
        gain = float(np.dot(source_luma, target_luma) / denom) if denom > np.finfo(float).eps else 1.0
        if warped_f.ndim == 3:
            warped_f = warped_f * gain
        else:
            warped_f = warped_f * gain

    if frame_f.ndim == 3:
        alpha_3 = alpha[..., None]
    else:
        alpha_3 = alpha
    out = alpha_3 * warped_f + (1.0 - alpha_3) * frame_f
    return out.astype(np.float32), alpha.astype(np.float32)


__all__ = ["hysteresis", "compensated_difference", "occlusion_mask", "select_difference_thresholds", "composite"]
