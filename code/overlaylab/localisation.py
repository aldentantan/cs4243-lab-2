"""Stage 1 Localise: board localisation from the reference image and single-frame overlay.

The localisation functions compose the keypoint and geometry branches.
Optional feature families and alternative settings are controlled through
``LocalisationConfig``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import LocalisationConfig
from .supplied import to_gray, to_float, rect_corners
from .keypoints import detect_corners, describe_keypoints, match_descriptors
from .geometry import ransac_homography, transfer_points, warp_artwork, rectangle_to_rectangle
from . import metrics


def localise_board(reference: np.ndarray, frame: np.ndarray, config: LocalisationConfig
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Find the board in ``frame`` from the rectified ``reference`` image alone.

    Detect and describe keypoints in both images, match with the ratio test,
    estimate the reference-to-frame homography with RANSAC, and project the
    four reference corners (TL, TR, BR, BL) into the frame.

    Returns (H, corners 4 x 2, matches M x 2 (reference index, frame index),
    inlier_mask M, diagnostics). ``diagnostics`` must contain exactly these keys
    (the notebooks and the grading harness read them): ``n_keypoints_ref``,
    ``n_keypoints_frame``, ``n_matches``, ``n_inliers``, ``inlier_ratio``
    (inliers / ratio-test matches), ``median_residual``, ``iterations``,
    ``status``, ``residuals`` (the per-match symmetric transfer errors e_i from
    RANSAC, length M), ``keypoints_ref`` and ``keypoints_frame`` (the arrays the
    match indices refer to). ``median_residual`` is the median over inliers of
    sqrt(e_i / 2), i.e. the RMS one-way transfer error in pixels.
    When localisation fails, ``H`` is None, ``corners`` is NaN and ``status``
    is False.
    """
    ref_gray = to_gray(reference)
    frame_gray = to_gray(frame)
    kp_ref, _ = detect_corners(ref_gray, config.detector)
    kp_frame, _ = detect_corners(frame_gray, config.detector)
    desc_ref, keep_ref = describe_keypoints(ref_gray, kp_ref, config.descriptor)
    desc_frame, keep_frame = describe_keypoints(frame_gray, kp_frame, config.descriptor)
    # Match indices address descriptor rows, so map through kept_indices before
    # exposing keypoints alongside the returned match array.
    kp_ref = kp_ref[keep_ref]
    kp_frame = kp_frame[keep_frame]
    pairs, _, _ = match_descriptors(desc_ref, desc_frame, config.matcher)
    n_matches = len(pairs)
    inliers = np.zeros(n_matches, dtype=bool)
    residuals = np.full(n_matches, np.inf, dtype=np.float64)
    H = None
    status = False
    iterations = 0
    corners = np.full((4, 2), np.nan, dtype=np.float64)
    n_inliers = 0
    inlier_ratio = 0.0
    median_residual = float("inf")

    if n_matches >= 4:
        src = kp_ref[pairs[:, 0]]
        dst = kp_frame[pairs[:, 1]]
        H, inliers, residuals, iterations, status = ransac_homography(
            src, dst, config.ransac
        )
        n_inliers = int(inliers.sum())
        inlier_ratio = n_inliers / n_matches
        if status and H is not None:
            corners = transfer_points(
                H, rect_corners(ref_gray.shape[1], ref_gray.shape[0])
            )
            valid_residuals = residuals[inliers]
            if len(valid_residuals):
                median_residual = float(np.median(np.sqrt(valid_residuals / 2.0)))
            status = bool(np.all(np.isfinite(corners)))
        if not status:
            H = None
            corners = np.full((4, 2), np.nan, dtype=np.float64)

    diagnostics = {
        "n_keypoints_ref": int(len(kp_ref)),
        "n_keypoints_frame": int(len(kp_frame)),
        "n_matches": int(n_matches),
        "n_inliers": int(n_inliers),
        "inlier_ratio": float(inlier_ratio),
        "median_residual": float(median_residual),
        "iterations": int(iterations),
        "status": bool(status),
        "residuals": residuals,
        "keypoints_ref": kp_ref,
        "keypoints_frame": kp_frame,
    }
    return H, corners, pairs, inliers, diagnostics


def artwork_to_frame(artwork: np.ndarray, H: np.ndarray, config: LocalisationConfig) -> np.ndarray:
    """Compose the artwork rectangle with the board-to-frame homography.

    The artwork is placed by mapping its own pixel rectangle onto the
    reference board rectangle (``config.warp.reference_shape``; the artwork's
    own shape when unset) and then applying ``H``. Document in your report how
    the two transformations compose.
    """
    artwork = np.asarray(artwork)
    if config.warp.reference_shape is None:
        reference_shape = artwork.shape[:2]
    else:
        reference_shape = tuple(config.warp.reference_shape)
    reference_h, reference_w = map(int, reference_shape)
    art_h, art_w = artwork.shape[:2]
    H_artwork_to_reference = rectangle_to_rectangle(
        (art_w, art_h), (reference_w, reference_h)
    )
    return np.asarray(H, dtype=np.float64) @ H_artwork_to_reference


def overlay_frame(frame: np.ndarray, artwork: np.ndarray, H: np.ndarray,
                  config: LocalisationConfig) -> np.ndarray:
    """Single overlaid frame without occlusion handling.

    Warp the artwork with the composed homography (``artwork_to_frame``) and
    paint it where the coverage mask is set. Returns an RGB float32 frame.
    """
    source = to_float(frame).astype(np.float32, copy=False)
    H_artwork = artwork_to_frame(artwork, H, config)
    warped, coverage = warp_artwork(artwork, H_artwork, source.shape)
    if warped.ndim == 2:
        warped = np.repeat(warped[..., None], 3, axis=2)
    if source.ndim == 2:
        source = np.repeat(source[..., None], 3, axis=2)
    out = source.copy()
    out[coverage] = warped[coverage]
    return out.astype(np.float32, copy=False)


def evaluate_audit_frames(reference: np.ndarray, frames: Dict[int, np.ndarray],
                          gt_corners: Dict[int, np.ndarray], config: LocalisationConfig) -> Dict:
    """Run ``localise_board`` on annotated frames and summarise with ``metrics``.

    ``frames`` and ``gt_corners`` map frame index -> image / 4 x 2 corners.
    Returns a dict with ``per_frame`` (index -> dict of error, n_matches,
    n_inliers, inlier_ratio, median_residual, status, corners) and ``summary``
    from ``metrics.summarise_errors`` using ``config.success_threshold``.
    """
    per_frame = {}
    for frame_id in sorted(frames):
        if frame_id not in gt_corners:
            continue
        H, corners, pairs, inliers, diag = localise_board(
            reference, frames[frame_id], config
        )
        error = metrics.corner_error(corners, gt_corners[frame_id])
        per_frame[frame_id] = {
            "error": float(error),
            "n_matches": int(diag["n_matches"]),
            "n_inliers": int(diag["n_inliers"]),
            "inlier_ratio": float(diag["inlier_ratio"]),
            "median_residual": float(diag["median_residual"]),
            "status": bool(diag["status"]),
            "corners": corners,
        }
    summary = metrics.summarise_errors(
        [row["error"] for row in per_frame.values()], config.success_threshold
    )
    return {"per_frame": per_frame, "summary": summary}


__all__ = ["localise_board", "artwork_to_frame", "overlay_frame", "evaluate_audit_frames"]
