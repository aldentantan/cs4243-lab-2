"""Stage 1 Localise, geometry branch.

Implement the functions marked ``YOUR CODE HERE``. Allowed: numpy (including
``numpy.linalg.svd`` and ``numpy.linalg.inv``) and your Lab 1
``bilinear_sample`` (pasted into this file). ``cv2.findHomography`` and ``cv2.warpPerspective``
may appear only in the labelled sanity-check and comparison cells of Notebook 1.

Conventions: points are N x 2 arrays of (x, y) with x indexing columns; a
homography H maps source homogeneous coordinates to destination ones,
x' ~ H x, and is scaled so that H[2, 2] = 1 whenever that entry is not close
to zero. Pixel centres sit at integer coordinates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import RansacConfig


def bilinear_sample(image: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample a 2-D or H x W x C image at floating-point image coordinates (reuse Lab 1).

    Same signature as Lab 1 (``y`` first, then ``x``); extend it so that a
    colour image returns an array with a trailing channel axis. Coordinates
    are clamped to the valid closed interval, and an exact integer coordinate
    reproduces the corresponding pixel. ``y`` and ``x`` may be arrays of any
    matching shape so that a whole grid can be sampled at once.
    """
    image = np.asarray(image)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y coordinates must have matching shapes")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D colour array")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("cannot sample an image with an empty spatial dimension")

    dtype = np.result_type(image.dtype, np.float32)
    img = image.astype(dtype, copy=False)
    yy = np.clip(y, 0.0, float(height - 1))
    xx = np.clip(x, 0.0, float(width - 1))

    y0 = np.floor(yy).astype(np.intp)
    x0 = np.floor(xx).astype(np.intp)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = yy - y0
    wx = xx - x0

    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]
    if image.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]

    top = (1.0 - wx) * p00 + wx * p01
    bottom = (1.0 - wx) * p10 + wx * p11
    return (1.0 - wy) * top + wy * bottom


def rectangle_to_rectangle(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    """Supplied helper: affine scaling that maps a (w, h) pixel rectangle onto another.

    Corner (w-1, h-1) of the source maps to corner (w'-1, h'-1) of the
    destination (pixel-centre convention). Used to compose the artwork
    rectangle with the board-to-frame homography.
    """
    sw, sh = src_size
    dw, dh = dst_size
    return np.array([[(dw - 1) / max(sw - 1, 1), 0, 0],
                     [0, (dh - 1) / max(sh - 1, 1), 0],
                     [0, 0, 1.0]], dtype=np.float64)


def normalise_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation.

    Returns (T, normalised) where T is the 3 x 3 similarity that translates the
    points to zero centroid and scales them so that the mean distance from the
    origin is sqrt(2), and ``normalised`` are the transformed N x 2 points.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1:] != (2,):
        raise ValueError("pts must be an N x 2 array")
    if len(pts) == 0:
        raise ValueError("at least one point is required")
    if not np.all(np.isfinite(pts)):
        raise ValueError("pts must contain only finite coordinates")

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    mean_distance = np.linalg.norm(centered, axis=1).mean()
    # A zero-spread set cannot be scaled to mean distance sqrt(2). Keep a
    # finite translation transform; downstream DLT rank checks reject it.
    scale = np.sqrt(2.0) / mean_distance if mean_distance > np.finfo(float).eps else 1.0
    T = np.array([[scale, 0.0, -scale * centroid[0]],
                  [0.0, scale, -scale * centroid[1]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    normalized = centered * scale
    return T, normalized


def dlt_homography(src: np.ndarray, dst: np.ndarray
                   ) -> Tuple[Optional[np.ndarray], np.ndarray, bool]:
    """Normalised direct linear transform.

    Builds the 2n x 9 system A h = 0 from normalised correspondences, solves
    for h as the right singular vector with the smallest singular value,
    denormalises H = T'^{-1} H_hat T, and scales so that H[2, 2] = 1 when that
    entry is not close to zero.

    Raises ``ValueError`` for fewer than four correspondences. Returns
    (H, singular_values, status); when A is rank deficient (e.g. collinear
    source points) ``status`` is False and ``H`` is None rather than a
    numerically meaningless matrix. ``singular_values`` are those of A, so
    that conditioning can be inspected.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (2,) or dst.shape[1:] != (2,):
        raise ValueError("src and dst must be N x 2 point arrays")
    if len(src) != len(dst):
        raise ValueError("src and dst must contain the same number of points")
    if len(src) < 4:
        raise ValueError("at least four correspondences are required")

    T_src, src_n = normalise_points(src)
    T_dst, dst_n = normalise_points(dst)
    n = len(src)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    x, y = src_n[:, 0], src_n[:, 1]
    u, v = dst_n[:, 0], dst_n[:, 1]

    A[0::2, 0:3] = np.column_stack((-x, -y, -np.ones(n)))
    A[0::2, 6:9] = np.column_stack((u * x, u * y, u))
    A[1::2, 3:6] = np.column_stack((-x, -y, -np.ones(n)))
    A[1::2, 6:9] = np.column_stack((v * x, v * y, v))

    try:
        _, singular_values, Vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None, np.empty((0,), dtype=np.float64), False

    # A unique homography requires rank at least eight. With exactly four
    # pairs, A is 8 x 9 and the last row of full Vt spans its null space.
    tolerance = np.finfo(np.float64).eps * max(A.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 8:
        return None, singular_values, False

    H_normalized = Vt[-1].reshape(3, 3)
    try:
        H = np.linalg.inv(T_dst) @ H_normalized @ T_src
    except np.linalg.LinAlgError:
        return None, singular_values, False

    if not np.all(np.isfinite(H)):
        return None, singular_values, False
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm <= np.finfo(np.float64).eps:
            return None, singular_values, False
        H = H / norm
        # Fix the otherwise arbitrary sign for deterministic output.
        pivot = np.unravel_index(np.argmax(np.abs(H)), H.shape)
        if H[pivot] < 0:
            H = -H
    return H, singular_values, True


def transfer_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to N x 2 points and dehomogenise: pi(H x)."""
    H = np.asarray(H, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((pts, np.ones(len(pts), dtype=np.float64)))
    projected = homogeneous @ H.T
    denominator = projected[:, 2]

    result = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=1) & (np.abs(denominator) > 1e-12)
    result[valid] = projected[valid, :2] / denominator[valid, None]
    return result


def symmetric_transfer_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-correspondence e_i = |x' - pi(H x)|^2 + |x - pi(H^-1 x')|^2."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (2,) or dst.shape[1:] != (2,):
        raise ValueError("src and dst must be N x 2 point arrays")
    if len(src) != len(dst):
        raise ValueError("src and dst must contain the same number of points")

    try:
        H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.full(len(src), np.inf, dtype=np.float64)

    forward = transfer_points(H, src)
    backward = transfer_points(H_inv, dst)
    forward_sq = np.sum((forward - dst) ** 2, axis=1)
    backward_sq = np.sum((backward - src) ** 2, axis=1)
    errors = forward_sq + backward_sq
    errors[~np.isfinite(errors)] = np.inf
    return errors


def _three_collinear(p: np.ndarray, tol: float = 1e-6) -> bool:
    """Supplied helper: True if any three of the four points are (nearly) collinear."""
    from itertools import combinations
    scale = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
    for i, j, k in combinations(range(len(p)), 3):
        a, b, c = p[i], p[j], p[k]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 < tol * scale * scale:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray, config: RansacConfig
                      ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, int, bool]:
    """Robust homography with adaptive RANSAC and a final refit on all inliers.

    Each iteration draws four correspondences with ``config.rng()``, rejects
    the sample if any three source points are collinear, fits
    ``dlt_homography``, and scores every correspondence by the symmetric
    transfer error; correspondences under ``config.threshold`` are inliers.
    Keep the model with the most inliers (ties: smaller sum of inlier
    residuals); update N = log(1-p) / log(1-(1-eps)^4) from the current inlier
    fraction, capped by ``config.max_iterations``; refit on all inliers.

    Returns (H, inlier_mask, residuals, iterations_used, status) where status
    is False when no model reached ``config.min_inliers`` (H may then be None).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1:] != (2,) or dst.shape[1:] != (2,):
        raise ValueError("src and dst must be N x 2 point arrays")
    if len(src) != len(dst):
        raise ValueError("src and dst must contain the same number of points")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise ValueError("src and dst must contain only finite coordinates")

    n = len(src)
    empty_mask = np.zeros(n, dtype=bool)
    empty_residuals = np.full(n, np.inf, dtype=np.float64)
    max_iterations = max(0, int(config.max_iterations))
    if n < 4 or max_iterations == 0:
        return None, empty_mask, empty_residuals, 0, False

    rng = config.rng()
    best_H = None
    best_mask = empty_mask
    best_residuals = empty_residuals
    best_count = 0
    best_sum = np.inf
    iterations = 0
    iteration_limit = max_iterations
    threshold = float(config.threshold)
    min_inliers = max(0, int(config.min_inliers))
    confidence = float(config.confidence)

    while iterations < iteration_limit:
        sample_idx = rng.choice(n, size=4, replace=False)
        iterations += 1
        src_sample = src[sample_idx]
        if _three_collinear(src_sample):
            continue

        candidate, _, ok = dlt_homography(src_sample, dst[sample_idx])
        if not ok or candidate is None:
            continue

        residuals = symmetric_transfer_error(candidate, src, dst)
        inliers = np.isfinite(residuals) & (residuals < threshold)
        count = int(inliers.sum())
        residual_sum = float(residuals[inliers].sum())
        if count > best_count or (count == best_count and residual_sum < best_sum):
            best_H = candidate
            best_mask = inliers
            best_residuals = residuals
            best_count = count
            best_sum = residual_sum

            # Update the iteration bound from the probability of drawing an
            # all-inlier four-point sample. Keep it at least as large as the
            # number of samples already consumed.
            inlier_fraction = count / n
            if inlier_fraction >= 1.0:
                required = 1
            elif inlier_fraction > 0.0 and confidence > 0.0:
                if confidence >= 1.0:
                    required = max_iterations
                else:
                    miss_probability = 1.0 - inlier_fraction ** 4
                    if miss_probability <= 0.0:
                        required = 1
                    elif miss_probability >= 1.0:
                        required = max_iterations
                    else:
                        required = int(np.ceil(
                            np.log1p(-confidence) / np.log(miss_probability)
                        ))
            else:
                required = max_iterations
            iteration_limit = min(
                iteration_limit,
                max(iterations, max(1, required)),
            )

    # A best model below the configured consensus size is diagnostic only.
    if best_H is None or best_count < min_inliers or best_count < 4:
        return None, best_mask, best_residuals, iterations, False

    refit, _, refit_ok = dlt_homography(src[best_mask], dst[best_mask])
    if not refit_ok or refit is None:
        return None, best_mask, best_residuals, iterations, False

    residuals = symmetric_transfer_error(refit, src, dst)
    inliers = np.isfinite(residuals) & (residuals < threshold)
    if int(inliers.sum()) < min_inliers:
        return None, inliers, residuals, iterations, False
    return refit, inliers, residuals, iterations, True


def warp_artwork(artwork: np.ndarray, H: np.ndarray, frame_shape: Tuple[int, ...]
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Destination-to-source warp of ``artwork`` into a frame of shape ``frame_shape``.

    ``H`` maps artwork pixel coordinates to frame pixel coordinates. For every
    frame pixel inside the projected artwork quadrilateral, map it back through
    H^{-1} and sample the artwork bilinearly. Returns (warped H x W x C float32,
    coverage H x W bool); pixels that map outside the artwork receive no
    coverage and are zero in ``warped``.
    """
    artwork = np.asarray(artwork)
    H = np.asarray(H, dtype=np.float64)
    frame_shape = tuple(frame_shape)
    if artwork.ndim not in (2, 3):
        raise ValueError("artwork must be a 2-D grayscale or 3-D colour image")
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    if H.shape != (3, 3):
        raise ValueError("H must be a 3 x 3 homography")

    artwork_h, artwork_w = artwork.shape[:2]
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    if artwork_h == 0 or artwork_w == 0 or frame_h < 0 or frame_w < 0:
        raise ValueError("artwork dimensions must be non-empty and frame dimensions non-negative")
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("H must be invertible") from exc

    output_shape = ((frame_h, frame_w) if artwork.ndim == 2
                    else (frame_h, frame_w, artwork.shape[2]))
    warped = np.zeros(output_shape, dtype=np.float32)
    coverage = np.zeros((frame_h, frame_w), dtype=bool)
    if frame_h == 0 or frame_w == 0:
        return warped, coverage

    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    frame_points = np.stack((xx.ravel(), yy.ravel(), np.ones(frame_h * frame_w)), axis=0)
    source_points = H_inv @ frame_points
    denominator = source_points[2]
    finite = np.isfinite(source_points).all(axis=0) & (np.abs(denominator) > 1e-12)

    source_x = np.full(denominator.shape, np.nan, dtype=np.float64)
    source_y = np.full(denominator.shape, np.nan, dtype=np.float64)
    source_x[finite] = source_points[0, finite] / denominator[finite]
    source_y[finite] = source_points[1, finite] / denominator[finite]

    # Allow tiny floating-point error at the projected rectangle boundary,
    # then clamp those boundary samples to the valid pixel-centre interval.
    tolerance = 1e-9
    valid = (finite &
             (source_x >= -tolerance) & (source_x <= artwork_w - 1 + tolerance) &
             (source_y >= -tolerance) & (source_y <= artwork_h - 1 + tolerance))
    coverage_flat = valid.reshape(frame_h, frame_w)
    coverage[:] = coverage_flat

    if np.any(valid):
        sx = np.clip(source_x[valid], 0.0, float(artwork_w - 1))
        sy = np.clip(source_y[valid], 0.0, float(artwork_h - 1))
        sampled = bilinear_sample(artwork.astype(np.float32, copy=False), sy, sx)
        warped[coverage] = sampled.astype(np.float32, copy=False)

    return warped, coverage


__all__ = ["bilinear_sample", "rectangle_to_rectangle", "normalise_points", "dlt_homography", "transfer_points",
           "symmetric_transfer_error", "ransac_homography", "warp_artwork"]
