"""Stage 1 Localise, keypoint branch.

Implement the four functions marked ``YOUR CODE HERE``. Do not change the
signatures or return types; the tests and the grading harness call them.
Use only numpy and the helpers in ``overlaylab.supplied`` (Sobel gradients,
Gaussian smoothing, bilinear sampling). OpenCV feature functions
(``cv2.cornerHarris``, ``cv2.SIFT`` and friends) are not allowed here; the
SIFT baseline lives only in the labelled ablation cell of Notebook 1.

Conventions: grayscale images are float32 in [0, 1]; ``x`` indexes columns and
``y`` rows; keypoints are N x 2 arrays of (x, y); descriptors are N x 128.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .config import DetectorConfig, DescriptorConfig, MatchConfig
from .supplied import sobel_gradients, gaussian_smooth
from .geometry import bilinear_sample


def harris_response(gray: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """Harris response R = det(M) - kappa * tr(M)^2 at every pixel.

    M is the Gaussian-weighted (scale ``config.sigma_w``) second-moment matrix
    built from the supplied normalised Sobel gradients. Returns a float32 map
    with the same shape as ``gray``.
    """
    gray = np.asarray(gray, dtype=np.float32)
    gx, gy, _, _ = sobel_gradients(gray)

    # Smooth the products of the image derivatives to form the local
    # second-moment matrix at each pixel.
    m_xx = gaussian_smooth(gx * gx, config.sigma_w)
    m_yy = gaussian_smooth(gy * gy, config.sigma_w)
    m_xy = gaussian_smooth(gx * gy, config.sigma_w)

    determinant = m_xx * m_yy - m_xy * m_xy
    trace = m_xx + m_yy
    response = determinant - config.kappa * trace * trace
    return response.astype(np.float32, copy=False)


def detect_corners(gray: np.ndarray, config: DetectorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Harris corners with non-maximum suppression.

    Steps: compute ``harris_response``; discard responses below
    ``config.response_fraction * max(R)``; apply NMS in mode ``config.nms_mode``
    with radius ``config.nms_radius`` ("window": keep a pixel only if it is the
    maximum of its (2r+1)x(2r+1) neighbourhood; "radius": sort survivors by R
    and greedily accept the strongest while discarding weaker candidates within
    distance r); drop corners within ``config.border`` pixels of the image edge.

    Returns (keypoints, responses): an N x 2 float array of (x, y) ordered by
    decreasing response (N <= config.max_corners) and the matching responses.
    """
    response = harris_response(gray, config)
    height, width = response.shape
    empty_points = np.empty((0, 2), dtype=np.float32)
    empty_responses = np.empty((0,), dtype=np.float32)

    if (height == 0 or width == 0 or config.max_corners <= 0 or
            config.border < 0 or 2 * config.border >= height or 2 * config.border >= width):
        return empty_points, empty_responses
    if config.nms_mode not in ("window", "radius"):
        raise ValueError(f"Unsupported NMS mode: {config.nms_mode!r}")

    peak = float(np.max(response))
    # Harris responses on an image with no corner structure may be all zero
    # or negative. In either case there is no positive corner candidate.
    if not np.isfinite(peak) or peak <= 0:
        return empty_points, empty_responses

    threshold = config.response_fraction * peak
    candidates = response >= threshold

    radius = max(0, int(config.nms_radius))
    if config.nms_mode == "window":
        # Compute the local maximum with shifted views rather than relying on
        # OpenCV feature/NMS routines. Padding prevents border wraparound.
        padded = np.pad(response, radius, mode="constant", constant_values=-np.inf)
        local_max = np.full(response.shape, -np.inf, dtype=np.float32)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                np.maximum(local_max,
                           padded[dy:dy + height, dx:dx + width],
                           out=local_max)
        candidates &= response == local_max

    border = int(config.border)
    valid = np.zeros(response.shape, dtype=bool)
    valid[border:height - border, border:width - border] = True
    candidates &= valid
    ys, xs = np.nonzero(candidates)
    if len(xs) == 0:
        return empty_points, empty_responses

    values = response[ys, xs]
    # np.nonzero is row-major, so stable sorting also gives deterministic
    # coordinate order for equal responses.
    order = np.argsort(-values, kind="stable")
    xs = xs[order]
    ys = ys[order]
    values = values[order]

    if config.nms_mode == "radius":
        accepted_x = []
        accepted_y = []
        accepted_values = []
        radius_sq = float(radius * radius)
        for x, y, value in zip(xs, ys, values):
            if accepted_x:
                dx = np.asarray(accepted_x, dtype=np.float32) - x
                dy = np.asarray(accepted_y, dtype=np.float32) - y
                if np.any(dx * dx + dy * dy <= radius_sq):
                    continue
            accepted_x.append(int(x))
            accepted_y.append(int(y))
            accepted_values.append(float(value))
            if len(accepted_x) >= config.max_corners:
                break
        points = np.column_stack((accepted_x, accepted_y)).astype(np.float32)
        responses = np.asarray(accepted_values, dtype=np.float32)
    else:
        xs = xs[:config.max_corners]
        ys = ys[:config.max_corners]
        values = values[:config.max_corners]
        points = np.column_stack((xs, ys)).astype(np.float32)
        responses = values.astype(np.float32, copy=False)

    return points, responses


def describe_keypoints(gray: np.ndarray, keypoints: np.ndarray,
                       config: DescriptorConfig) -> Tuple[np.ndarray, np.ndarray]:
    """128-D gradient-histogram descriptors with dominant-orientation normalisation.

    For each keypoint take a 16 x 16 grid of samples spaced ``config.patch_scale``
    pixels apart (so the window is 16 * patch_scale pixels wide),
    build a ``config.orientation_bins``-bin orientation histogram weighted by
    gradient magnitude, assign the dominant orientation, rotate the sampling
    grid so that the first histogram bin is aligned to it, and accumulate a
    ``cells x cells`` grid of ``cell_bins``-bin histograms (4 x 4 x 8 = 128).
    Apply Gaussian spatial weighting when ``config.gaussian_weighting`` is set,
    normalise to unit L2 norm, clamp to ``config.clamp``, and renormalise.

    Keypoints whose (rotated) window leaves the image are dropped. Returns
    (descriptors N' x 128, kept_indices N') where kept_indices indexes the input.
    """
    gray = np.asarray(gray, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    cells = int(config.cells)
    cell_bins = int(config.cell_bins)
    orientation_bins = int(config.orientation_bins)
    scale = float(config.patch_scale)
    if cells <= 0 or cell_bins <= 0 or orientation_bins <= 0 or scale <= 0:
        raise ValueError("descriptor grid dimensions and patch_scale must be positive")
    if 16 % cells != 0:
        raise ValueError("config.cells must evenly divide the 16 x 16 descriptor grid")

    gx, gy, _, _ = sobel_gradients(gray)
    height, width = gray.shape
    offsets = (np.arange(16, dtype=np.float64) - 7.5) * scale
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    cell_size = 16 // cells
    descriptor_size = cells * cells * cell_bins
    descriptors = []
    kept_indices = []
    angle_width = (2.0 * np.pi) / orientation_bins

    # A circularly symmetric Gaussian gives central samples more influence
    # without favoring any particular in-plane orientation.
    if config.gaussian_weighting:
        sigma = max(8.0 * scale, np.finfo(np.float64).eps)
        spatial_weight = np.exp(-0.5 * (grid_x * grid_x + grid_y * grid_y) / (sigma * sigma))
    else:
        spatial_weight = np.ones((16, 16), dtype=np.float64)

    for index, (x, y) in enumerate(keypoints):
        # Estimate a dominant orientation in the unrotated support window.
        base_x = x + grid_x
        base_y = y + grid_y
        gx_base = bilinear_sample(gx, base_y, base_x).astype(np.float64)
        gy_base = bilinear_sample(gy, base_y, base_x).astype(np.float64)
        mag_base = np.hypot(gx_base, gy_base)
        angle_base = np.arctan2(gy_base, gx_base)
        orient_index = np.floor(((angle_base + np.pi) % (2.0 * np.pi)) /
                                angle_width).astype(np.int64)
        orient_hist = np.bincount(
            orient_index.ravel(),
            weights=(mag_base * spatial_weight).ravel(),
            minlength=orientation_bins,
        )
        dominant_bin = int(np.argmax(orient_hist))
        dominant_angle = -np.pi + (dominant_bin + 0.5) * angle_width

        # Rotate the sample grid around the keypoint to align the descriptor
        # with the dominant orientation. Reject any support that leaves image.
        cos_a = np.cos(dominant_angle)
        sin_a = np.sin(dominant_angle)
        sample_x = x + cos_a * grid_x - sin_a * grid_y
        sample_y = y + sin_a * grid_x + cos_a * grid_y
        if (np.any(sample_x < 0) or np.any(sample_x > width - 1) or
                np.any(sample_y < 0) or np.any(sample_y > height - 1)):
            continue

        gx_sample = bilinear_sample(gx, sample_y, sample_x).astype(np.float64)
        gy_sample = bilinear_sample(gy, sample_y, sample_x).astype(np.float64)
        magnitude = np.hypot(gx_sample, gy_sample) * spatial_weight
        relative_angle = np.arctan2(gy_sample, gx_sample) - dominant_angle
        bin_index = np.floor(((relative_angle + np.pi) % (2.0 * np.pi)) *
                             (cell_bins / (2.0 * np.pi))).astype(np.int64)
        bin_index %= cell_bins

        histograms = np.zeros((cells, cells, cell_bins), dtype=np.float64)
        for row in range(cells):
            y0, y1 = row * cell_size, (row + 1) * cell_size
            for col in range(cells):
                x0, x1 = col * cell_size, (col + 1) * cell_size
                bins = bin_index[y0:y1, x0:x1].ravel()
                weights = magnitude[y0:y1, x0:x1].ravel()
                histograms[row, col] = np.bincount(
                    bins, weights=weights, minlength=cell_bins
                )[:cell_bins]

        descriptor = histograms.ravel()
        norm = np.linalg.norm(descriptor)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            continue
        descriptor /= norm
        descriptor = np.minimum(descriptor, config.clamp)
        norm = np.linalg.norm(descriptor)
        if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            continue
        descriptor /= norm
        descriptors.append(descriptor.astype(np.float32))
        kept_indices.append(index)

    if descriptors:
        descriptor_array = np.stack(descriptors).reshape(-1, descriptor_size)
    else:
        descriptor_array = np.empty((0, descriptor_size), dtype=np.float32)
    return descriptor_array, np.asarray(kept_indices, dtype=np.int64)


def match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray,
                      config: MatchConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour matching with Lowe's ratio test (optionally mutual).

    For every descriptor in A find the nearest and second-nearest descriptor in
    B by Euclidean distance; accept the match when d1 / d2 < ``config.ratio``.
    With ``config.mutual`` the match must also be the nearest neighbour of the B
    descriptor in A. Returns (pairs M x 2 of (index_a, index_b), distances M,
    ratios M).
    """
    desc_a = np.asarray(desc_a, dtype=np.float64)
    desc_b = np.asarray(desc_b, dtype=np.float64)
    if desc_a.ndim != 2 or desc_b.ndim != 2:
        raise ValueError("descriptor inputs must be two-dimensional arrays")
    if desc_a.shape[1] != desc_b.shape[1]:
        raise ValueError("descriptor dimensions must match")

    if len(desc_a) == 0 or len(desc_b) < 2:
        return (np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.float64))

    # Pairwise Euclidean distances, clipping round-off before square root.
    dist_sq = (np.sum(desc_a * desc_a, axis=1)[:, None] +
               np.sum(desc_b * desc_b, axis=1)[None, :] -
               2.0 * (desc_a @ desc_b.T))
    distances = np.sqrt(np.maximum(dist_sq, 0.0))
    nearest_order = np.argsort(distances, axis=1, kind="stable")
    best_b = nearest_order[:, 0]
    best_dist = distances[np.arange(len(desc_a)), best_b]
    second_dist = distances[np.arange(len(desc_a)), nearest_order[:, 1]]
    ratios = np.full(len(desc_a), np.inf, dtype=np.float64)
    np.divide(best_dist, second_dist, out=ratios, where=second_dist > 0)
    accepted = ratios < config.ratio

    if config.mutual and np.any(accepted):
        best_a_for_b = np.argmin(distances, axis=0)
        rows = np.arange(len(desc_a))
        accepted &= best_a_for_b[best_b] == rows

    rows = np.flatnonzero(accepted)
    pairs = np.column_stack((rows, best_b[rows])).astype(np.int64, copy=False)
    return pairs, best_dist[rows], ratios[rows]


__all__ = ["harris_response", "detect_corners", "describe_keypoints", "match_descriptors"]
