"""Stage 2 Track: optical flow, temporal variants, and failure monitoring.

The assessed implementation uses the supplied gradients and image samplers.
``cv2.calcOpticalFlowPyrLK`` and ``cv2.goodFeaturesToTrack`` may appear only in
the labelled comparison cell of Notebook 2. Grayscale images are float32 in
[0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import FlowConfig, MonitorConfig, TrackConfig, LocalisationConfig
from .supplied import to_gray, sobel_gradients, gaussian_pyramid, quad_mask, rect_corners
from .geometry import ransac_homography, transfer_points, bilinear_sample
from .localisation import localise_board

OK, REINIT, HIDE = "OK", "REINIT", "HIDE"


def lk_flow(prev_gray: np.ndarray, next_gray: np.ndarray, points: np.ndarray, config: FlowConfig
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterative single-level Lucas-Kanade at sparse points.

    For each point use a (2w+1) x (2w+1) window (``config.window``), the
    supplied Sobel gradients of ``prev_gray``, bilinear sampling of the warped
    window in ``next_gray`` at every iteration, and at most
    ``config.iterations`` updates (stop early when the update is below
    ``config.convergence``). The structure tensor is normalised by the number
    of window pixels so that ``config.min_eigenvalue`` is scale-free.

    Returns (displacements N x 2, eigenvalues N x 2 with lambda_1 >= lambda_2,
    residuals N (mean absolute brightness residual), valid N bool). A point is
    invalid when lambda_2 < config.min_eigenvalue, the residual exceeds
    config.max_residual, the update failed to converge, or the point leaves
    the image. Coarse-to-fine estimation over ``config.pyramid_levels`` is an
    optional extension.
    """
    prev = np.asarray(prev_gray, dtype=np.float32)
    nxt = np.asarray(next_gray, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if prev.ndim != 2 or nxt.ndim != 2 or prev.shape != nxt.shape:
        raise ValueError("prev_gray and next_gray must be same-size grayscale images")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain finite coordinates")

    n = len(pts)
    displacement = np.zeros((n, 2), dtype=np.float64)
    eigenvalues = np.zeros((n, 2), dtype=np.float64)
    residuals = np.full(n, np.inf, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    radius = max(0, int(config.window))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    ox, oy = np.meshgrid(offsets, offsets)
    gx, gy, _, _ = sobel_gradients(prev)
    height, width = prev.shape
    sample_count = ox.size

    for i, (x, y) in enumerate(pts):
        x0, y0 = x + ox, y + oy
        if (np.min(x0) < 0 or np.max(x0) > width - 1 or
                np.min(y0) < 0 or np.max(y0) > height - 1):
            continue
        ix = bilinear_sample(gx, y0, x0).astype(np.float64).ravel()
        iy = bilinear_sample(gy, y0, x0).astype(np.float64).ravel()
        tensor = np.array([[np.dot(ix, ix), np.dot(ix, iy)],
                           [np.dot(ix, iy), np.dot(iy, iy)]], dtype=np.float64) / sample_count
        try:
            eig = np.linalg.eigvalsh(tensor)[::-1]
        except np.linalg.LinAlgError:
            continue
        eigenvalues[i] = eig
        if not np.all(np.isfinite(eig)) or eig[1] < config.min_eigenvalue:
            continue

        inverse_tensor = np.linalg.inv(tensor)
        d = np.zeros(2, dtype=np.float64)
        converged = False
        for _ in range(max(0, int(config.iterations))):
            sx, sy = x0 + d[0], y0 + d[1]
            if (np.min(sx) < 0 or np.max(sx) > width - 1 or
                    np.min(sy) < 0 or np.max(sy) > height - 1):
                break
            target = bilinear_sample(nxt, sy, sx).astype(np.float64).ravel()
            source = bilinear_sample(prev, y0, x0).astype(np.float64).ravel()
            difference = target - source
            rhs = np.array([np.dot(ix, difference), np.dot(iy, difference)]) / sample_count
            update = -inverse_tensor @ rhs
            if not np.all(np.isfinite(update)):
                break
            d += update
            if np.linalg.norm(update) < config.convergence:
                converged = True
                break
        displacement[i] = d

        sx, sy = x0 + d[0], y0 + d[1]
        if (np.min(sx) < 0 or np.max(sx) > width - 1 or
                np.min(sy) < 0 or np.max(sy) > height - 1):
            continue
        target = bilinear_sample(nxt, sy, sx).astype(np.float64).ravel()
        source = bilinear_sample(prev, y0, x0).astype(np.float64).ravel()
        residuals[i] = float(np.mean(np.abs(target - source)))
        valid[i] = (converged and np.isfinite(residuals[i]) and
                    residuals[i] <= config.max_residual)

    return (displacement.astype(np.float32), eigenvalues.astype(np.float32),
            residuals.astype(np.float32), valid)


@dataclass
class TrackInit:
    """What ``track_board`` needs to (re)initialise from the reference image."""
    reference: np.ndarray
    localisation: LocalisationConfig


@dataclass
class TrackResult:
    homographies: List[Optional[np.ndarray]]
    corners: List[np.ndarray]                 # 4 x 2 per frame (NaN when lost)
    status: List[str]                          # "ok" or "lost"
    stats: List[Dict]                          # consumed by failure_monitor
    reinit_frames: List[int] = field(default_factory=list)
    flow: List[Dict] = field(default_factory=list)  # per-frame vectors, eigenvalues, validity


def _frame_stats(prev_corners, corners, gray_prev, gray, inlier_count, inlier_ratio,
                 median_residual, mean_flow, n_valid, status, n_matches=None):
    """Supplied helper: assemble the per-frame statistics dictionary (``n_matches`` on (re)initialised frames)."""
    disp = float(np.mean(np.linalg.norm(corners - prev_corners, axis=1))) if (
        prev_corners is not None and np.all(np.isfinite(corners)) and np.all(np.isfinite(prev_corners))) else float("nan")
    if gray_prev is not None and np.all(np.isfinite(corners)):
        m = quad_mask(gray.shape, corners)
        bright = float(np.mean(np.abs(gray[m] - gray_prev[m]))) if m.any() else float("nan")
    else:
        bright = float("nan")
    return {"inlier_count": int(inlier_count), "inlier_ratio": float(inlier_ratio),
            "median_residual": float(median_residual), "corner_displacement": disp,
            "mean_flow": float(mean_flow), "brightness_change": bright,
            "n_valid_points": int(n_valid), "status": status,
            "n_matches": (int(n_matches) if n_matches is not None else None)}


def track_board(frames: Sequence[np.ndarray], init: TrackInit, config: TrackConfig) -> TrackResult:
    """Track the board through a sequence with one of two variants.

    ``config.variant == "independent"``: run ``localise_board`` on every frame
    from the reference image.

    ``config.variant == "flow_assisted"``: localise frame 0, then propagate the
    inlier correspondences (or the corners, ``config.propagate``) to the next
    frame with ``lk_flow``, drop invalid points, re-estimate the homography
    with RANSAC between the reference points and the propagated points, and
    reinitialise from the reference image when fewer than
    ``config.reinit_min_points`` valid points remain, every
    ``config.reinit_every`` frames (0 disables), or when RANSAC fails.

    Optional corner propagation has four points only: use reinit_min_points <= 4,
    RANSAC min_inliers = 4 and monitor min_inlier_count <= 4 if selecting it.

    Both variants return a ``TrackResult`` with a homography, projected
    corners, a status flag and the statistics the failure monitor consumes
    for every frame (build them with the supplied ``_frame_stats``; use
    ``inlier_ratio`` = inliers / valid propagated points for the flow variant
    and ``median_residual`` = median over inliers of sqrt(e_i / 2) px). The
    flow-assisted variant also stores, for every frame after the first, a dict
    in ``TrackResult.flow[t]`` with keys ``points`` (tracked points N x 2),
    ``vectors`` (displacements N x 2), ``eigenvalues`` (N x 2), ``residuals``
    (N) and ``valid`` (N bool); frames that were (re)initialised store {}.
    """
    frames = list(frames)
    homographies: List[Optional[np.ndarray]] = []
    corners_out: List[np.ndarray] = []
    statuses: List[str] = []
    stats: List[Dict] = []
    reinit_frames: List[int] = []
    flow_records: List[Dict] = []
    if not frames:
        return TrackResult(homographies, corners_out, statuses, stats, reinit_frames, flow_records)

    variant = config.variant
    if variant not in ("independent", "flow_assisted"):
        raise ValueError(f"unsupported tracking variant: {variant!r}")
    if config.propagate not in ("inliers", "corners"):
        raise ValueError(f"unsupported propagation mode: {config.propagate!r}")

    reference_gray = to_gray(init.reference)
    ref_rect = rect_corners(reference_gray.shape[1], reference_gray.shape[0])
    prev_gray = None
    prev_corners = None
    src_tracks = np.empty((0, 2), dtype=np.float64)
    dst_tracks = np.empty((0, 2), dtype=np.float64)

    def register(frame, frame_index, previous_gray, previous_corners):
        H, corners, pairs, inliers, diag = localise_board(
            init.reference, frame, init.localisation
        )
        ok = bool(diag["status"] and H is not None)
        if not ok:
            H = None
            corners = np.full((4, 2), np.nan, dtype=np.float64)
        else:
            reinit_frames.append(frame_index)
        current_gray = to_gray(frame)
        stats_row = _frame_stats(
            previous_corners, corners, previous_gray, current_gray,
            diag["n_inliers"], diag["inlier_ratio"], diag["median_residual"],
            0.0, diag["n_inliers"], "ok" if ok else "lost",
            n_matches=diag["n_matches"],
        )
        return H, corners, "ok" if ok else "lost", stats_row, pairs, inliers, diag, current_gray

    for t, frame in enumerate(frames):
        current_gray = to_gray(frame)
        use_independent = variant == "independent"
        scheduled = (variant == "flow_assisted" and t > 0 and
                     config.reinit_every > 0 and t % config.reinit_every == 0)
        must_register = (t == 0 or use_independent or scheduled or
                         prev_corners is None or not np.all(np.isfinite(prev_corners)))
        flow_record: Dict = {}

        if not must_register:
            if config.propagate == "corners":
                flow_points = dst_tracks.copy()
                source_for_flow = ref_rect.copy()
            else:
                flow_points = dst_tracks.copy()
                source_for_flow = src_tracks.copy()
            vectors, eig, flow_resid, flow_valid = lk_flow(
                prev_gray, current_gray, flow_points, config.flow
            )
            propagated = flow_points + vectors
            flow_record = {"points": flow_points.astype(np.float32),
                           "vectors": vectors, "eigenvalues": eig,
                           "residuals": flow_resid, "valid": flow_valid}
            good_src = source_for_flow[flow_valid]
            good_dst = propagated[flow_valid]
            mean_flow = (float(np.mean(np.linalg.norm(vectors[flow_valid], axis=1)))
                         if np.any(flow_valid) else float("nan"))
            enough = len(good_src) >= max(4, int(config.reinit_min_points))
            ransac_cfg = init.localisation.ransac
            if config.propagate == "corners":
                ransac_cfg = dc_replace(ransac_cfg, min_inliers=4)
            if enough:
                H, inliers, residual, _, model_ok = ransac_homography(
                    good_src, good_dst, ransac_cfg
                )
            else:
                H, inliers, residual, model_ok = None, np.zeros(len(good_src), bool), np.full(len(good_src), np.inf), False

            if model_ok and H is not None:
                corners = transfer_points(H, ref_rect)
                model_ok = bool(np.all(np.isfinite(corners)))
            if model_ok:
                n_inliers = int(inliers.sum())
                inlier_ratio = n_inliers / max(len(good_src), 1)
                med_res = (float(np.median(np.sqrt(residual[inliers] / 2.0)))
                           if n_inliers else float("inf"))
                row = _frame_stats(
                    prev_corners, corners, prev_gray, current_gray,
                    n_inliers, inlier_ratio, med_res, mean_flow,
                    int(flow_valid.sum()), "ok",
                )
                if config.propagate == "corners":
                    src_tracks, dst_tracks = ref_rect.copy(), transfer_points(H, ref_rect)
                else:
                    src_tracks, dst_tracks = good_src[inliers], good_dst[inliers]
                status = "ok"
            else:
                must_register = True

        if must_register:
            H, corners, status, row, pairs, inliers, diag, current_gray = register(
                frame, t, prev_gray, prev_corners
            )
            flow_record = {}
            if status == "ok":
                if config.propagate == "corners":
                    src_tracks, dst_tracks = ref_rect.copy(), corners.copy()
                else:
                    src_tracks = diag["keypoints_ref"][pairs[inliers, 0]].astype(np.float64)
                    dst_tracks = diag["keypoints_frame"][pairs[inliers, 1]].astype(np.float64)
            else:
                src_tracks = np.empty((0, 2), dtype=np.float64)
                dst_tracks = np.empty((0, 2), dtype=np.float64)

        homographies.append(H)
        corners_out.append(np.asarray(corners, dtype=np.float64))
        statuses.append(status)
        stats.append(row)
        flow_records.append(flow_record)
        prev_gray = current_gray
        prev_corners = corners

    return TrackResult(homographies, corners_out, statuses, stats, reinit_frames, flow_records)


def failure_monitor(stats: Sequence[Dict], config: MonitorConfig) -> List[Tuple[str, str]]:
    """Decide OK / REINIT / HIDE for every frame from the per-frame statistics.

    ``stats`` is the list produced by ``track_board``: dictionaries with
    ``inlier_count``, ``inlier_ratio``, ``median_residual``,
    ``corner_displacement``, ``mean_flow``, ``brightness_change`` and
    ``status``. Returns a list of (flag, trigger) where ``trigger`` names the
    statistic (or "none") responsible for the decision. Because a homography
    is defined only up to scale, compare projected-corner displacement and
    reprojection error, never homography matrices.

    Required policy (the tests check it; the thresholds are yours to set):
    1. ``status != "ok"`` -> HIDE with trigger "no_model";
    2. ``inlier_count`` < min_inlier_count or ``inlier_ratio`` < min_inlier_ratio
       -> REINIT with trigger "inlier_count" / "inlier_ratio", escalating to
       HIDE once this has happened on ``hide_after_consecutive`` consecutive
       frames (frames under rule 1 count towards the run);
    3. otherwise the first of ``median_residual``, ``corner_displacement``,
       ``mean_flow``, ``brightness_change`` that exceeds its maximum -> REINIT
       with that statistic as trigger (NaN statistics never fire);
    4. otherwise OK with trigger "none"; an OK frame resets the run.
    """
    flags: List[Tuple[str, str]] = []
    bad_run = 0
    hide_after = max(1, int(config.hide_after_consecutive))
    quality_limits = (
        ("median_residual", config.max_median_residual),
        ("corner_displacement", config.max_corner_displacement),
        ("mean_flow", config.max_mean_flow),
        ("brightness_change", config.max_brightness_change),
    )
    for row in stats:
        status = row.get("status", "lost")
        if status != "ok":
            bad_run += 1
            flags.append((HIDE, "no_model"))
            continue

        count = row.get("inlier_count", 0)
        ratio = row.get("inlier_ratio", float("nan"))
        low_reason = None
        try:
            if np.isfinite(count) and count < config.min_inlier_count:
                low_reason = "inlier_count"
            elif np.isfinite(ratio) and ratio < config.min_inlier_ratio:
                low_reason = "inlier_ratio"
        except TypeError:
            low_reason = None
        if low_reason is not None:
            bad_run += 1
            flags.append((HIDE if bad_run >= hide_after else REINIT, low_reason))
            continue

        trigger = None
        for key, maximum in quality_limits:
            value = row.get(key, float("nan"))
            try:
                if np.isfinite(value) and value > maximum:
                    trigger = key
                    break
            except TypeError:
                continue
        if trigger is not None:
            flags.append((REINIT, trigger))
        else:
            bad_run = 0
            flags.append((OK, "none"))
    return flags


__all__ = ["OK", "REINIT", "HIDE", "lk_flow", "TrackInit", "TrackResult", "track_board", "failure_monitor"]
