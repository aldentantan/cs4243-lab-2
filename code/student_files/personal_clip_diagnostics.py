"""Read-only analysis of saved personal-clip predictions; no tracking or tuning. [AI-CODE]"""

from pathlib import Path
import hashlib
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from overlaylab import data, metrics, supplied as sp
from overlaylab.compositing import occlusion_mask
from overlaylab.config import PipelineConfig
from overlaylab.geometry import dlt_homography, transfer_points, warp_artwork
from overlaylab.temporal import failure_monitor


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _thumbnail(image, nearest=False):
    height, width = image.shape[:2]
    return cv2.resize(image, (720, round(height * 720 / width)),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def create_personal_diagnostics(root, clip_id, representative):
    """Save six-panel figures from existing videos/stats and reconstructed differences.

    Uses saved flow-assisted corners to reconstruct reference-to-frame H with
    the student's normalised DLT. Audit corners are drawn and scored only;
    they never construct H. Saved predicted occlusion masks remain authoritative.
    Every reconstructed mask must match the saved mask exactly before use.
    The absent homography on a lost frame remains absent: no GT fallback.
    """
    root = Path(root)
    out_dir = root / "student_files" / "personal_clip_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = root / "outputs" / "videos" / f"{clip_id}_flow_assisted_stats.json"
    stats = json.loads(stats_path.read_text())
    independent_path = stats_path.with_name(f"{clip_id}_independent_stats.json")
    independent = json.loads(independent_path.read_text())
    cfg = PipelineConfig.from_json(json.dumps(stats["config"]))
    assert [list(flag) for flag in failure_monitor(stats["stats"], cfg.track.monitor)] == stats["flags"]
    source_path, reference_path = data.clip_paths(clip_id)
    output_path = root / "outputs" / "videos" / f"{clip_id}_flow_assisted.mp4"
    reference = sp.read_image(reference_path)
    gt_corners = data.load_audit_corners(clip_id)
    gt_masks = data.load_audit_masks(clip_id)
    info = sp.video_info(source_path)
    if stats.get("resize_width") not in (None, info["width"]):
        raise ValueError("These diagnostics require saved full-resolution predictions.")
    errors = {frame: metrics.corner_error(np.asarray(stats["corners"][frame]), quad)
              for frame, quad in gt_corners.items()}
    finite_frames = [frame for frame, error in errors.items() if np.isfinite(error)]
    missing_frames = sorted(frame for frame, error in errors.items() if not np.isfinite(error))
    worst_finite = max(finite_frames, key=lambda frame: errors[frame])
    cases = [(representative, "Representative clean overlay")]
    if missing_frames:
        cases.append((missing_frames[0], "Failure: no saved localisation model"))
    if worst_finite != representative:
        cases.append((worst_finite, "Failure: largest finite audit corner error"))
    wanted = {frame for frame, _ in cases}
    originals = {frame: image for frame, image in sp.iter_video(source_path, stop=max(wanted) + 1)
                 if frame in wanted}
    overlays = {frame: image for frame, image in sp.iter_video(output_path, stop=max(wanted) + 1)
                if frame in wanted}
    assert set(originals) == set(overlays) == wanted
    protected = [source_path, reference_path, output_path, stats_path, independent_path,
                 data.MANIFEST_DIR / "audit_corners.csv", data.MANIFEST_DIR / "audit_masks.csv"]
    protected += [data.PERSONAL_DIR / clip_id / "masks" / f"mask_{frame:06d}.png"
                  for frame in gt_masks]
    protected += list((root / "outputs" / "videos" / stats["mask_dir"]).glob("mask_*.png"))
    hashes = {str(path): _digest(path) for path in protected}
    evidence = {
        "clip_id": clip_id, "variant": "flow_assisted", "source_fps": info["fps"],
        "helper_sha256": _digest(__file__),
        "selection": "Declared clean frame; first missing-model audit frame; largest finite saved-corner audit error.",
        "difference_provenance": "Recomputed with frozen saved config and H reconstructed by normalised DLT from saved predicted corners; no audit-corner correction.",
        "mask_provenance": "Saved predicted PNGs; reconstructed masks checked for exact agreement.",
        "saved_artifact_sha256": hashes, "frames": [],
    }
    for frame, role in cases:
        original, overlay = originals[frame], overlays[frame]
        predicted = np.asarray(stats["corners"][frame], dtype=float)
        row = stats["stats"][frame]
        flag, trigger = stats["flags"][frame]
        rec = {
            "frame": frame, "role": role, "time_s": frame / info["fps"],
            "predicted_corners": predicted, "audit_corners": gt_corners[frame],
            "saved_corner_error": errors[frame], "saved_statistics": row,
            "monitor_flag": flag, "monitor_trigger": trigger,
            "independent_statistics": independent["stats"][frame],
            "independent_monitor": independent["flags"][frame],
            "monitor_thresholds": stats["config"]["track"]["monitor"],
        }
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax in axes.flat:
            ax.set_axis_off()
        axes[0, 0].imshow(_thumbnail(original)); axes[0, 0].set_title("A. Source frame")
        axes[0, 1].imshow(_thumbnail(overlay)); axes[0, 1].set_title("B. Saved classical output (decoded MP4)")
        annotated = sp.to_uint8(original).copy()
        cv2.polylines(annotated, [np.rint(gt_corners[frame]).astype(np.int32)], True, (0, 255, 0), 5)
        if np.all(np.isfinite(predicted)):
            cv2.polylines(annotated, [np.rint(predicted).astype(np.int32)], True, (255, 0, 255), 5)
        axes[0, 2].imshow(_thumbnail(annotated))
        axes[0, 2].set_title("C. Corners: audit green / prediction magenta")
        monitor_text = [f"Saved flag: {flag} / {trigger}", f"Tracking status: {stats['status'][frame]}",
                        f"Inliers: {row['inlier_count']} (min {cfg.track.monitor.min_inlier_count})",
                        f"Ratio: {row['inlier_ratio']:.3f} (min {cfg.track.monitor.min_inlier_ratio:.2f})",
                        f"Residual: {row['median_residual']:.3f} px (max {cfg.track.monitor.max_median_residual:g})",
                        f"Displacement: {row['corner_displacement']:.3f} px (max {cfg.track.monitor.max_corner_displacement:g})",
                        f"Mean flow: {row['mean_flow']:.3f} px (max {cfg.track.monitor.max_mean_flow:g})",
                        f"Brightness change: {row['brightness_change']:.4f} (max {cfg.track.monitor.max_brightness_change:g})"]
        if np.all(np.isfinite(predicted)):
            H, _, valid = dlt_homography(sp.rect_corners(reference.shape[1], reference.shape[0]), predicted)
            assert valid
            rec["reconstructed_H"] = H
            rec["corner_reconstruction_max_error_px"] = float(np.linalg.norm(
                transfer_points(H, sp.rect_corners(reference.shape[1], reference.shape[0])) - predicted, axis=1).max())
            reconstructed, difference, _ = occlusion_mask(original, H, reference, cfg.occlusion)
            _, coverage = warp_artwork(reference, H, original.shape)
            saved_path = root / "outputs" / "videos" / stats["mask_dir"] / f"mask_{frame:06d}.png"
            saved_mask = sp.read_mask(saved_path)
            if not np.array_equal(reconstructed, saved_mask):
                raise RuntimeError(f"Frame {frame}: reconstructed mask differs from saved mask; investigate before claiming reproduction.")
            rec["reconstructed_mask_mismatch_pixels"] = 0
            diff_small = _thumbnail(difference)
            cov_small = _thumbnail(coverage.astype(np.uint8), nearest=True).astype(bool)
            image = axes[1, 0].imshow(np.ma.masked_where(~cov_small, diff_small), cmap="magma", vmin=0, vmax=.5)
            axes[1, 0].set_title("D. Reconstructed compensated difference (0–0.5)")
            fig.colorbar(image, ax=axes[1, 0], fraction=.025, pad=.015)
            axes[1, 1].imshow(_thumbnail(saved_mask.astype(np.uint8), nearest=True), cmap="gray", vmin=0, vmax=1)
            axes[1, 1].set_title("E. Saved predicted occluder mask (white)")
            if frame in gt_masks:
                gt = gt_masks[frame]
                contours, _ = cv2.findContours(_thumbnail(gt.astype(np.uint8), nearest=True), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    axes[1, 1].plot(contour[:, 0, 0], contour[:, 0, 1], color="lime", linewidth=1)
                rec.update({"mask_iou": metrics.mask_iou(saved_mask, gt), "mask_f1": metrics.mask_f1(saved_mask, gt),
                            "predicted_occluder_pixels": int(saved_mask.sum()), "audit_occluder_pixels": int(gt.sum()),
                            "false_positive_mask_pixels": int((saved_mask & ~gt).sum()),
                            "missed_occluder_pixels": int((gt & ~saved_mask).sum())})
                missed = gt & ~saved_mask
                rec["missed_outside_predicted_coverage"] = int((missed & ~coverage).sum())
                rec["missed_below_weak_inside_coverage"] = int((missed & coverage & (difference < cfg.occlusion.weak)).sum())
                rec["missed_in_weak_band_inside_coverage"] = int((missed & coverage & (difference >= cfg.occlusion.weak) & (difference < cfg.occlusion.strong)).sum())
                rec["missed_above_strong_inside_coverage"] = int((missed & coverage & (difference >= cfg.occlusion.strong)).sum())
                monitor_text += [f"Mask IoU / F1: {rec['mask_iou']:.3f} / {rec['mask_f1']:.3f}",
                                 f"Missed occluder pixels: {rec['missed_occluder_pixels']:,}",
                                 f"False-positive pixels: {rec['false_positive_mask_pixels']:,}",
                                 "Mask green outline = manual occluder"]
            monitor_text += [f"Saved-corner audit error: {errors[frame]:.2%}",
                             f"Audit success threshold: {cfg.localisation.success_threshold:.1%}",
                             "Reconstructed mask = saved mask exactly"]
        else:
            rec.update({"reconstructed_H": None, "difference_available": False,
                        "mask_available": False, "reason": "No finite saved model; pipeline skips warp and mask, then copies source."})
            for ax, label in [(axes[1, 0], "D. Compensated difference"), (axes[1, 1], "E. Predicted occlusion mask")]:
                ax.set_title(label)
                ax.text(.5, .5, "Unavailable: no saved homography\nPipeline hides the overlay\nNo ground-truth fallback", ha="center", va="center", transform=ax.transAxes)
            rec["mean_absolute_saved_output_difference"] = float(np.mean(np.abs(overlay - original)))
            monitor_text += ["No corner model; audit placement fails", "No mask or difference was fabricated"]
        axes[1, 2].set_title("F. Saved monitor evidence and audit measurements")
        axes[1, 2].text(0, 1, "\n".join(monitor_text), va="top", fontsize=10, family="monospace", transform=axes[1, 2].transAxes)
        fig.suptitle(f"{clip_id} / flow-assisted / frame {frame}: {role}", fontsize=15)
        fig.text(.02, .015, "Saved source/output, corners, mask and monitor. Difference map reconstructed from saved predictions with frozen settings. [AI-CODE]", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        figure_path = out_dir / f"{clip_id}_frame_{frame:06d}.png"
        fig.savefig(figure_path, dpi=130)
        plt.close(fig)
        rec["figure"] = str(figure_path.relative_to(root))
        evidence["frames"].append(rec)
    assert all(_digest(path) == expected for path, expected in hashes.items()), "Saved evidence must remain unchanged"
    evidence = _json_safe(evidence)
    (out_dir / f"{clip_id}_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return evidence
