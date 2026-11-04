"""Summarise saved personal tracking for the Scorecard; no inference or tuning. [AI-CODE]"""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np

from overlaylab import data, metrics


def build_summary(root):
    root = Path(root)
    conditions = json.loads((root / "manifests/genai_conditions.json").read_text())
    clip_id = conditions["clip_id"]
    interval = conditions["model"]["interval"]
    first, last = interval["start"], interval["end"]
    gt = data.load_audit_corners(clip_id)
    gt_frames = sorted(gt)
    gt_quads = [gt[f] for f in gt_frames]
    runs, visible, summaries = {}, {}, {}
    provenance = {}
    for variant in ("independent", "flow_assisted"):
        path = root / "outputs/videos" / f"{clip_id}_{variant}_stats.json"
        run = json.loads(path.read_text())
        runs[variant] = run
        provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        visible[variant] = {
            f: np.asarray(q) for f, q in enumerate(run["corners"])
            if run["status"][f] == "ok" and run["flags"][f][0] != "HIDE"
            and np.isfinite(q).all()
        }
        errors = metrics.corner_errors(visible[variant], gt)
        late = [
            {"frame": f, "age": age, "error": error}
            for f, (age, error) in zip(sorted(errors), metrics.drift(errors, run["reinit_frames"]))
            if age >= 16 and np.isfinite(error)
        ]
        registrations = set(run["reinit_frames"])
        longest = length = 0
        longest_range = None
        for f in range(run["n_frames"]):
            if f not in visible[variant]:
                length = 0
                continue
            if f in registrations or length == 0:
                length, start = 1, f
            else:
                length += 1
            if length > longest:
                longest, longest_range = length, [start, f]
        summaries[variant] = {
            "successful_registrations_including_frame_0": len(registrations),
            "subsequent_successful_registrations": sum(f > 0 for f in registrations),
            "longest_visible_run_frames": longest,
            "longest_visible_run_range": longest_range,
            "drift_error": float(np.median([x["error"] for x in late])) if late else None,
            "drift_samples": late,
        }

    common = sorted(set(visible["independent"]) & set(visible["flow_assisted"]) & set(range(first, last + 1)))
    pairs = [(f - 1, f) for f in common if f - 1 in common]
    fig, ax = plt.subplots(figsize=(10, 4))
    for variant in runs:
        predictions = {f: visible[variant][f] for f in common}
        jitter = metrics.jitter(predictions, gt_frames, gt_quads)
        per_pair = [metrics.jitter({f: predictions[f] for f in pair}, gt_frames, gt_quads) for pair in pairs]
        summaries[variant].update(
            comparison_cut_jitter_px=jitter,
            comparison_cut_successful_registrations=sum(first <= f <= last for f in runs[variant]["reinit_frames"]),
            comparison_pair_jitter_px=per_pair,
        )
        by_frame = {b: value for (_, b), value in zip(pairs, per_pair)}
        line, = ax.plot(range(first + 1, last + 1),
                        [by_frame.get(f, np.nan) for f in range(first + 1, last + 1)], ".-", linewidth=.8,
                        label=f"{variant.replace('_', '-')} (median {jitter:.2f} px)")
        ax.axhline(jitter, color=line.get_color(), linestyle="--", alpha=.7)
    ax.set(title=f"{clip_id}: frozen saved poses, frames {first}–{last}; {len(pairs)} shared visible adjacent pairs",
           xlabel="Source frame (second frame of pair)", ylabel="Motion-compensated corner displacement (px)")
    ax.legend()
    fig.tight_layout()
    summary = {
        "clip_id": clip_id,
        "provenance_sha256": provenance,
        "drift_definition": "Median finite metrics.corner_error on audit frames at age >=16 from metrics.drift; exclude HIDE/lost poses.",
        "run_definition": "Consecutive finite, status-ok, non-HIDE frames; reset on successful registration or loss/hiding; include registration frame.",
        "comparison_definition": "Existing generation cut; both variants use identical adjacent pairs visible in both. metrics.jitter uses saved tracker corners and the complete manual audit trajectory, not output-artwork re-localisation.",
        "comparison_cut": [first, last],
        "comparison_pairs": pairs,
        "variants": summaries,
    }
    out = root / "student_files/personal_tracking_summary"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fig.savefig(out / "stabilisation.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    build_summary(Path(__file__).resolve().parents[1])
