"""Browse source frames and mark candidates for manual occlusion annotation.

From the repository root:
    .venv/bin/python code/student_files/browse_clip.py
    .venv/bin/python code/student_files/browse_clip.py --start 180

A/D: previous/next frame; J/L: back/forward 10; M: mark/unmark; Q/Esc: quit.
The slider seeks to a zero-based frame index. Selected indices and annotation
commands are printed on exit. This tool does not write annotations or run
inference; deciding whether the poster is occluded remains a manual task.
[AI-CODE]
"""
from __future__ import annotations

import argparse
import math
import shlex
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WINDOW = "Occlusion frame browser"


class VideoReader:
    """Read exact frame indices without retaining the entire video in memory."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise ValueError(f"Video does not exist: {path}")
        self.cap = cv2.VideoCapture(str(path))
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if (not self.cap.isOpened() or not math.isfinite(count) or count < 1
                or not math.isfinite(self.fps) or self.fps <= 0):
            self.close()
            raise ValueError(f"Cannot read video frame count/fps: {path}")
        self.count = int(count)
        self.next_index = 0

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < self.count:
            raise ValueError(f"Frame {index} is outside 0..{self.count - 1}")
        if index != self.next_index:
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"Could not seek to frame {index}")
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"Could not decode frame {index}")
        position = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if abs(position - (index + 1)) > 0.5:
            raise ValueError(f"Decoder position {position} does not match frame {index}")
        self.next_index = index + 1
        return frame

    def close(self) -> None:
        self.cap.release()


def draw_frame(frame: np.ndarray, index: int, reader: VideoReader,
               selected: set[int]) -> np.ndarray:
    """Fit the complete source image between the status and control strips."""
    h, w = frame.shape[:2]
    scale = min(1280 / w, 640 / h, 1.0)
    image = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas_width = max(1000, image.shape[1])
    canvas = np.full((image.shape[0] + 130, canvas_width, 3), 25, np.uint8)
    left = (canvas_width - image.shape[1]) // 2
    canvas[48:48 + image.shape[0], left:left + image.shape[1]] = image
    marked = index in selected
    status = "MARKED" if marked else "unmarked"
    colour = (80, 240, 80) if marked else (235, 235, 235)
    cv2.putText(canvas,
                f"Frame {index} / {reader.count - 1}  |  {index / reader.fps:.3f}s  |  {status}",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, colour, 1, cv2.LINE_AA)
    y = image.shape[0] + 73
    lines = [
        "A/D: previous/next | J/L: -10/+10 | M: mark/unmark | Q/Esc: quit",
        f"Marked: {len(selected)} (need at least 3 genuinely occluded mask frames)",
        "Frames: " + (" ".join(map(str, sorted(selected))) or "none"),
    ]
    for line in lines:
        if len(line) > 145:
            line = line[:142] + "..."
        cv2.putText(canvas, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (235, 235, 235), 1, cv2.LINE_AA)
        y += 23
    return canvas


def print_selection(selected: set[int], clip_id: str) -> None:
    indices = " ".join(map(str, sorted(selected)))
    print(f"\nSelected candidate occlusion frames ({len(selected)}): {indices or 'none'}", flush=True)
    if not selected:
        return
    if len(selected) < 3:
        print("Select at least 3 genuinely occluded frames for your mask audit.")
    clip = shlex.quote(clip_id)
    print("\nRun from code/ to annotate these corners first:")
    print(f"../.venv/bin/python -m overlaylab.annotate corners {clip} "
          f"--frames {indices} --annotator Alden")
    print("\nThen annotate their occlusion masks:")
    print(f"../.venv/bin/python -m overlaylab.annotate masks {clip} "
          f'--frames {indices} --occluder "blue book" --annotator Alden')
    print("\nThese are candidate frames. Include them in your final spread-out corner set;")
    print("the final mask audit needs 5 frames, at least 3 genuinely occluded.")


def browse(path: Path, clip_id: str, start: int = 0) -> None:
    reader = VideoReader(path)
    selected: set[int] = set()
    window_created = False
    try:
        if not 0 <= start < reader.count:
            raise ValueError(f"--start must be between 0 and {reader.count - 1}")
        requested = start

        def seek(index: int) -> None:
            nonlocal requested
            requested = index

        print(f"Browsing {path}\n{reader.count} frames at {reader.fps:g} fps", flush=True)
        print("A/D: previous/next; J/L: -10/+10; M: mark/unmark; Q/Esc: quit.", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_created = True
        cv2.createTrackbar("Frame", WINDOW, 0, max(1, reader.count - 1), seek)
        cv2.setTrackbarPos("Frame", WINDOW, start)
        current = -1
        frame = None
        dirty = True
        while True:
            target = max(0, min(reader.count - 1, requested))
            if current != target:
                frame = reader.read(target)
                current = target
                dirty = True
            if dirty:
                cv2.imshow(WINDOW, draw_frame(frame, current, reader, selected))
                dirty = False
            key = cv2.waitKeyEx(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                if current in selected:
                    selected.remove(current)
                else:
                    selected.add(current)
                print(f"Frame {current}: {'marked' if current in selected else 'unmarked'}; "
                      f"selected = {sorted(selected)}", flush=True)
                dirty = True
            else:
                steps = {ord("a"): -1, ord("d"): 1, ord("j"): -10, ord("l"): 10,
                         ord("A"): -1, ord("D"): 1, ord("J"): -10, ord("L"): 10}
                if key in steps:
                    requested = max(0, min(reader.count - 1, current + steps[key]))
                    cv2.setTrackbarPos("Frame", WINDOW, requested)
    finally:
        reader.close()
        if window_created:
            cv2.destroyAllWindows()
        print_selection(selected, clip_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip-id", default="personal_clip", help="folder under data/personal (default: personal_clip)")
    parser.add_argument("--start", type=int, default=0, help="initial zero-based frame index")
    args = parser.parse_args()
    if not args.clip_id or args.clip_id in (".", "..") or Path(args.clip_id).name != args.clip_id:
        parser.error("--clip-id must be a folder name under data/personal")
    path = REPO_ROOT / "data" / "personal" / args.clip_id / "clip.mp4"
    try:
        browse(path, args.clip_id, args.start)
    except KeyboardInterrupt:
        pass
    except (ValueError, cv2.error) as error:
        parser.exit(1, f"Browser error: {error}\nRun locally with an OpenCV GUI/display available.\n")


if __name__ == "__main__":
    main()
