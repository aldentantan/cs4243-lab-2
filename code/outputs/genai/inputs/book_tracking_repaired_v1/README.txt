VACE INPUTS: DEDICATED BOOK TRACKER + REPAIRED BOARD GAPS

Use these two files together:
  interval_inpaint.mp4  -> --src_video
  src_mask.mp4          -> --src_mask
Continue supplying nus_ranking.png through --src_ref_images.

The files contain 150 frames at 1920x1080 and 30 fps (5 seconds).
Their original source indices are 120 through 269 inclusive.
Video frame 0 corresponds to source frame 120. PNG filenames retain source indices.

MASK CONVENTIONS
  src_mask.mp4 / editable_masks/: white = editable visible poster; black = retain.
  book_masks/: white = predicted book. These are diagnostic masks, not VACE inputs.
  board_masks/: white = board coverage, including the book overlap.
  editable = board & ~dilate(book), using a 5-source-pixel book boundary margin.
Only editable pixels were gray-filled. No NUS artwork was composited into the input.

METHOD
The book is detected in the full frame using navy colour, connected foreground
regions outside the board, colour variation learned outside the board, optical
flow for component association, and straight-edge refinement where supported.
Every frame is processed. No audit annotation, NUS artwork pixels, generated
output, or promptable segmentation model was used to generate these masks.
The tracker was not retuned after audit evaluation.

All 113 originally nonempty board masks are preserved pixel-for-pixel.
The 37 empty board masks were filled by linear interpolation of neighbouring
valid frozen-pipeline corner predictions. The longest gap spans 15 frames.
Each repair, its bounding frames, interpolation weight and corners are recorded
in preparation.json. Original masks and frozen pipeline settings were preserved.
Interpolation is an estimate; existing board tracking drift can leave poster
strips outside the editable region. Inspect preview.mp4.

VALIDATION
12 tracker/export/repair tests and 9 existing compositing tests passed.
Both VACE inputs have matching dimensions, frame counts and frame rates.
The encoded mask agrees with all 150 editable PNGs after binary thresholding.
There are no empty editable masks after repairing the board gaps.
Audit masks were loaded only in a separate evaluation invocation.

Current audit PNGs appear to mark the visible poster WHITE. The lab evaluator
normally expects the OCCLUDER white. Evaluation here explicitly uses --audit-mask-kind visible;
the original audit files have not been changed.

At source frame 257, under that explicit convention:
  Book pixels excluded from editing: 98.6395% (target >=98%)
  Visible poster pixels editable:    96.3686% (target >=95%)
For comparison, the previous frozen-difference preprocessing excluded 40.6205%
of book pixels and left 95.5322% of visible poster pixels editable at this frame.
The earlier reported 4.2% figure used the wrong audit-mask convention and is invalid.

The formal acceptance status is REQUIRES ANNOTATION REVIEW, not an unconditional pass.
At clean frame 183, the visible-mask outline and audit corners differ by a thin
843-pixel border band. Treating that band as occlusion fails the strict book-recall
check, despite no book appearing in the frame. See acceptance_report_visible.json.
Audit conventions should be reviewed before relying on existing lab occlusion scores.
Sparse audit frames do not certify every boundary, and these are input-mask
measurements, not measurements of a generated VACE output.

REPRODUCE FROM THE LOCAL code/ DIRECTORY
  python3 student_files/prepare_vace_inputs.py prepare --repair-missing-board --out outputs/genai/inputs/book_tracking_repaired_v2
  python3 student_files/prepare_vace_inputs.py evaluate --out outputs/genai/inputs/book_tracking_repaired_v2 --audit-mask-kind visible
The script refuses to overwrite an existing output directory.

CLUSTER INPUT FLAGS AFTER COPYING THIS FOLDER UNDER ~/vace_stage4/inputs/
  --src_video /home/e/e1122474/vace_stage4/inputs/book_tracking_repaired_v1/interval_inpaint.mp4
  --src_mask /home/e/e1122474/vace_stage4/inputs/book_tracking_repaired_v1/src_mask.mp4
  --src_ref_images /home/e/e1122474/vace_stage4/inputs/nus_ranking.png

VACE inference has not been run as part of this preprocessing implementation.

REPOSITORY PACKAGING NOTE (2026-11-02)
The helper and tracker now live under code/student_files/. Their algorithms and
these historical model inputs were not changed by the packaging repair.
Intermediate board_masks/, book_masks/ and editable_masks/ PNGs and most previews
are kept locally but ignored by Git; the two paired input MP4s, provenance JSONs
and review_000257.jpg remain included. Reproduce into a fresh output directory
before running the evaluate subcommand if those intermediate PNGs are needed.
