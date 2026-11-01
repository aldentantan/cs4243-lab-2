# Overlay Scorecard — clip

**Clip**: clip; 1920x1080 at 30 fps, 13.74 s. A handheld view of the wall-mounted textured poster with changes in distance and viewing angle; a blue book crosses the board.
**Initialisation**: Frame 0: 320 matches, 283 inliers (88.4%), median residual 0.752 px.
**Geometry**: Independent: median 1.1%, p90 1.7%, success 5/12 (41.7%), detected 100.0% (12/12); success threshold 1.0%. Flow-assisted: median 1.4%, p90 4.4%, success 2/12 (16.7%), detected 100.0% (12/12); success threshold 1.0%.
**Fidelity**: Independent: NCC 0.740, SSIM 0.477, structure 0.409, MAE 0.102. Flow-assisted: NCC 0.673, SSIM 0.425, structure 0.333, MAE 0.123.
**Temporal and preservation**: Independent: jitter 8.54 px, flicker 0.0805, warp error 0.0604, background drift 0.29 px. Flow-assisted: jitter 3.68 px, flicker 0.0909, warp error 0.0704, background drift 0.40 px. Outside-overlay PSNR/SSIM: independent 35.4 dB/0.974; flow-assisted 35.6 dB/0.974.
**Occlusion**: Across 5 audit mask frames, leakage was 61.6% for independent and 53.5% for flow-assisted; visible-board coverage was 94.3% and 89.0%, respectively. The high leakage indicates that the overlay often remained on the masked blue-book region.
**Description**: Independent gave better audit-frame placement and fidelity in this run; flow-assisted reduced recovered-corner jitter. Independent had lower flicker and warp error, while background preservation was similar. Both variants struggled to keep the overlay off the book.
**Uncertainty**: Evaluation uses 12 annotated corner frames and 5 mask frames, so it samples only a small part of the clip and occlusion event. Corner and mask annotation accuracy limits the interpretation of placement and visible-board coverage.
