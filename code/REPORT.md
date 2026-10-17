# CS4243 Lab 2 report

This compact report records how every `YOUR CODE HERE` section is exercised. Elaborate on your implementation and the results you observed; add figures, plots and grounded reasoning that show your understanding. Every reported number must name its clip (common / viewpoint / scale / blur / exit / combined / your clip), frame subset (validation / test / audit) and method variant, and must come from `overlaylab.metrics` or `overlaylab.evaluate`. Thresholds are frozen on the common clip's validation frames; each Stage's **stress table** (`python -m overlaylab.report`, `outputs/stress_tables.md`) reports the common clip (validation and test) and the Stage's two stress clips with that frozen configuration. Paste the generated table into the Stage's section unchanged and explain the row the Stage's question points at; the other rows need no interpretation.

## Stage 1 — Localise
<!-- harris_response, detect_corners, describe_keypoints, match_descriptors; bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork; localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames; frozen success threshold -->

### Implemented functions

In `keypoints.py`, `harris_response` forms the Gaussian-smoothed second-moment matrix from the supplied image gradients and computes the Harris score. `detect_corners` thresholds that response, applies the selected window or greedy radius NMS, removes border points, and returns corners sorted by response. `describe_keypoints` estimates a dominant orientation and builds a normalized spatial gradient histogram around each keypoint; `match_descriptors` uses nearest and second-nearest Euclidean distances for Lowe's ratio test, with optional mutual matching.

In `geometry.py`, `bilinear_sample` samples gray or color images at floating-point coordinates. `normalise_points` applies Hartley centering and scaling; `dlt_homography` uses those normalized points to solve the homography by SVD and rejects rank-deficient configurations. `symmetric_transfer_error` scores forward and inverse reprojection; `ransac_homography` rejects degenerate samples, selects a consensus set, adapts its iteration bound, and refits on its inliers. `transfer_points` applies a homography to image points, and `warp_artwork` inverse-maps frame pixels into the artwork and returns both the sampled image and its coverage mask.

Notebook 1 exercises these implementations in its Harris/NMS, descriptor/ratio-test, normalized-DLT, RANSAC, and warp-convention sections. The descriptions above cover the keypoint and geometry functions implemented so far; the higher-level `localisation.py` functions and later Stage 1 evidence are not included here.

### Stress table (Stage 1)
<!-- generated rows: common · validation, common · test, viewpoint, scale — median / p90 corner error, success at the frozen threshold, inliers, inlier ratio, affine median error / success -->

TODO: paste the generated Stage 1 table from `outputs/stress_tables.md` unchanged after the final pipeline runs. That generated file is not present yet.

### Question 1 — Matching

Unless identified as SIFT, these are Notebook 1 matching rates for our fixed-patch oriented-gradient descriptor, selected radius-NMS detector, and configured ratio matcher; the rate is accepted matches divided by reference descriptors. Notebook 1's labelled rotation check on the **common clip, validation frame 176** shows the rate changing from **0.454** to **0.439** after a synthetic 30-degree in-plane rotation. On that same common-clip validation frame, downscaling the frame by 0.6 reduces the rate to **0.036**. The dominant-orientation step therefore absorbs much of an in-plane rotation: it rotates the descriptor's sampling grid to the keypoint's dominant gradient direction. The descriptor uses a fixed-size support window, however, so a scale change changes which image structures fall into its spatial cells. It also does not undo the spatial distortion caused by an oblique viewpoint.

The strongest adjacent-frame collapse in Notebook 1's annotated **viewpoint stress clip (`coke_perspective`)** is from frame **448 to 464**: matching rate **0.214 to 0.036**. Over those frames, the annotated quad's top-edge angle changes by **-13.5 degrees**, apparent scale changes from **235.7 to 222.6 pixels**, and the top-versus-bottom edge keystone measure increases from **0.013 to 0.034**. In the annotated **scale stress clip (`coke_scale`)**, frame **80 to 96** drops from **0.500 to 0.132**; the angle changes by **-1.9 degrees**, apparent scale from **190.5 to 172.6 pixels**, while keystone changes only from **0.014 to 0.003**. These are diagnostic annotated stress frames, not the common-clip validation/test split. The scale pair points to the fixed patch scale as the failing descriptor component; the viewpoint pair adds perspective distortion that orientation alignment cannot remove.

Because the reference has a fixed pixel size, the scale row of the Stage 1 stress table should be read as the effect of the board changing size relative to that unchanged reference. The Notebook 1 stress-pair results are consistent with a descriptor that is rotation-normalized but not scale-invariant. The table itself is still pending, so its pipeline success and corner-error values are not claimed here.

The labelled SIFT ablation compares the reference to **common-clip validation frame 448**: the notebook reports **0.086** for this descriptor and **0.179** for SIFT. That is evidence from the common clip, not a SIFT comparison on either stress pair. SIFT's scale-space keypoint detection gives it scale handling that this fixed-support descriptor lacks; the saved notebook output does not quantify SIFT on the `coke_perspective` or `coke_scale` pairs.

### Question 2 — Geometry

On the **viewpoint stress clip (`coke_perspective`)**, the camera orbits and translates around the poster. This produces a general projective view of the planar board: opposite edges can have different apparent lengths and the board develops keystone. An affine map can model translation, rotation, scale, and shear, but it preserves parallel lines and cannot represent that perspective convergence. A homography has the projective degrees of freedom needed to map the flat board across viewpoints, so the affine column is expected to have larger corner error as the orbit becomes more oblique.

The whole-frame warp is expected to align the poster plane and leave residual displacement in the background. The homography was fitted to points on the poster plane; camera translation makes points at other depths move by different amounts, which is parallax. This does not harm the overlay because the replacement artwork is rendered only on the poster's coverage, while the original frame is preserved outside that mask. Parallax would affect placement if the target surface itself had depth variation—for example, if the poster were bent or the intended overlay crossed a protruding object during lateral camera motion.

## Stage 2 — Track
<!-- lk_flow, track_board (both variants on the common clip), failure_monitor; jitter and drift; frozen monitor thresholds -->
### Stress table (Stage 2)
<!-- generated rows: common · validation / test for both variants, blur, exit — corner metrics, jitter, drift, frames tracked, REINIT / HIDE counts -->
### Question 3 — Tracking
<!-- flicker (independent) vs drift (flow-assisted); eigenvalues: aperture vs brightness constancy, where each kind lies on the board (blur clip); monitor timeline of the combined clip: which statistic fired first at the card crossing and at the fast rotation, occlusion vs drift; what the exit clip's HIDE / REINIT sequence shows -->

## Stage 3 — Composite
<!-- hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite; frozen difference thresholds; no-occlusion baseline -->
### Stress table (Stage 3)
<!-- generated rows: common · validation / test, combined, viewpoint — mask IoU / F1 / leakage, baseline leakage (common), false occlusion (viewpoint) -->
### Question 4 — Occlusion
<!-- which occluders differencing recovers and which stay ambiguous; effect of the strong threshold and the minimum area; why hysteresis matters for thin occluders and why opening works against it (thinnest occluder that survives the frozen open_radius; what the radius buys over radius 0 on the validation masks); why the viewpoint clip produces false occlusion under strong perspective with a correct homography, and which part of the differencing is responsible -->

## Stage 4 — Your clip
### Capture and audit
<!-- capture conditions (resolution, fps, duration, board fraction, pan / distance / angle / occlusion crossing), privacy, audit annotations recorded before any method ran -->
### Overlay Scorecard
<!-- outputs/scorecards/<clip_id>_scorecard.md: both temporal variants, overlaylab.evaluate --clip <clip_id> -->
### Generative comparison
<!-- the open-weight video model you chose (direct pixel generation; inference scripts permitted), its settings, the video you gave it (whole clip, a cut, or joined segments), and why (cite the model); the seed, the log, the overlaylab.evaluate --frames report, comparison table against the classical output on the same frames, [AI-GENERATION] markers -->
### Question 5 — Reflection
<!-- classical vs your generative model on your clip: geometric placement, artwork identity, occlusion order, background preservation (what the background PSNR says about generated vs composited pixels); controllability, cost; what the model generated outside the board and in front of it; whether it kept the clip's timing (compare with the staff MiniMax-H3 example in data/genai_oracle/) -->

## Evidence table
| Student file | Function | Report section | Notebook figure / table / video |
|---|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | Stage 1 — implemented functions; Question 1 | Notebook 1: Harris/NMS; descriptor/ratio test; matching-rate and SIFT cells |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | Stage 1 — implemented functions; Question 2 | Notebook 1: DLT conditioning, RANSAC, and warp-convention sections; Notebook 2: viewpoint affine and parallax exercises (outputs pending) |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | | |
| temporal.py | lk_flow, track_board, failure_monitor | | |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | | |

## AI-use disclosure table
| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| Codex | `code/overlaylab/keypoints.py`, `code/overlaylab/geometry.py`, `code/REPORT.md` Stage 1 | [AI-CODE] [AI-LANGUAGE] | Assisted with the listed keypoint and geometry implementations in this collaboration and drafted the Stage 1 descriptions. | Compared the descriptions and reported observations with the function docstrings and saved Notebook 1 outputs |
