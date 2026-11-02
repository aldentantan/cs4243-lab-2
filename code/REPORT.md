# CS4243 Lab 2 report

This compact report records how every `YOUR CODE HERE` section is exercised. Elaborate on your implementation and the results you observed; add figures, plots and grounded reasoning that show your understanding. Every reported number must name its clip (common / viewpoint / scale / blur / exit / combined / your clip), frame subset (validation / test / audit) and method variant, and must come from `overlaylab.metrics` or `overlaylab.evaluate`. Thresholds are frozen on the common clip's validation frames; each Stage's **stress table** (`python -m overlaylab.report`, `outputs/stress_tables.md`) reports the common clip (validation and test) and the Stage's two stress clips with that frozen configuration. Paste the generated table into the Stage's section unchanged and explain the row the Stage's question points at; the other rows need no interpretation.

## Stage 1 — Localise
<!-- harris_response, detect_corners, describe_keypoints, match_descriptors; bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork; localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames; frozen success threshold -->

### Implemented functions

In `keypoints.py`, `harris_response` forms the Gaussian-smoothed second-moment matrix from the supplied image gradients and computes the Harris score. `detect_corners` thresholds that response, applies the selected window or greedy radius NMS, removes border points, and returns corners sorted by response. `describe_keypoints` estimates a dominant orientation and builds a normalized spatial gradient histogram around each keypoint; `match_descriptors` uses nearest and second-nearest Euclidean distances for Lowe's ratio test, with optional mutual matching. [AI-LANGUAGE]

In `geometry.py`, `bilinear_sample` samples gray or color images at floating-point coordinates. `normalise_points` applies Hartley centering and scaling; `dlt_homography` uses those normalized points to solve the homography by SVD and rejects rank-deficient configurations. `symmetric_transfer_error` scores forward and inverse reprojection; `ransac_homography` rejects degenerate samples, selects a consensus set, adapts its iteration bound, and refits on its inliers. `transfer_points` applies a homography to image points, and `warp_artwork` inverse-maps frame pixels into the artwork and returns both the sampled image and its coverage mask. [AI-LANGUAGE]

Notebook 1 exercises these implementations in its Harris/NMS, descriptor/ratio-test, normalized-DLT, RANSAC, and warp-convention sections. Notebook 2 exercises the higher-level localisation functions. `localise_board` joins detected features to RANSAC and projects the reference corners; `artwork_to_frame` composes the artwork-to-reference rectangle map with the board-to-frame homography; `overlay_frame` paints the warped coverage; and `evaluate_audit_frames` measures corner errors and summaries using the supplied metrics. [AI-LANGUAGE]

On the **common clip, validation subset** at the **640-pixel notebook resolution** with the frozen keypoint, descriptor, ratio-test, and RANSAC configuration, the p90 normalized corner error was **0.82%**. The notebook selected a **1.00%** success threshold by rounding that validation p90 up to the next percentage point. On the same common clip and method at 640 pixels, the **test subset** had median error **0.52%**, p90 error **0.71%**, and success on **92.31%** of its annotated frames, with no failed localisations. These are notebook-resolution results; the generated table below reports the final full-resolution results separately. [AI-LANGUAGE]

### Stress table (Stage 1)
<!-- generated rows: common · validation, common · test, viewpoint, scale — median / p90 corner error, success at the frozen threshold, inliers, inlier ratio, affine median error / success -->

| row | frames | median err | p90 err | success | inliers (median) | inlier ratio | affine median err | affine success |
|---|---|---|---|---|---|---|---|---|
| common · validation | 13 | 0.39% | 0.61% | 77% | 65 | 0.859 | 0.82% | 46% |
| common · test | 13 | 0.37% | 0.52% | 92% | 62 | 0.896 | 0.89% | 62% |
| viewpoint (coke_perspective) | 32 | 0.51% | 1.04% | 50% | 13 | 0.680 | 3.07% | 6% |
| scale (coke_scale) | 32 | 0.33% | 0.43% | 22% | 4 | 0.667 | 0.63% | 19% |

`localise_board` on every annotated frame with the frozen configuration; *affine* = 6-parameter least-squares fit to the same RANSAC inliers. [AI-LANGUAGE]

### Question 1 — Matching

Unless identified as SIFT, these are Notebook 1 matching rates for our fixed-patch oriented-gradient descriptor, selected radius-NMS detector, and configured ratio matcher; the rate is accepted matches divided by reference descriptors. Notebook 1's labelled rotation check on the **common clip, validation frame 176** shows the rate changing from **0.454** to **0.439** after a synthetic 30-degree in-plane rotation. On that same common-clip validation frame, downscaling the frame by 0.6 reduces the rate to **0.036**. The dominant-orientation step therefore absorbs much of an in-plane rotation: it rotates the descriptor's sampling grid to the keypoint's dominant gradient direction. The descriptor uses a fixed-size support window, however, so a scale change changes which image structures fall into its spatial cells. It also does not undo the spatial distortion caused by an oblique viewpoint. [AI-LANGUAGE]

The strongest adjacent-frame collapse in Notebook 1's annotated **viewpoint stress clip (`coke_perspective`)** is from frame **448 to 464**: matching rate **0.214 to 0.036**. Over those frames, the annotated quad's top-edge angle changes by **-13.5 degrees**, apparent scale changes from **235.7 to 222.6 pixels**, and the top-versus-bottom edge keystone measure increases from **0.013 to 0.034**. In the annotated **scale stress clip (`coke_scale`)**, frame **80 to 96** drops from **0.500 to 0.132**; the angle changes by **-1.9 degrees**, apparent scale from **190.5 to 172.6 pixels**, while keystone changes only from **0.014 to 0.003**. These are diagnostic annotated stress frames, not the common-clip validation/test split. The scale pair points to the fixed patch scale as the failing descriptor component; the viewpoint pair adds perspective distortion that orientation alignment cannot remove. [AI-LANGUAGE]

Because the reference has a fixed pixel size, the scale row of the Stage 1 stress table should be read as the effect of the board changing size relative to that unchanged reference. The Notebook 1 stress-pair results are consistent with a descriptor that is rotation-normalized but not scale-invariant. The generated table above now supplies the final full-resolution pipeline results; the notebook matching-rate experiment remains a separate descriptor diagnostic. [AI-LANGUAGE]

The labelled SIFT ablation compares the reference to **common-clip validation frame 448**: the notebook reports **0.086** for this descriptor and **0.179** for SIFT. That is evidence from the common clip, not a SIFT comparison on either stress pair. SIFT's scale-space keypoint detection gives it scale handling that this fixed-support descriptor lacks; the saved notebook output does not quantify SIFT on the `coke_perspective` or `coke_scale` pairs. [AI-LANGUAGE]

### Question 2 — Geometry

On the **viewpoint stress clip (`coke_perspective`)**, the camera orbits and translates around the poster, producing a general projective view of the planar board. An affine map can model translation, rotation, scale, and shear, but it preserves parallel lines and cannot represent the perspective convergence (keystone). A homography models the projective mapping of the flat board. In Notebook 2's **640-pixel diagnostic test frames**, using the same inlier correspondences, the median normalized corner error was **0.62%** for the homography and **3.50%** for the affine fit, computed over frames where both estimates were finite. At the frozen **1.00%** threshold, the homography succeeded on **56%** of all diagnostic test frames, compared with **6%** for affine. For example, on `coke_perspective` diagnostic test frame **240**, the homography error was **3.10%** and the affine error was **11.05%**; the fit degrades as the camera moves toward a more oblique view. [AI-LANGUAGE]

The whole-frame warp is expected to align the poster plane and leave residual displacement in the background. The homography was fitted to points on the poster plane. Camera translation makes points at other depths move by different amounts, which is parallax. This does not harm the overlay because the replacement artwork is rendered only on the poster's coverage, while the original frame is preserved outside that mask. Parallax would affect placement if the target surface itself had depth variation, for example, if the poster were bent or the intended overlay crossed a protruding object during lateral camera motion. [AI-LANGUAGE]

Notebook 2 section 6 chooses the viewpoint comparison from frames whose frozen localiser produced valid homographies. It composes `H_b @ inverse(H_a)` and warps the entire source frame in the labelled comparison cell. The embedded target / warped-source / difference montage shows the board-plane alignment and residual background displacement. It uses predicted homographies throughout, with no ground-truth fallback; the refreshed example warps **viewpoint frame 224 to frame 496**, at **640-pixel width**. Artwork composition is `H_artwork_to_frame = H_board_to_frame @ S_artwork_to_reference`; points use x along columns, y along rows and integer pixel centres. [AI-LANGUAGE]

## Stage 2 — Track
<!-- lk_flow, track_board (both variants on the common clip), failure_monitor; jitter and drift; frozen monitor thresholds -->
### Stress table (Stage 2)
<!-- generated rows: common · validation / test for both variants, blur, exit — corner metrics, jitter, drift, frames tracked, REINIT / HIDE counts -->

| row | frames | median err | p90 err | success | jitter | drift (≥16 fr) | longest run | tracked | (re)localisations | REINIT flags | HIDE flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| common · validation · independent | 13 | 0.39% | 0.61% | 77% | 6.07 px | n/a | 17 | 426 | 426 | 95 | 74 |
| common · validation · flow-assisted | 13 | 0.39% | 0.69% | 77% | 6.18 px | n/a | 20 | 433 | 185 | 95 | 67 |
| common · test · independent | 13 | 0.37% | 0.52% | 92% | 6.07 px | n/a | 17 | 426 | 426 | 95 | 74 |
| common · test · flow-assisted | 13 | 0.36% | 0.53% | 92% | 6.18 px | n/a | 20 | 433 | 185 | 95 | 67 |
| blur (coke_blur) · flow-assisted | 32 | 0.65% | 0.91% | 6% | 3.50 px | n/a | 147 | 31 | 15 | 10 | 469 |
| exit (coke_out_of_view) · flow-assisted | 30 | 0.70% | 1.52% | 47% | 6.86 px | n/a | 25 | 299 | 187 | 112 | 201 |

From the `_stats.json` of each run. Jitter uses the full available corner trajectory for motion interpolation. Jitter, longest run, tracked and the counts are whole-clip statistics (identical in validation and test rows for a variant); (re)localisations counts actual localiser calls, whereas REINIT/HIDE count monitor flags; drift = median corner error on the row's annotated frames at least 16 frames after the last (re)initialisation (n/a when no track is carried that long: the independent variant reinitialises every frame); longest run = most consecutive frames without a (re)initialisation. [AI-LANGUAGE]

### Implemented functions

In the `temporal.py` file, `lk_flow` performs iterative sparse Lucas–Kanade updates using the supplied image gradients and bilinear sampling. It returns point displacements, structure-tensor eigenvalues, brightness residuals, and validity flags. `track_board` implements independent reference registration and flow-assisted propagation, refits the homography on propagated correspondences and reinitialises from the reference when tracking loses support or reaches its configured interval. `failure_monitor` applies the ordered model, inlier, residual, corner-motion, flow, and brightness checks and returns `OK`, `REINIT`, or `HIDE` with the triggering statistic. `_put_text` draws into an 8-bit mask and blends into float textures so the synthetic diagnostic runs with OpenCV 5. [AI-LANGUAGE]

### Question 3 — Tracking
<!-- flicker (independent) vs drift (flow-assisted); eigenvalues: aperture vs brightness constancy, where each kind lies on the board (blur clip); monitor timeline of the combined clip: which statistic fired first at the card crossing and at the fast rotation, occlusion vs drift; what the exit clip's HIDE / REINIT sequence shows -->

On the **common clip, 640-pixel segment from frames 170–239**, independent registration returned valid estimates on **64 of 70** frames with **64 successful** reference registrations; its `metrics.jitter` was **2.24 px**. Flow-assisted tracking returned valid estimates on **66 of 70** frames with **7 successful** reference registrations; its `metrics.jitter` was **3.41 px**. Both jitter values use the annotated common-clip frames in that segment (validation and test). In this segment flow assistance required fewer reference registrations, but it did not reduce measured jitter; the annotated corner-error sequence also contains a **3.03%** flow-assisted error, consistent with a local drift episode. Independent registration can flicker as each fresh keypoint set and RANSAC fit changes; flow-assisted tracking can accumulate small Lucas–Kanade errors until it reinitialises. A successful stabilisation example is the frozen **personal clip** in Notebook 4 Part I: recovered-output jitter falls from **8.54 px independent to 3.68 px flow-assisted** across the full audit trajectory. The saved flow-assisted output also shows the frame-274 drift failure, so smoother output does not imply better placement. [AI-LANGUAGE]

The **full-resolution common validation/test table**, using all available annotations for jitter, gives **6.07 px independent versus 6.18 px flow-assisted**: flow does not reduce jitter on this common-clip run. The supplied builder reports **426 independent versus 185 flow-assisted (re)localisations**, the successful reference registrations recorded in `reinit_frames`. Counting the non-null per-frame `n_matches` records separately gives **500 versus 252 actual call attempts**, including failures. HIDE frames fall from **74 to 67**. The supplied table's **longest run** is the largest interval between successful registrations or the clip end: **17 frames independent**, **20 flow-assisted**, and **147 on the blur clip**. These intervals can contain lost frames; the independent variant still attempts registration on every frame, and the blur run has only **31 tracked frames** with **469 HIDE flags**. The blur table's **15 (re)localisations** are successful registrations, whereas its per-frame records show **484 attempts**. Thus the long blur interval indicates a gap in successful registration, not a 147-frame valid track. Drift at age ≥16 is **n/a** because there is no eligible finite annotated error at that age; this is missing long-age evidence rather than proof of zero drift. The 640-pixel segment's **3.03%** error provides separate evidence of a local drift episode. [AI-LANGUAGE]

Notebook 2 section 7 retains the **common-clip frames 170–171** comparison (**49 points**, **5 valid**, median disagreement with OpenCV **0.005 px**). Section 10 adds one compact spatial/eigenvalue figure for **blur frames 2–3, full resolution**, using the frozen LK settings. It combines **104 saved inliers** with **36 evaluation-only points on the predicted board edges**; those extra queries never initialise or correct the pipeline. Of these **140 queries**, **31** have adequate `lambda_1` but `lambda_2` below **0.0002**, **28** have adequate `lambda_2` but finite residual above **0.08**, and **72** are valid. The yellow points lie predominantly on the straight board edges: motion perpendicular to an edge is constrained, while motion along it remains ambiguous (the aperture problem). The red points lie in the textured interior and have two adequate eigenvalues, so their large residual is consistent with brightness-constancy failure under blur rather than lack of directional texture. Other invalid points can fail convergence or image bounds; neither eigenvalues nor residuals alone uniquely determine the physical cause. [AI-LANGUAGE]

Notebook 2 section 10 plots the **combined and exit full-resolution flow-assisted timelines** and prints event transitions. At the combined clip's **first labelled card window, 212–220**, the model is already lost at frame **212** (`HIDE/no_model`), recovers at **214**, and has no monitor warning on **214–220**. In the preceding context, **205** raises `REINIT/brightness_change` (**0.197**, threshold **0.15**) before another loss on **206**. At the **second labelled card window, 304–314**, the first non-OK flag is **309, HIDE/no_model**; it recovers at **310**, hides again at **311–312**, recovers at **313** and hides at **314**. These windows are bounded by the supplied card-mask annotations; they are not newly estimated exact crossing onset times. [AI-LANGUAGE]

For fast rotation, the largest adjacent annotated top-edge-angle change occurs over combined frames **80–96** (about **5.24 degrees/frame**, evaluation diagnostic only). The first non-OK flag in that window is **83, HIDE/no_model**; at **96**, a successful reference registration raises `REINIT/brightness_change` (**0.200**). Across the whole combined run the monitor records **357 HIDE/no_model**, **41 REINIT/brightness_change** and **102 OK** frames. It does not actually label these episodes as occlusion versus drift. Lost-model status has priority; when a model exists, brightness change can be caused by both foreground occlusion and moving texture sampled at different positions. Mean flow and corner displacement provide motion context but are unavailable or reset on many failed/relocalised frames. A low inlier ratio alone cannot establish either cause. [AI-LANGUAGE]

On the **exit clip**, the supplied full-resolution table gives **187 (re)localisations**, a longest registration interval of **25 frames**, **201 HIDE** and **112 REINIT** flags; the saved per-frame records separately show **388 actual localiser attempts**, and every REINIT trigger is `brightness_change`. Around the first prolonged loss, frame **46** requests REINIT, **48–71** hide with `no_model`, **72** successfully registers from the reference but still exceeds the brightness threshold, **73–83** hide again, and **84** successfully registers with an **OK** flag. The timeline shows recovery from image/reference matching, without GT initialisation. A warning on the first reappearance is therefore not the same as a failed reinitialisation. The current tracker attempts reference registration after loss or insufficient propagated support; the monitor's post-tracking REINIT flag is a policy measurement, not itself a count of localiser calls. [AI-LANGUAGE]

## Stage 3 — Composite
<!-- hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite; frozen difference thresholds; no-occlusion baseline -->
### Stress table (Stage 3)
<!-- generated rows: common · validation / test, combined, viewpoint — mask IoU / F1 / leakage, baseline leakage (common), false occlusion (viewpoint) -->

| row | eligible masks | scored masks | hidden masks | missing masks | overlay absent | IoU | F1 | leakage | baseline leakage | false occlusion |
|---|---|---|---|---|---|---|---|---|---|---|
| common · validation | 3 | 3 | 0 | 0 | 0 | 0.915 | 0.956 | 1.52% | 96.25% | n/a |
| common · test | 3 | 2 | 1 | 0 | 1 | 0.932 | 0.965 | 0.61% | 61.53% | n/a |
| combined (coke_unconstrained) | 6 | 4 | 2 | 0 | 2 | 0.936 | 0.966 | 0.20% | n/a | n/a |
| viewpoint (coke_perspective) | 3 | 0 | 2 | 0 | 2 | n/a | n/a | n/a | n/a | 5.49% |

IoU / F1 from the occlusion masks saved by `run_pipeline` on the ground-truth mask frames (frames on which the overlay is hidden are not scored); leakage, baseline leakage and false occlusion from `overlaylab.evaluate` on the finished videos. The viewpoint clip has empty masks: only false occlusion applies. [AI-LANGUAGE]

### Question 4 — Occlusion
<!-- which occluders differencing recovers and which stay ambiguous; effect of the strong threshold and the minimum area; why hysteresis matters for thin occluders and why opening works against it (thinnest occluder that survives the frozen open_radius; what the radius buys over radius 0 on the validation masks); why the viewpoint clip produces false occlusion under strong perspective with a correct homography, and which part of the differencing is responsible -->

[`occlusion_mask`](overlaylab/compositing.py#L113) warps the original poster reference; [`compensated_difference`](overlaylab/compositing.py#L53) fits a global gain and offset on the best-agreeing covered pixels and measures the remaining colour difference. [`occlusion_mask`](overlaylab/compositing.py#L113) uses this signal to protect foreground objects before [`composite`](overlaylab/compositing.py#L222) inserts the artwork. A card region that differs strongly from the original poster is recovered. A blue card over similarly blue poster pixels, or a motion-blurred edge, can remain below the weak threshold. This is an appearance ambiguity, even with correct geometry. The saved common-clip test results below show weaker boundary recovery at frame 448 than at frames 144 and 208. Shadows are another ambiguity. A local shadow changes intensity without replacing poster texture, and one global brightness fit cannot necessarily remove it. The current code has no separate shadow classifier. A specular highlight can likewise become false foreground because a global gain/offset cannot explain a local view-dependent reflection. A stationary occluder remains detectable: differencing compares the observed frame to the clean warped reference, not to the previous frame, so the occluder does not disappear merely because it stops moving. Similar-colour regions can still be missed. [AI-LANGUAGE]

These are **Notebook 3, section 4**, common-clip **test** masks at **640-pixel processing width**, using frozen **per-frame localisation + difference masking**; they are not the full-resolution flow-assisted video results above. All entries come from `overlaylab.metrics` calls in [Notebook 3](notebooks/03_occlusion_and_compositing.ipynb), cell `a0a0a2a7`. [AI-LANGUAGE]

| common test frame | IoU | F1 | boundary F1 | leakage | no-occlusion baseline leakage |
|---|---|---|---|---|---|
| 144 | 0.924 | 0.960 | 0.673 | 0.9% | 93.4% |
| 208 | 0.950 | 0.975 | 0.926 | 0.7% | 96.4% |
| 448 | 0.806 | 0.892 | 0.416 | 0.6% | 82.8% |

**Thresholds and hysteresis:** [`select_difference_thresholds`](overlaylab/compositing.py#L165) selected settings on common-clip **validation frames 96, 192 and 272** only. The saved [frozen configuration](outputs/thresholds.json), `occlusion`, uses strong **0.20**, weak **0.12**, minimum area **152 full-resolution pixels** (**38** at Notebook 3's processing width), and opening radius **2**. Strong pixels seed [`hysteresis`](overlaylab/compositing.py#L23) while connected weak pixels extend the mask through low-contrast or blurred boundaries. Raising the strong threshold can remove seeds, while raising minimum area rejects any small components that are disconnected. When validation settings tie in mean IoU (absolute tolerance 1e-12), `select_difference_thresholds` takes the median of the distinct tied strong levels, chooses the nearest tied grid value (lower on an equal-distance tie), then prefers weak/minimum-area values closest to the incoming validation configuration, finally breaking ties by their numeric values. This deterministic rule selects a point inside the validation plateau; stress and personal results do not select it. [AI-LANGUAGE]

The saved **common validation frame 192**, frozen per-frame method, **Notebook 3 section 5** ablation (cell `ee16eb20`) indicates: [AI-LANGUAGE]

| setting on common validation frame 192 | strong | weak | minimum area at notebook resolution | IoU | leakage |
|---|---|---|---|---|---|
| frozen | 0.20 | 0.12 | 38 | 0.903 | 0.1% |
| higher strong threshold | 0.30 | 0.12 | 38 | 0.903 | 0.1% |
| larger minimum area | 0.20 | 0.12 | 228 | 0.903 | 0.1% |
| hysteresis disabled | 0.20 | 0.20 | 38 | 0.671 | 20.1% |

Thus this frame supports retaining weak pixels connected to strong evidence; it does not show a benefit from increasing either the strong threshold or component size. [AI-LANGUAGE]

**Opening versus thin occluders:** 

Opening erodes then dilates the candidate mask, removing narrow protrusions and speckles that hysteresis may retain. The frozen radius produces a **5 × 5 elliptical kernel**. An axis-aligned straight strip needs at least **5 processing pixels** of width to contain that kernel. This is a geometric bound from `occlusion_mask`, not a measured minimum physical occluder width in these clips. Connectivity, threshold contrast and component area must also pass, so width alone does not guarantee survival of the occluder pixels. [AI-LANGUAGE]

An evaluation-only ablation kept all other frozen settings and the same localisations on **common validation frames 96, 192 and 272**, at **640-pixel width**, and compared the means from `overlaylab.metrics`. The exact per-frame values and configuration are in [question4_opening_ablation.json](student_files/question4_opening_ablation.json). [AI-LANGUAGE]

| common validation per-frame method | mean IoU | mean boundary F1 | mean leakage |
|---|---|---|---|
| opening radius 0 | 0.9005 | 0.7144 | 1.3313% |
| frozen opening radius 2 | 0.9024 | 0.7213 | 1.3454% |

Opening results in a small improvement in mask overlap and average boundary agreement, with slightly more artwork leakage. It is a cleanup tradeoff, not a universal improvement. The frozen thresholds were not changed after this ablation. [AI-LANGUAGE]

**False occlusion under perspective:**

A homography aligns a plane's geometry but does not guarantee identical sampled colours. Foreshortening, resampling, blur and spatially varying illumination can leave residual differences that the single gain/offset fit cannot remove. The thresholding stage then treats changed poster appearance as foreground. In **Notebook 3 section 6**, the **viewpoint diagnostic frames 224 and 288** use explicitly labelled **evaluation-only GT-fallback homographies**. Their masks still remove visible poster pixels. This supports a photometric failure rather than attributing every false region to tracking. The fallback is never used to render the submitted pipeline videos. [AI-LANGUAGE]

## Stage 4 — Your clip
### Capture and audit
<!-- capture conditions (resolution, fps, duration, board fraction, pan / distance / angle / occlusion crossing), privacy, audit annotations recorded before any method ran -->

My personal clip shows a flat wall-mounted colourful poster, handheld changes of distance and viewing angle, and a blue book crossing in front. The [capture manifest](manifests/capture_manifest.csv), row `clip`, records an iPhone capture at **1920 × 1080, 30 fps, 13.735 s**, a **0.21 m × 0.14 m** board and a recorded board fraction of **0.113**, with consent obtained and metadata removed. The original is [clip.mp4](../data/personal/clip/clip.mp4); the [source frame 257](../data/personal/clip/frames/frame_000257.png) shows the book crossing. [AI-LANGUAGE]

The [corner manifest](manifests/audit_corners.csv) records **12 audit frames**: **0, 46, 91, 137, 183, 228, 257, 274, 320, 365, 385 and 411**. The [mask manifest](manifests/audit_masks.csv) records **0, 183, 257, 274 and 385**. Both manifests declare `annotated_before_inference=yes`. The saved audit PNGs now use the evaluator convention: **occluder white; visible poster and background black**. A one-time representation migration saved **annotated board quadrilateral AND NOT original visible-poster mask** to the existing audit paths. Notebook 4 and the standard evaluator both load those PNGs directly, without inversion. The original visible-white PNG bytes are preserved in [mask provenance](student_files/mask_provenance/visible_poster_originals/). All five canonical masks match the conversion formula pixel-for-pixel; the original migration time was not retained. Annotation indices, corners and manifest declarations were unchanged. This is not new annotation. Thin differences between the corner quadrilateral and original visible-mask outline were retained and can be counted as foreground even on clean frames; the historical [input acceptance report](outputs/genai/inputs/book_tracking_repaired_v1/acceptance_report_visible.json) flags this annotation-review limitation. [AI-LANGUAGE]

### Overlay Scorecard
<!-- outputs/scorecards/<clip_id>_scorecard.md: both temporal variants, overlaylab.evaluate --clip <clip_id> -->

The [complete personal-clip scorecard](outputs/scorecards/clip_scorecard.md) compares [independent](outputs/videos/clip_independent.mp4) and [flow-assisted](outputs/videos/clip_flow_assisted.mp4) outputs made with the frozen configuration and `nus_ranking.png`. The following rows quote the **full-video audit evaluation**, not the smaller generative comparison subset. Geometry and fidelity use the **12 corner audit frames** listed above; occlusion uses the **five mask audit frames**, with **five scored coverage frames, no detected-overlay absences and no missing output frames** for each variant. [AI-LANGUAGE]

| personal clip, full-video audit | median / p90 corner error | placement success | NCC / SSIM / structure | mask-frame leakage | visible-board coverage |
|---|---|---|---|---|---|
| independent | 1.08% / 1.74% | 41.7% | 0.740 / 0.477 / 0.409 | 61.6% | 94.3% |
| flow-assisted | 1.40% / 4.42% | 16.7% | 0.673 / 0.425 / 0.333 | 53.5% | 89.0% |

Sources: [independent full-video report](outputs/evaluation/clip_independent_report.json) and [flow-assisted full-video report](outputs/evaluation/clip_flow_assisted_report.json) measures `placement`, `fidelity` and `occlusion`. The table favours independent placement and identity on this clip although none of the variants reliably protect the book. The mask-outline ambiguity described above limits interpretation of the aggregate leakage, so the clearly occluded **frame 257** is examined separately below. [AI-LANGUAGE]

### Generative comparison
<!-- the open-weight video model you chose (direct pixel generation; inference scripts permitted), its settings, the video you gave it (whole clip, a cut, or joined segments), and why (cite the model); the seed, the log, the overlaylab.evaluate --frames report, comparison table against the classical output on the same frames, [AI-GENERATION] markers -->

**[AI-GENERATION]** I used the open-weight [Wan2.1-VACE-1.3B model](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B), the smaller VACE option supporting 480p, with model-generated edited pixels. The [run log](manifests/genai_log.csv) records **seed 500, 50 sampling steps, 81 requested frames, 480p and model offloading**, on the NUS SoC cluster requesting an **A100-40 GPU and eight CPU cores**. Actual allocation, peak VRAM and the inference repository commit were not recorded. The preserved result is [raw output.mp4](outputs/genai/output.mp4) **[AI-GENERATION]**. The log contains one video generation. [AI-LANGUAGE]

The model received the **personal source interval 120–269 inclusive**, a **5.00 s** cut, [`nus_ranking.png`](../data/artwork/nus_ranking.png), and the paired [gray-filled video](outputs/genai/inputs/book_tracking_repaired_v1/interval_inpaint.mp4) / [editable mask](outputs/genai/inputs/book_tracking_repaired_v1/src_mask.mp4). The separate [book tracker and helper](student_files/prepare_vace_inputs.py) exclude the predicted blue book from editing and repair missing board geometry from neighbouring frozen predictions; [preparation.json](outputs/genai/inputs/book_tracking_repaired_v1/preparation.json) records the repairs. This prepares inputs without compositing the replacement artwork. The classical comparison retains its original Stage 3 masks, so occlusion improvements cannot be assigned to the generative model alone. See the [prepared frame 257](outputs/genai/inputs/book_tracking_repaired_v1/review_000257.jpg). [AI-LANGUAGE]

The submission prompt in [genai_conditions.json](manifests/genai_conditions.json), `prompt.template`, is: [AI-DESIGN]

> Replace the advertisement on the board with the attached artwork.

Prompt records were standardized on **3 November 2026**; the existing video was retained without regeneration. [AI-LANGUAGE]


**Audit protocol:** 

All methods use source frames **137 (clean), 228 (perspective), 257 (occluded), and pair 229–230**, with the same complete audit dictionary loaded directly from the canonical occluder-white PNGs described above. Placement, fidelity and preservation are scored on the selected corner-labelled frames **137, 228 and 257**; occlusion uses **257** only; temporal statistics are restricted to **229–230**. This pair precedes the book crossing visible at frame 257, so it does not measure temporal stability through that crossing. No source frame is missing from the extracts. [Notebook 4](notebooks/04_personal_clips_and_genai_analysis.ipynb), cell `5b106b28`, records the evaluation protocol and preserves other adjacent-pair results separately. [AI-LANGUAGE]

The [extraction manifest](outputs/genai/audit_extraction_manifest.json) records the raw **832 × 480, 16 fps** output and nominal timestamp mapping **137→9, 228→58, 229→58, 230→59, 257→73**, followed by resizing audit extracts to source resolution. The true temporal pair maps to distinct output frames and the perspective frame 228 shares an output frame with 229 because the output frame rate is lower. This is declared sampling, with no favourable-frame search or retiming correction. [AI-LANGUAGE]

The following three tables quote `overlaylab.evaluate` results from the [independent selected-frame report](outputs/evaluation/classical_independent_frames_report.json), [flow-assisted selected-frame report](outputs/evaluation/classical_flow_assisted_frames_report.json) and [generative selected-frame report](outputs/evaluation/genai_frames_report.json). The saved [comparison contact sheet](outputs/genai/audit_contact_sheet.jpg) labels the same source indices and marks generated pixels **[AI-GENERATION]**. [AI-LANGUAGE]

**Table A — personal clip, corner audit subset 137 / 228 / 257: placement and artwork fidelity.**

| method | median error | success | artwork detected | NCC | SSIM | structure |
|---|---|---|---|---|---|---|
| classical_independent | 1.48% | 33% | 100% | 0.682 | 0.450 | 0.342 |
| classical_flow_assisted | 1.48% | 33% | 100% | 0.682 | 0.449 | 0.343 |
| generative [AI-GENERATION] | n/a | 0% | 0% | 0.234 | 0.325 | 0.101 |

**Table B — personal clip: temporal pair 229–230; background preservation on corner audit subset 137 / 228 / 257.** [AI-LANGUAGE]

| method | pair flicker | pair warping error | pair jitter (px) | background PSNR (dB) | background SSIM | background drift (px) |
|---|---|---|---|---|---|---|
| classical_independent | 0.0918 | 0.0627 | 10.22 | 36.2 | 0.983 | 0.33 |
| classical_flow_assisted | 0.0918 | 0.0627 | 10.22 | 36.2 | 0.983 | 0.33 |
| generative [AI-GENERATION] | 0.0903 | 0.0761 | n/a | 25.4 | 0.922 | 1238.24 |

**Table C — personal clip, occluded audit frame 257 only.** `overlay absent` is the evaluator's artwork-detection flag, not proof of visually absent generated content. [AI-LANGUAGE]

| method | leakage | visible-board coverage | coverage-scored / hidden / missing frames | overlay absent |
|---|---|---|---|---|
| classical_independent | 53.3% | 92.2% | 1 / 0 / 0 | 0 |
| classical_flow_assisted | 53.4% | 92.3% | 1 / 0 / 0 | 0 |
| generative [AI-GENERATION] | 2.1% | n/a | 0 / n/a / 0 | 1 |

The classical videos have no hidden selected output frames. A classical tracker `HIDE` flag does not apply to VACE. Its coverage is unscored because the artwork detector fails. Low leakage alone therefore cannot establish a successful complete replacement. [AI-LANGUAGE]

**Table D — recorded processing cost, with different duration and hardware scopes.** Times come from `timing.track_s + timing.composite_s` in the [independent statistics](outputs/videos/clip_independent_stats.json) and [flow-assisted statistics](outputs/videos/clip_flow_assisted_stats.json), and `processing_time_s` in the [VACE log](manifests/genai_log.csv); they are run metadata, not quality scores. [AI-LANGUAGE]

| method | processing time (s) | input scope | recorded compute |
|---|---|---|---|
| classical_independent | 400.25 | 13.73 s full personal clip | Macbook Pro M5 chip, as recorded in Notebook 4's comparison table |
| classical_flow_assisted | 199.63 | 13.73 s full personal clip | Macbook Pro M5 chip, as recorded in Notebook 4's comparison table |
| generative [AI-GENERATION] | 900.0 | 5.00 s personal-clip cut | NUS SoC SLURM; requested A100-40 GPU and eight CPU cores |

These timings exclude the evaluation pass and are not a controlled speed benchmark. Manual annotation, preprocessing and debugging effort was not timed. [AI-LANGUAGE]

### Question 5 — Reflection
<!-- classical vs your generative model on your clip: geometric placement, artwork identity, occlusion order, background preservation (what the background PSNR says about generated vs composited pixels); controllability, cost; what the model generated outside the board and in front of it; whether it kept the clip's timing (compare with the staff MiniMax-H3 example in data/genai_oracle/) -->

1. **Geometric placement.** Both classical variants follow the board's quadrilateral more faithfully in [clean frame 137](outputs/genai/classical_independent/frame_000137.png) and [perspective frame 228](outputs/genai/classical_independent/frame_000228.png), although original poster fragments remain near the boundary. **Table A**, personal corner audit subset, quotes **1.48% median error and 33% success** for each classical variant. **Table A's generative row [AI-GENERATION]** has **0% detection and n/a error**, so VACE has no reliable measured placement score; [its frame 228](outputs/genai/frame_000228.png) **[AI-GENERATION]** shows the broad plane but does not establish accurate corner alignment.

2. **Artwork identity.** **Table A**, same personal audit subset, quotes classical **NCC 0.682** versus generative **0.234**; SSIM and structure also favour the classical rows. The [classical clean frame](outputs/genai/classical_independent/frame_000137.png) preserves recognisable NUS wording, while [VACE frame 137](outputs/genai/frame_000137.png) **[AI-GENERATION]** distorts the lettering and layout despite retaining blue tones and a building-like image. Neither method's evaluated output reproduces the reference perfectly after warping, masking and encoding but the VACE-generated video retains much fewer identifiable characteristics of the original artwork.

3. **Occlusion order.** **Table C**, personal occluded frame **257**, quotes classical leakage **53.3% / 53.4%** with coverage **92.2% / 92.3%**, versus VACE **2.1% leakage and n/a coverage**. Comparing [independent frame 257](outputs/genai/classical_independent/frame_000257.png), [flow-assisted frame 257](outputs/genai/classical_flow_assisted/frame_000257.png) and [VACE frame 257](outputs/genai/frame_000257.png) **[AI-GENERATION]**, classical artwork spills onto the book, whereas VACE preserves its foreground order more successfully. VACE received the [dedicated book-aware prepared input](outputs/genai/inputs/book_tracking_repaired_v1/review_000257.jpg), so this is a comparison of the complete methods, including preprocessing. Its unscored coverage prevents a claim that it painted the entire visible poster.

4. **Background preservation.** **Table B**, personal corner audit subset, quotes classical **36.2 dB PSNR / 0.983 SSIM** versus VACE **25.4 dB / 0.922**. This supports more outside-board change in VACE, visible in the [contact sheet](outputs/genai/audit_contact_sheet.jpg) **[AI-GENERATION]**. PSNR alone cannot prove how pixels were produced, and no matched source re-encoding baseline was measured here. The stronger code guarantee is limited as [`composite`](overlaylab/compositing.py#L222) preserves source pixels wherever its alpha is zero **before encoding**. Incorrect predicted coverage can reach outside the true board, and encoding can alter decoded pixels; therefore neither finished MP4 is proven pixel-identical outside the annotated board.

5. **Controllability.** The documented instruction is “Replace the advertisement on the board with the attached artwork.” The saved output in [VACE frame 137](outputs/genai/frame_000137.png) **[AI-GENERATION]** still changes the text. Prompt edits can request a different appearance, but exact geometry, the compositing support and mask logic are code-controlled through [`artwork_to_frame`](overlaylab/localisation.py), [`composite`](overlaylab/compositing.py) and the [book tracker](student_files/book_tracking.py). The [classical frame 137](outputs/genai/classical_independent/frame_000137.png) and [prepared book frame 257](outputs/genai/inputs/book_tracking_repaired_v1/review_000257.jpg) illustrate those controls. A model input mask requests where to edit; it does not guarantee exact generated artwork or unchanged output pixels elsewhere.

6. **Timing.** **Table B**, declared personal pair **229–230**, quotes classical **flicker 0.0918 / warping error 0.0627** and VACE **0.0903 / 0.0761**. VACE's slightly lower flicker is accompanied by higher warping error; it is not evidence that timing is preserved. Compare [VACE frame 229](outputs/genai/frame_000229.png) and [230](outputs/genai/frame_000230.png) **[AI-GENERATION]** with the corresponding classical extracts. The extraction manifest preserves nominal timestamps, but sampling cannot undo model retiming. **Table B's 1238.24 px VACE background drift** is an unreliable correspondence estimate on changed imagery, rather than literal camera displacement or a measured timing offset. We can report appearance/motion mismatch, but these metrics do not isolate retiming from re-rendering.

7. **Cost.** **Table D** quotes **400.25 s independent**, **199.63 s flow-assisted**, and **900.0 s VACE [AI-GENERATION]**, for the explicitly different full-clip/cut scopes and hardware. VACE also required the separate prepared inputs; see [frame 257's preprocessing preview](outputs/genai/inputs/book_tracking_repaired_v1/review_000257.jpg). 

**Staff oracle comparison.** The [staff MiniMax-H3 report](../data/genai_oracle/minimax_h3/report.json) **[AI-GENERATION]** scores the **common clip**, corner audit frames **160 / 192 / 320**, occlusion frame **192**, and temporal pair **160–161**. Its `placement` / `fidelity` fields quote **100% artwork detection and NCC 0.317**, compared with **0% and 0.234** in **Table A's personal-clip VACE row**. In these examples the larger model produces more detectable artwork, but does not solve precise geometry or background preservation: the oracle reports **2.67% median placement error** and **20.04 dB background PSNR**. On its **occluded frame 192**, it still has **8.3% leakage and 90.0% coverage**. The [staff frame 320](../data/genai_oracle/minimax_h3/frame_000320.png) and [contact sheet](../data/genai_oracle/contact_sheet.jpg) **[AI-GENERATION]** also show a substantial mismatch at the nominal source time, consistent with changed camera motion or timing. Different clips, artworks and conditioning prevent attributing these differences solely to model size; each run is a qualitative probe.

## Evidence table
| Student file | Function | Report section | Notebook figure / table / video |
|---|---|---|---|
| keypoints.py | harris_response, detect_corners, describe_keypoints, match_descriptors | Stage 1 — implemented functions; Question 1 | Notebook 1: Harris/NMS; descriptor/ratio test; matching-rate and SIFT cells |
| geometry.py | bilinear_sample, normalise_points, dlt_homography, symmetric_transfer_error, ransac_homography, transfer_points, warp_artwork | Stage 1 — implemented functions; Question 2 | Notebook 1: DLT conditioning, RANSAC, and warp-convention sections; Notebook 2: saved viewpoint affine comparison; predicted-homography viewpoint warp montage in Notebook 2 section 6 |
| localisation.py | localise_board, artwork_to_frame, overlay_frame, evaluate_audit_frames | Stage 1 — implemented functions; validation, test, and Question 2 sections | Notebook 2: validation/test metrics and viewpoint affine comparison; predicted-homography viewpoint warp in Notebook 2 section 6 |
| temporal.py | lk_flow, track_board, failure_monitor | Stage 2 — implemented functions; Question 3 | Notebook 2: common-segment flow/variant metrics and synthetic illumination monitor; common-clip eigenvalue/location plots (section 7), compact blur eigenvalue/location diagnostic (section 10), combined/exit monitor timelines and event transitions (section 10), and both common-clip videos |
| compositing.py | hysteresis, compensated_difference, occlusion_mask, select_difference_thresholds, composite | Stage 3 — Question 4, frozen thresholds, hysteresis/opening and false-occlusion analysis | Notebook 3 sections 2–6: difference/gain fit, validation search, test-mask table and composites, fixed-radius validation ablation, labelled GT-fallback stress diagnostics; Stage 3 table and no-occlusion baseline video |

## AI-use disclosure table

| Tool / model | Affected artefacts (file, cell, figure, section) | Markers | What the tool contributed | How it was verified |
|---|---|---|---|---|
| Codex | Assessed `overlaylab/keypoints.py`, `geometry.py`, `localisation.py`, `temporal.py`, `compositing.py`; synthetic text rendering; report Stages 1–4 | [AI-CODE] [AI-LANGUAGE] | Assistance with the listed Python implementations and report prose, as previously disclosed | Supplied tests, saved annotations, evaluator reports and regenerated stress tables. |
| Codex | `student_files/book_tracking.py`, `prepare_vace_inputs.py`, their tests; Notebook 4 Part II section 1, marked source-frame/board-mask export block | [AI-CODE] | Separate book-aware preprocessing, predicted-board gap repair and VACE input export; current helper prior uses the lab Harris/LK functions | Synthetic foreground/clean-board, disappearance, motion, export-alignment and gap-repair tests; notebook input export. Historical inputs/output are retained; the old helper's restricted calls remain a historical protocol limitation. |
| Codex | Notebook 2 section 10, marked blur inlier/edge-query diagnostic; Notebook 4 Part I, marked canonical audit-mask loading; Notebook 4 Part II section 4, marked canonical audit-dictionary reload | [AI-CODE] | Compact blur spatial/eigenvalue diagnostic and direct loading of canonical audit masks/dictionaries | Notebooks 2 and 4 re-executed without errors; mask convention checked against saved annotations; saved table rows checked against the restored supplied generator; report interpretation reconciled with the saved statistics. |
| Codex | Notebook 4 Scorecard dictionary's Description and Uncertainty text; Part II prompt-record note and section 4 mask-convention explanation | [AI-LANGUAGE] | Wording of the marked Scorecard descriptions, prompt-record note and audit-mask explanation | Scorecard statements checked against saved evaluator results; mask explanation checked against the canonical and preserved visible-poster masks. |
| Wan2.1-VACE-1.3B | `outputs/genai/output.mp4`, generated audit extracts and contact-sheet generated panels | [AI-GENERATION] | One direct pixel-generation attempt, seed 500; settings in `manifests/genai_log.csv` | Same-index, same-audit-dictionary evaluator comparison with classical outputs. No claim that low leakage establishes full replacement. |
| Alden (capture and annotation, as declared in manifests) | `data/personal/clip/`, audit/capture manifests; original visible-white masks retained under `student_files/mask_provenance/` | [HUMAN-DATA] | Personally captured footage, selected frames and hand annotations; manifests declare annotation before inference | File existence, dimensions, corner/mask consistency and conversion formula checked automatically. |
