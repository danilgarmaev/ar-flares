# Presentation analysis and verification

Prepared for Danil Garmaev, ICANN 2026, paper 750.

## Deliverables and intended use

- `ICANN_2026_Solar_Flare_Prediction.pptx`: editable 16:9 PowerPoint, 14 main slides, 8 hidden backup slides, notes on every slide.
- `ICANN_2026_Solar_Flare_Prediction_preview.pdf`: vector preview generated from the same slide geometry and content. This is a companion rendering, not an export from Microsoft PowerPoint.
- `Speaker_Notes_and_Rehearsal.md` and `Speaker_Notes_and_Rehearsal.pdf`: speaking script, timing, transitions, sources and Q&A guidance. The same content is embedded in PowerPoint's Notes pane.
- `preview/`: images for inspecting individual slides and contact sheets.
- `build_presentation.py`: source for the deck and notes.
- `*_audit.json`: supporting repository checks.

The revised delivery target is **14 minutes 25 seconds**, leaving about 35 seconds within the allotted 15-minute talk. The script uses plain spoken explanations and leaves most exact values to the plots and backups. Speak the script naturally and allow time to point to plots. The question period is a separate five minutes. Hidden backups do not consume normal presentation time; open them through PowerPoint's slide navigator when needed.

## Scientific story

The strongest claim is specific: within the three tested 16-frame configurations, the ≥C task improves with longer temporal coverage. Mean TSS rises from 0.674 to 0.789 and ROC-AUC from 0.910 to 0.958. The paper reports significant pairwise TSS differences for ≥C. The ≥M task shows a similar mean trend with larger uncertainty, without significant TSS differences.

The frame-count experiment adds the practical conclusion: a 45-hour history can remain useful with four or eight observations. Four instead of sixteen frames is exactly 75% fewer **input frames**. It is not evidence of 75% lower runtime, memory, cost or energy. The TSS comparisons do not establish statistical equivalence or a unique optimum. For ≥M, the sparse settings have higher mean TSS but lower mean PR-AUC and F1 than sixteen frames.

The resolution result concerns retained information. Downsampling is followed by resizing back to 224 × 224; network input dimensions remain constant. The deck therefore makes no spatial-inference-speedup claim.

The longitude slides use descriptive ROC-AUC and PR-AUC results, for the reason below.

## Issue to resolve before discussing longitude TSS

**The saved longitude evaluation pipeline does not match the paper's stated threshold provenance.**

The accepted paper describes reported TSS as evaluated with validation-selected thresholds. However:

1. `classifier_NN/legacy/run_experiments_A2.py`, function `_try_read_best_threshold_from_exp_dir`, prioritizes `Best_threshold` from the experiment's `metrics.json`. The evaluation's default threshold logic uses that value unless explicitly overridden. The central-only evaluation path also reads that value.
2. `classifier_NN/metrics.py`, function `evaluate_model`, computes `Best_threshold` by searching the labels and probabilities of the evaluated dataset. When called on the test loader, that is a test-optimized threshold. It separately records the supplied validation threshold as `fixed_threshold` and scores as `fixed_TSS` and `fixed_F1`.
3. All **20** saved June 11 baseline A2 evaluation records used by the plotting script have a threshold exactly equal to their source experiment's `Best_threshold`. **None** matches that experiment's `fixed_threshold`. The per-record comparison is saved in `a2_threshold_audit.json`.

For example, the ≥C seed-4 source experiment is:

`results/2026-04-27 19:26:57_A1-224_mvit_seed4_screen2dres_mvitv1_notemporal_cplus_dsnone300k_wbce_ftv2_aug/metrics.json`

It records `Best_threshold = 0.003250761520117521` and `fixed_threshold = 0.003`. The corresponding A2 record uses `0.003250761520117521`. Other seeds also differ; the issue should not be dismissed based on one numerically small difference.

**How the deck handles this:** longitude TSS, F1 and their p-values are omitted. Main slide 8 and backup B03 show threshold-free AUC evidence from Table 3, explicitly as descriptive comparisons. The talk's principal resolution and temporal TSS plots retain the paper's validation-threshold values.

**Recommended scientific follow-up:** reevaluate each original longitude checkpoint on the same subsets using its recorded validation-selected threshold, including the central-only models, then reaggregate the paired-seed comparisons. Prefer `val_threshold_for_test` or the equivalent `fixed_threshold`, with explicit validation provenance. The current central-only helper needs care because its threshold inference is separate from the baseline `--threshold` argument. No training, checkpoint alteration or expensive model reevaluation was performed as part of preparing the slides. Confirm corrected results with the coauthor before restoring longitude TSS significance claims or revising manuscript claims.

## Repository checks completed

- All six resolution-study TSS means and sample standard deviations were reproduced to three decimals from ten seeds per task/resolution using `fixed_TSS`. The selected result paths are in `resolution_results_audit.json`.
- The cadence audit collected ten augmented MViT runs per task and cadence for 60, 120 and 180 minutes, using the validation-threshold `fixed_TSS` field.
- All Table 4 means and sample standard deviations were reproduced to three decimals for **TSS, ROC-AUC, PR-AUC and fixed-threshold F1**.
- The four- and eight-frame, 180-minute-base-grid runs also have ten seeds per task. Together with the 16-frame baseline, they reproduce the full Table 5 means and sample standard deviations for those four metrics to three decimals.
- Details and the selected metric-file paths are in `temporal_results_audit.json`. Test-optimized `F1` is kept in that audit only to distinguish it from the correct `fixed_F1`; the presentation uses `fixed_F1` values.
- The longitude threshold audit compared all 20 baseline A2 records with their original experiment metrics.
- The T4 configuration uses offsets `[-225, -150, -75, 0]` in 12-minute steps: lookbacks `[45, 30, 15, 0]` hours.
- The actual T8 subset uses lookbacks `[45, 39, 33, 27, 18, 12, 6, 0]` hours. It spans the same endpoints but is not perfectly evenly spaced. The backup and notes use this precise description.
- The 16 example images were copied from the actual WebDataset shards for AR 1339, starting 2011-11-04 06:36 TAI at three-hour spacing. `assets/frame_provenance.json` records the original names. They are illustrations of the inputs, not claimed flare-event explanations or saliency maps.

## Additional analysis included in the deck

The revision checks the actual evaluated test counts in the saved M-task metric files. For 60/120/180-minute cadences there are **14,551/5,475/2,702** evaluated test examples, with **494/165/73** positives. Each count agrees across the ten seeds for its setting. Backup B06 now uses these directly checked counts. The previous version used a cached dataset-size audit with larger counts; that snapshot is preserved in `archive_v1` but is no longer used in the talk.

For the fixed 45-hour frame-count comparison, all thirty M-task runs (4, 8 and 16 frames, ten seeds each) have 73 positives out of 2,702 evaluated examples. The reference precision of uninformative ranking is therefore **73/2702 ≈ 0.027**, shown as approximately 0.03 on slide 12. This is an evaluated-test prevalence, not the 3.1% training prevalence. `pr_baseline_audit.json` records the checked groups and counts; source run paths remain in `temporal_results_audit.json`.

The PR-AUC slide uses a full 0–1 axis, three paper-reported means (0.393, 0.389, 0.359), and the prevalence reference. It explains that these are well above the uninformative-ranking reference without treating PR-AUC as accuracy, a single-threshold precision, a probability of success, or a multiplier of forecasting quality. The flat prevalence line is the conventional no-skill reference; a finite randomly ordered test list need not have area exactly equal to prevalence. Remaining tradeoffs and uncertainty are described in notes and backup B05.

This reinforces a limitation of causal interpretation: extending required history changes sample eligibility and class composition. A useful next experiment would hold prediction endpoints fixed across cadences. Similarly, a fair static-versus-sequence comparison should use the final frames of the same sequence-derived examples.

## Statistical and presentation choices

- Error bars are **±1 standard deviation across ten random seeds**, not confidence intervals and not independent-solar-cycle uncertainty.
- The cadence significance statements are the paper's paired-test results; they were not independently recomputed for this presentation.
- No new architecture, new training result, state-of-the-art ranking or deployment-readiness claim is introduced.
- The PR-AUC slide uses a single 0–1 scale and an explicit random-ranking reference. Values are descriptive means; standard deviations remain in backup B05.
- The accepted paper remains the source for experimental results. Added background uses the attached master’s thesis (§1.1–1.2) and official NOAA/NASA explanations. Images are credited; no AI-generated scientific data are used.
- The accepted author proof is the authority for published values. Older repository manuscript drafts and exploratory settings outside the accepted experiments are not substituted for the paper.

## File and visual checks

The PowerPoint package was reopened with python-pptx, all package XML was parsed, the ZIP passed its integrity check, and every slide has notes. There are 14 visible main slides and eight hidden backup slides. Shape-boundary and font-metric checks found no out-of-bounds objects or text-height overflow. All companion-preview pages were rendered and inspected; an unsupported superscript glyph was replaced with portable scientific notation.

Microsoft PowerPoint and LibreOffice were not available in this environment. The companion PDF uses the same geometry with Liberation Sans, metrically compatible with the Arial specified in the PowerPoint. A final rehearsal on the conference laptop should confirm actual PowerPoint rendering and Presenter View. No animations, video, live internet or linked external images are required.

The repository's existing code, results, checkpoints and manuscript files were left untouched. New artifacts are confined to `presentation/`.

## September 14 revision: clearer context and spoken delivery

- Added an opening slide defining a solar flare, explaining the radio-communication motivation, and introducing A/B/C/M/X classes. The spoken explanation distinguishes the next-day prediction from detecting an ongoing flare. M1 versus C1 is used for the precise factor-of-ten example.
- Rewrote all main-slide scripts and titles. The notes explain the point of each comparison rather than reciting its values. Dense methodological details remain under “If asked.”
- Restored the disk-position diagrams from `figures/figures_pdf/latlon_60.png` and `latlon_30.png`, used as thesis Fig. 2.2. Blue dots represent active-region centers; the diagrams are not maps of observed flares.
- Replaced “preserve a useful 45-hour history” with “Do we need all sixteen images?” and a description of four images spread over the same period.
- Added “How to say it” notes on every main slide and a separate `Pronunciation_and_Delivery.md` cue sheet.
- The introductory photo is an actual NASA image hosted by NOAA, downloaded unchanged to `assets/solar_flare_nasa_noaa.jpg`. No date or instrument is inferred beyond the credit supplied by NOAA.
- The earlier threshold-provenance issue remains unresolved; restored thesis figures do not change that finding. Longitude results continue to use threshold-free AUC, not the disputed TSS claims.

Background sources checked on September 14, 2026:

- [NOAA — Solar Flares (Radio Blackouts)](https://www.spaceweather.gov/phenomena/solar-flares-radio-blackouts)
- [NASA — Solar Storms and Flares](https://science.nasa.gov/sun/solar-storms-and-flares/)
- [NASA image hosted by NOAA](https://www.spaceweather.gov/sites/default/files/FlareImage.jpg)
- `Garmaev_Danil_2026_memoire.pdf`, §1.1–1.2, Table 1.1 and Fig. 2.2.

The previous deck, notes and builder are preserved in `archive_v1/`. The updated files retain the original filenames for convenience.
