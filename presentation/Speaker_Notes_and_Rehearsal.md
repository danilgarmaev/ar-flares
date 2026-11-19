# ICANN 2026 speaker guide

Paper 750 · Danil Garmaev and Jean Meunier

14 main slides · target 14 minutes 25 seconds · 35 seconds of buffer · 5 minutes for questions.

The same notes are embedded in the PowerPoint Notes pane. Backup slides are hidden from normal slideshow progression; access them through PowerPoint’s slide navigator.

## 01. Solar Flare Prediction from Magnetograms (0:00–0:20)

Hello, I’m Danil Garmaev, a master’s student at Université de Montréal. This is work I did with Jean Meunier on predicting solar flares. Our question is quite practical: what images should we give a model to help it make that prediction?

**Delivery:** Start slowly. Give the audience a moment to settle before moving to the background.

**How to say it:** Garmaev and Meunier: use your usual pronunciation of your names. No need to read the full subtitle.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Figure 1; SDO/HMI imagery credited in the paper.

## 02. Solar flares: what they are and why we predict them (0:20–1:50)

Before I get into the model, let me explain what we are trying to predict. A solar flare is a sudden release of magnetic energy in the Sun’s atmosphere. We observe it as a burst of radiation, like the bright region in this image. Strong flares can disrupt radio communication on the sunlit side of Earth. Forecasting them helps people prepare for those disruptions.

The letters along the bottom describe flare strength, measured using X-rays. The scale runs from A to X. Each step has a threshold ten times higher than the previous one: for example, M-one is ten times as intense as C-one in that measurement. We focus on two questions: will there be a C-class flare or stronger, and will there be an M-class flare or stronger? The second group contains the stronger, rarer events. In both cases, we want to predict what happens in the next twenty-four hours, before the flare occurs.

**Delivery:** Point to the bright flare, then follow the letters from left to right. Explain the two tasks once; after this, “C-class or stronger” and “M-class or stronger” are enough.

**How to say it:** A, B, C, M, X: “ay, bee, see, em, ex.” ≥C: “C-class or stronger.” ≥M: “M-class or stronger.” M1 and C1: “M-one” and “C-one.”

**If asked:** Flare classes measure peak GOES soft X-ray flux, not total energy. M1 has ten times the peak flux of C1; comparing arbitrary M and C events is not a fixed factor of ten. A flare is a burst of radiation; a CME is an ejection of plasma and magnetic field. They can occur together, but the task here is flare prediction, not CME or geomagnetic-storm forecasting. Radiation reaches Earth as we observe the flare, so advance prediction differs from detecting a flare that has already occurred.

**Source:** Garmaev, master’s thesis, §1.1 and Table 1.1. NOAA: https://www.spaceweather.gov/phenomena/solar-flares-radio-blackouts . NASA: https://science.nasa.gov/sun/solar-storms-and-flares/ . Image: https://www.spaceweather.gov/sites/default/files/FlareImage.jpg

## 03. What the model sees (1:50–3:00)

To look for warning signs, we use images of the Sun’s magnetic field. These are called magnetograms. They are different from the flare image we just saw: black and white show magnetic fields pointing in opposite directions.

We focus on active regions, which are areas where the magnetic field is concentrated. The box on the full image shows one of these regions, and the smaller image shows the cutout we give to the model. The two views use different contrast settings, which is why they look different.

The model sees either one image or a series of images from the same region. It then estimates whether a flare above our chosen class will happen during the following day. So we are looking for signs before an event, rather than recognizing a flare that is already happening.

**Delivery:** Point to the full disk and then the cutout. Make the distinction between the magnetic-field image and the earlier flare image explicit.

**How to say it:** Magnetogram: “mag-NEE-toh-gram.” AR means “active region”; use the words in the talk. 24 h: “the next twenty-four hours” or “the following day.”

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §1–2, Fig. 1. Boucheron et al., Scientific Data 10, 825 (2023), doi:10.1038/s41597-023-02628-8.

## 04. Four questions about the input (3:00–3:50)

We looked at four choices. First, how much detail do we need in each image? Second, does it matter where the region is on the visible Sun? Third, how far back should the sequence of images go? And fourth, how many images do we need within that period?

The idea is to study these choices using the same model family and a similar training procedure. That makes the comparisons easier to interpret. I’ll go through the spatial questions first, then spend more time on the sequence results, because that is where we found the clearest differences.

**Delivery:** Treat this as a short roadmap. Do not read all settings from the four boxes.

**How to say it:** 224 × 224: “two twenty-four by two twenty-four pixels,” if needed. “Image size” is usually enough here.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.1–4.7.

## 05. Data and evaluation (3:50–4:55)

The images come from the Solar Dynamics Observatory. We use an existing dataset with separate training, validation and test sets. Each active region appears in only one set, so we test on regions the model has not seen during training.

Most examples are not followed by a strong flare. For the M-class task, only about three in a hundred training examples are positive. Always predicting ‘no flare’ could therefore look accurate without helping us.

Our main score is TSS. It rewards finding flares and penalizes false alarms among non-flaring examples. One is perfect; zero means no skill by this measure. We choose the decision threshold using validation data.

**Delivery:** Leave the exact dataset counts on screen. Point to the short orange bar to explain why ordinary accuracy can mislead.

**How to say it:** TSS: say the letters, “tee ess ess.” On first use: “the True Skill Statistic, or TSS.” 3.1%: “about three percent,” not the full decimal.

**If asked:** TSS = TP/(TP+FN) − FP/(FP+TN). A trivial all-negative classifier has TSS 0. Validation threshold selection targets TSS with recall ≥0.5 and precision ≥0.15 constraints.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §2–3; dataset image counts and training positive fractions.

## 06. The prediction model (4:55–5:45)

We use MViT, a Multiscale Vision Transformer. It looks at smaller patterns and the larger structure of an active region. It has an image version and a sequence version, so we can study both types of input within the same model family.

We start from pretrained models and fine-tune them on magnetograms. We repeat each setting ten times with different random seeds. The error bars show how much the score varies between those runs.

The detailed training settings are in the backup slides. Here, we keep the setup consistent and focus on what information the model receives.

**Delivery:** Spend more time on what the model receives than on the optimizer. Explain the error bars once here.

**How to say it:** MViT: “em-vit,” followed once by “Multiscale Vision Transformer.” “Ten training runs” is clearer than repeatedly saying “ten random seeds.”

**If asked:** There is no claim of a new MViT architecture. The controlled factors are input choices within the shared model family and fine-tuning protocol.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.1–4.7, Table 1. Fan et al., Multiscale Vision Transformers, ICCV 2021, pp. 6824–6835.

## 07. Image resolution (5:45–6:40)

First, we reduce the detail in each image. These examples show what remains as the image becomes coarser. We then resize it back to the model’s expected input size.

Notice that the scores stay fairly close. For C-class or stronger flares, there is little change. For M-class or stronger, the original resolution scores highest on average, but the differences are not statistically clear.

The model can still use information visible at a coarser scale. That does not mean fine detail never matters. And because we resize the images back up, this experiment tests image information, not processing speed.

**Delivery:** Describe the mostly flat trend. There is no need to read any TSS values on this slide.

**How to say it:** 56 × 56: “fifty-six by fifty-six.” 112 × 112: “one twelve by one twelve.” “Original, half-size and quarter-size images” is also fine.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.3, §5.1, Table 2. Visual pipeline applied to repository frame_15 (AR 1339).

## 08. Position on the solar disk (6:40–7:50)

The Sun is round, so our view of a region changes as it moves across the visible disk. Near the center, we see it more directly. Closer to the edge, the view is more distorted. These figures show the region locations and the two ranges used in the study.

We asked two questions. Does the same model perform differently near the center and farther out? And does training only on central regions help?

On the right, the model ranks the examples somewhat better in the central region for both tasks. This score is ROC-AUC: higher means better separation between flaring and non-flaring examples. Training only on central regions gave mixed results, though. It did not help both tasks. So location is worth checking, but simply removing the outer regions is not an automatic improvement.

**Delivery:** Explain the restored disk diagrams before discussing the chart. Blue dots are region centers, not flare events. Keep numerical specialization results in backup B03.

**How to say it:** ROC-AUC: “rock A-U-C.” Longitude: “LON-jih-tood.” ±30°: “within thirty degrees of the center.” Do not read the absolute-value notation aloud.

**If asked:** Preparation audit found that saved A2 evaluations use Best_threshold from test evaluation, while the paper describes validation-selected thresholds. This deck therefore omits longitude TSS/F1 and associated p-values. Confirm or reevaluate A2 with validation-selected thresholds before making those claims; no new training or reevaluation was performed for this deck.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.4, §5.2, Table 3; thesis Fig. 2.2; restored latlon_30.png and latlon_60.png; repository A2 evaluation records.

## 09. Time between images and number of images (7:50–8:55)

For the sequence experiments, it helps to separate two ideas. One is how much time the images cover. The other is how many images we use.

At the top, every sequence has sixteen images and ends at the moment we make the prediction. When the images are farther apart, we can look further into the past. The longest setting covers forty-five hours, which is almost two days.

At the bottom, we keep those same first and last times, but use fewer images between them. This lets us ask whether we need all sixteen observations to make a useful prediction.

The forty-five hours are the past that the model can see. The twenty-four hours are the future we want to predict. Those two periods have different roles.

**Delivery:** Trace the right-aligned endpoints first. Use your pointer to show “more time covered” versus “fewer dots.”

**How to say it:** 45 h: “forty-five hours,” or “almost two days” when explaining the idea. Cadence means “time between images”; prefer that phrase.

**If asked:** The implemented T8 subset uses lookbacks [45,39,33,27,18,12,6,0] h; it is approximately spaced on the available grid, not eight perfectly equal intervals.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.5–4.6; repository frame-count configs: T4 offsets [-225,-150,-75,0] in 12-minute steps.

## 10. How much past information helps? (8:55–10:05)

Here, we keep sixteen images and increase the time they cover. The green line shows the clearest result: performance improves as we give the model a longer view of the region’s past. For C-class or stronger flares, TSS goes from about zero point six seven to zero point seven nine.

The improvement is consistent enough across the repeated runs to be statistically significant. Another ranking score improves as well. For the stronger M-class task, the average moves in the same direction, but the variation is larger, so that result is less certain.

Our interpretation is that changes over a longer period can help the prediction. There is still a limitation: asking for longer sequences changes which examples are available. A follow-up should compare the same regions at the same prediction times across all settings.

**Delivery:** This is the one result where reading two rounded values helps. Then point to the trend and the larger orange error bars. Leave p-values to questions.

**How to say it:** 0.674 → 0.789: say “about zero point six seven to zero point seven nine.” These are TSS scores, not percentages of correct predictions.

**If asked:** Cadence and span change together in this experiment. The cohort-count backup shows sample attrition. These results support, but do not prove a purely causal temporal-span explanation.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.3, Table 4. Repository aggregation of fixed_TSS reproduced all six means and sample SDs to 3 decimals.

## 11. Do we need all sixteen images? (10:05–11:15)

Next, we keep the same period of almost two days and try fewer images. The four images on the left are an example: the first observation, two in between, and the most recent one.

The result is that using four or eight images gives scores that are close to, and sometimes slightly above, the sixteen-image result. The differences are not statistically clear, so I would not say that four is better. What we can say is that using fewer images still works reasonably well in this comparison.

That is useful because it suggests we may not need every observation to capture the changes that help prediction. Four instead of sixteen means three quarters fewer input images. We would still need to measure the actual speed and memory savings. There is also a trade-off for the stronger flares, which I’ll show next.

**Delivery:** Point to the four real observations. Say “the same period” or “almost two days,” rather than “preserving a 45-hour history.”

**How to say it:** 75%: “seventy-five percent” or, more naturally here, “three quarters fewer images.” Do not read all six plotted TSS values.

**If asked:** 75% = (16−4)/16. Lack of statistical significance is not an equivalence or non-inferiority test. No measured runtime or memory savings are claimed.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.4, Table 5. AR1339 frames copied unchanged from data/wds_out/train/shard-000015.tar through shard-000017.tar.

## 12. Understanding the PR-AUC score (11:15–12:50)

Before interpreting this plot, I want to explain the scale. PR-AUC looks at the balance between two questions: when we give a warning, how often is it followed by a flare? And how many of the flares do we actually find? It summarizes that balance across different warning thresholds.

A score around zero point three six might look disappointing if we read it as thirty-six percent accuracy. But that is not what it means. In these test examples, only about three in a hundred are followed by an M-class flare or stronger. Random ranking therefore has a reference level of about zero point zero three, shown near the bottom. Our scores are clearly above that reference.

There is still room to improve. With fewer images, this score drops a little, even though TSS was slightly higher. So the model is learning useful information, but there is a trade-off. The preferred setting depends on which kinds of errors matter most for the application.

**Delivery:** Point to the random-ranking reference before the model points. Do not say “thirteen times more accurate,” and do not treat PR-AUC as precision at a single threshold.

**How to say it:** PR-AUC: “P-R A-U-C.” On first use: “the area under the precision–recall curve.” 0.36: “zero point three six.” 0.03: “zero point zero three.”

**If asked:** TSS is not alert precision. Calibration, deployment prevalence, costs of missed events and false alarms, and a pre-specified operating policy would matter for operational deployment.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.4, Table 5; thesis §1.2; pr_baseline_audit.json. The exact random-ranking reference is 73/2702, about 0.027, in all 30 M-task frame-count runs.

## 13. Limitations and next steps (12:50–13:45)

There are three things I would test next. First, compare the different time windows using exactly the same prediction examples. That would help separate the value of extra past information from changes in the data.

Second, measure speed and memory use with fewer images. We need those measurements before claiming a practical saving.

Third, test on other regions and time periods to see whether the results carry over. I would also compare the sequence model with a single-image model using the last image of those same sequences. That would give us a fairer comparison.

**Delivery:** Present three concrete next steps. Keep additional limitations in the backup notes.

**How to say it:** “The same examples” is clearer than “matched cohorts.” “Other time periods” is enough unless someone asks specifically about solar cycles.

**If asked:** A stronger static baseline uses the final frame of the same sequence-derived examples with matched labels, splits and training budget. A solar-cycle holdout probes a different generalization question from the current AR-disjoint split.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §6; presentation/pr_baseline_audit.json.

## 14. Main findings (13:45–14:25)

Looking further into the past helped, especially for C-class or stronger flares. Once that longer period was covered, fewer images still gave similar scores.

The spatial results were more mixed: reducing image detail had a limited effect, and central-only training did not help both tasks.

The main lesson is to consider the input alongside the model: what information does it need, and how much of that information is enough? Thank you.

**Delivery:** Finish here at roughly 14 minutes 25 seconds. Keep this slide visible for questions; use the hidden backups only when needed.

**How to say it:** No numbers need to be read on the closing slide.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §7; Tables 2–5.

## 15. B01 — Training and evaluation details (Q&A backup)

Use this slide only for a training-protocol question. The learning rates differ between the pretrained backbone and classification head. Geometric transforms and polarity inversion are consistent across a sequence; noise can vary across frames. Early stopping and checkpoint selection use validation information. Repository configs also include negative subsampling, so do not interpret the full image-benchmark counts as the number of effective training examples for every run.

**If asked:** The code can fall back to unconstrained TSS if no threshold satisfies the constraints. The constraints describe the selection target, not a guarantee about test precision or recall.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §3, §4.2, Table 1; repository saved config.json files.

## 16. B02 — Resolution study: all reported metrics (Q&A backup)

The full-resolution M-task setting has the highest means on all four metrics, and its PR-AUC advantage over the coarsest setting is more noticeable than the TSS difference. Do not generalize a non-significant TSS test to every metric. The resolution operation changes the information available before resizing; the final network dimensions remain fixed.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 2.

## 17. B03 — Longitude: threshold-free ranking metrics (Q&A backup)

Compare inner versus outer only within the same training setup. Compare specialization using the same central test subset. The central-only model is also evaluated on the outer bucket, but those numbers do not establish that central-only training is a general improvement. PR-AUC depends on the positive fraction of the evaluated population. Longitude TSS and F1 are intentionally omitted pending resolution of the threshold-provenance discrepancy found in the repository. The analysis memo has concrete evidence and the recommended reevaluation.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 3; A2 JSON evaluation records; Analysis_and_Verification.md.

## 18. B04 — Cadence study: full results (Q&A backup)

For C-or-stronger, all four mean metrics improve across the three tested spans. For M-or-stronger, TSS and ROC-AUC peak at 180 minutes, but PR-AUC peaks at 60 minutes. This reinforces the metric-dependent interpretation. The significant tests reported here concern TSS only. Do not imply that all changes in AUC, PR-AUC or F1 were tested or found significant.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 4, §5.3.

## 19. B05 — Frame-count study: full results (Q&A backup)

This table supports a viable sparse regime rather than a unique optimum. For C-or-stronger, TSS means are close. For M-or-stronger, the larger TSS means with sparse inputs coexist with lower mean PR-AUC and F1. The eight-frame implementation selects an approximately spaced subset from the three-hour grid, with a nine-hour gap in the middle. Do not claim mathematically uniform spacing for that configuration. The endpoints remain exactly forty-five hours apart.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 5; saved config.json seq_offsets fields and scripts/run_mvit_framecount_unique_5seed_a100.slurm.

## 20. B06 — Test example counts (Q&A backup)

These counts come directly from the evaluated test predictions in the saved run metrics. Longer sequences leave fewer usable test examples. The forty-five-hour setting has 73 M-or-stronger positives out of 2,702 examples, which gives the approximately 0.03 random-ranking reference used on the PR-AUC slide. The four-, eight- and sixteen-frame settings all have these same class counts across their ten runs. Equal counts alone do not prove that all example identities match. These are correlated sequence examples, not counts of independent flare events.

**Source:** presentation/pr_baseline_audit.json, using the run paths saved in temporal_results_audit.json. Earlier cached sequence_size_counts differed and are not used here.

## 21. B07 — Q&A preparation (Q&A backup)

Answer the direct question first, then one supporting fact. These prompts are a rehearsal aid; normally keep the final takeaway slide visible and navigate directly to the relevant evidence table.

**If asked:** Why MViT? It supports multiscale image and temporal representations in a common family; architecture novelty is not the claim. Why ten seeds? To quantify optimization variability; this does not replace AR-level or temporal uncertainty analysis. Why TSS? It balances hit rate and false-positive rate under class imbalance, but it is not sufficient for an operational policy. Is 180 minutes optimal? It is the strongest mean TSS/AUC among the paper’s three 16-frame settings, not a global optimum. Did lower resolution save compute? No measured backbone speedup: inputs were returned to 224×224. Are four frames equivalent to sixteen? No formal equivalence test; they are competitive in these results. Is the threshold tuned on test? The reported main TSS uses a validation-selected threshold; repository metrics also contain other threshold variants, which must not be substituted. Is a 3D model proven better than a 2D model? No: use a matched last-frame baseline before making that claim.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §6.

## 22. B08 — References and visual provenance (Q&A backup)

These references are copied from the accepted paper. The main charts are newly drawn for readability using the paper’s numeric results. The magnetograms are actual data, not generated scientific images. The cohort analysis is explicitly identified as supplementary repository material.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. references 1–3 and 17; Garmaev_Danil_2026_memoire.pdf §1.1–1.2 and Fig. 2.2; https://www.spaceweather.gov/phenomena/solar-flares-radio-blackouts ; https://science.nasa.gov/sun/solar-storms-and-flares/ ; presentation/assets/frame_provenance.json.
