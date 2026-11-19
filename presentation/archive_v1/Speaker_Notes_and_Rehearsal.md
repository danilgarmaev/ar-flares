# ICANN 2026 speaker guide

Paper 750 · Danil Garmaev and Jean Meunier

13 main slides · target 14 minutes · 1 minute buffer within the 15-minute slot · 5 minutes for questions.

The same notes are embedded in the PowerPoint Notes pane. Backup slides are hidden from normal slideshow progression; access them through PowerPoint’s slide navigator.

## 01. Solar Flare Prediction from Magnetograms (0:00–0:30)

Good [morning/afternoon]. I’m Danil Garmaev from Université de Montréal. This work with Jean Meunier asks a practical question about solar flare prediction: what information should we give a deep learning model? We study spatial detail, position on the solar disk, and temporal sampling. Our clearest finding is that covering a longer history of an active region can be more useful than giving the model many closely spaced observations.

**Delivery:** Pause after the central question. The bracketed greeting is for rehearsal only.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Figure 1; SDO/HMI imagery credited in the paper.

## 02. Forecast the next 24 hours from magnetic structure (0:30–1:30)

Solar activity can affect satellites, communications and electrical infrastructure, so forecasting is useful before an event happens. The input here is a magnetogram: an image of the line-of-sight magnetic field. Black and white show opposite polarities; this is not an ordinary photograph of the Sun. The box identifies an active region, and the cutout is the image used by the learning pipeline. The full disk and cutout have different display scaling, so their contrasts should not be compared quantitatively. We ask two binary questions: will a C-class-or-stronger flare occur in the next 24 hours, and will an M-class-or-stronger flare occur? These are separate tasks. We can give the model a single observation or several observations of the same region leading up to prediction time.

**Delivery:** Point to the boxed region, then the cutout. Establish the forecast horizon before discussing the lookback window.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §1–2, Fig. 1. Boucheron et al., Scientific Data 10, 825 (2023), doi:10.1038/s41597-023-02628-8.

## 03. Four input choices, one experimental framework (1:30–2:25)

Our contribution is a controlled study of input design. First, we reduce the spatial detail in an active-region image. Second, we examine performance at different solar longitudes and test central-region specialization. Third, we hold the number of frames at sixteen and increase the temporal span. Fourth, we keep the span at forty-five hours and reduce the number of frames. We use the MViT architecture family throughout, with a common fine-tuning protocol. That lets us focus on practical decisions about the observations. The controls differ by experiment: for example, changing the required temporal history can also change which sequences are eligible. I will return to that limitation when interpreting the temporal results.

**Delivery:** Use this as the roadmap. Do not spend time enumerating alternative architectures.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.1–4.7.

## 04. Separate active regions; evaluate beyond accuracy (2:25–3:35)

We use the reduced-resolution Boucheron dataset, built from SDO/HMI observations between 2010 and 2018. These are the image-benchmark counts. The predefined split separates active-region identities: a region in training does not also appear in validation or test. This reduces active-region leakage, although it is not a prospective solar-cycle holdout. The two tasks are imbalanced. About twenty percent of training images are positive for the C-or-stronger task, but only about three percent for M-or-stronger. Accuracy alone would therefore be misleading. Our primary score is the True Skill Statistic: the hit rate minus the false-positive rate. We choose the operating threshold on validation data and then evaluate on test data. We also report ROC-AUC, PR-AUC and F1. The sequence experiments use eligible temporal samples, so their counts are smaller than the image counts shown here.

**Delivery:** Explain the denominator distinction: false-positive rate is not the fraction of issued alerts that are false.

**If asked:** TSS = TP/(TP+FN) − FP/(FP+TN). A trivial all-negative classifier has TSS 0. Validation threshold selection targets TSS with recall ≥0.5 and precision ≥0.15 constraints.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §2–3; dataset image counts and training positive fractions.

## 05. Keep learning consistent while changing the inputs (3:35–4:50)

MViT provides a multiscale feature hierarchy, which is useful here because the magnetic information may occur at both local and active-region scales. We use a two-dimensional MViT for images and an MViT sequence model for temporal inputs. Both start from pretrained models. Fine-tuning uses AdamW, weighted binary cross-entropy, and a cosine learning-rate schedule. The backbone learning rate is smaller than the classification head learning rate. We use augmentation during training, including flips, limited rotations and polarity inversion. For a sequence, geometric transforms and polarity inversion are shared across frames so that augmentation does not create inconsistent motion. The maximum budget is fifty epochs, with validation-based model selection and early stopping. Each reported configuration has ten random seeds. The plotted error bars summarize standard deviation across those seeds; they represent training variability, not uncertainty across independent solar cycles.

**Delivery:** The architecture schematic is conceptual; it does not specify exact layer counts. Save hyperparameters for backup B01.

**If asked:** There is no claim of a new MViT architecture. The controlled factors are input choices within the shared model family and fine-tuning protocol.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.1–4.7, Table 1. Fan et al., Multiscale Vision Transformers, ICCV 2021, pp. 6824–6835.

## 06. Much of the predictive signal survives coarser images (4:50–5:55)

Here we reduce the available spatial information while keeping the backbone input dimensions unchanged. Images are downsampled with nearest-neighbor interpolation and then returned to 224 by 224 with bilinear interpolation. These real magnetograms illustrate the operation. For C-or-stronger prediction, mean TSS stays close to 0.53 across all three resolutions. For M-or-stronger, full resolution has the highest mean TSS, but the lower-resolution settings remain fairly close. The paper reports no significant pairwise TSS differences. This supports a cautious conclusion: coarse images preserve much of the information used by this model. It does not prove that small-scale magnetic structure is unimportant, and it is not a runtime benchmark. Because every image is resized back to the same backbone size, we have not demonstrated an inference speedup from this spatial experiment.

**Delivery:** Point to image detail, then the modest TSS changes. Use “retains much of the signal,” not “equivalent.”

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.3, §5.1, Table 2. Visual pipeline applied to repository frame_15 (AR 1339).

## 07. Disk position matters; specialization is task-dependent (5:55–7:05)

We separate two questions here and use ROC-AUC, which does not require an operating threshold. On the left, the same model trained over the full longitude range is evaluated on inner and outer regions. The inner bucket has higher mean ROC-AUC for both tasks: 0.863 versus 0.843 for C-or-stronger, and 0.916 versus 0.887 for M-or-stronger. This is consistent with position-dependent reliability and projection effects, although different region populations can also contribute. On the right, we compare full-range training with central-only training while keeping the central test subset fixed. C-task mean AUC decreases from 0.863 to 0.852, while M-task mean AUC increases slightly from 0.916 to 0.920. These are descriptive comparisons, not AUC significance claims. The useful conclusion is to evaluate disk position explicitly: central-only specialization is not uniformly beneficial.

**Delivery:** Separate “where we test” from “where we train.” Use the threshold-free AUC evidence shown here. See Analysis_and_Verification.md before discussing longitude TSS significance.

**If asked:** Preparation audit found that saved A2 evaluations use Best_threshold from test evaluation, while the paper describes validation-selected thresholds. This deck therefore omits longitude TSS/F1 and associated p-values. Confirm or reevaluate A2 with validation-selected thresholds before making those claims; no new training or reevaluation was performed for this deck.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.4, §5.2, Table 3; repository A2 evaluation records.

## 08. Separate how far back we look from how often we sample (7:05–8:15)

This distinction is central to the talk. In the first experiment, every input has sixteen frames. A sixty-minute cadence covers fifteen hours; a hundred-and-twenty-minute cadence covers thirty hours; and a hundred-and-eighty-minute cadence covers forty-five hours. The span is the number of intervals, fifteen, multiplied by the interval length. All inputs end at prediction time. Increasing the interval here therefore also extends the available history. In the second experiment, we hold the forty-five-hour window fixed and reduce the observations. The four-frame input samples forty-five, thirty and fifteen hours before prediction, plus the current frame. This asks whether densely observing that window is necessary. The eight-frame configuration also spans the same endpoints using a subset of the available grid. In both experiments the label still concerns the following twenty-four hours: lookback and forecast horizon are different quantities.

**Delivery:** Trace the right-aligned endpoints, then the four orange observations. Keep the two time windows distinct.

**If asked:** The implemented T8 subset uses lookbacks [45,39,33,27,18,12,6,0] h; it is approximately spaced on the available grid, not eight perfectly equal intervals.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §4.5–4.6; repository frame-count configs: T4 offsets [-225,-150,-75,0] in 12-minute steps.

## 09. Longer history gives the clearest improvement for ≥C (8:15–9:45)

This is the strongest result in the study. With sixteen frames, C-or-stronger TSS rises from 0.674 for fifteen hours, to 0.742 for thirty hours, and 0.789 for forty-five hours. The endpoint difference is an absolute TSS gain of 0.115. All three pairwise TSS comparisons are significant in the paper’s paired tests. ROC-AUC also rises, from 0.910 to 0.958, so the improvement is not visible only at one operating threshold. For M-or-stronger, the means move in the same direction, but uncertainty is much larger and the TSS differences are not significant. We therefore emphasize the C-task evidence. The result is consistent with useful information in longer active-region evolution. It does not identify an optimal cadence in general: we are comparing three settings, and requiring a longer complete history changes the eligible sequence population. A follow-up using matched prediction endpoints would isolate the temporal effect more tightly.

**Delivery:** Give the audience time to read the teal progression. Say “0.115 absolute TSS,” not “11.5 percent more accurate.”

**If asked:** Cadence and span change together in this experiment. The cohort-count backup shows sample attrition. These results support, but do not prove a purely causal temporal-span explanation.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.3, Table 4. Repository aggregation of fixed_TSS reproduced all six means and sample SDs to 3 decimals.

## 10. Four observations can preserve a useful 45-hour history (9:45–11:10)

Now we keep the forty-five-hour span and reduce the frame count. The images on the left are actual observations from the repository, illustrating the four timestamps. They are an input example, not an explanation of a particular model prediction. On the right, C-or-stronger TSS is 0.789 with sixteen frames, 0.802 with eight, and 0.805 with four. For M-or-stronger, the means are 0.706, 0.751 and 0.752. The TSS differences are not significant, so I would describe the sparse settings as competitive, rather than claim four frames are better. The practical result is that four frames can preserve useful information across the long window while reducing the number of input frames by seventy-five percent. That percentage describes the input count. We did not benchmark wall-clock latency, memory or energy, and their savings need not scale linearly with frame count.

**Delivery:** Keep attention on preserved temporal endpoints and competitive scores. Do not imply formal statistical equivalence.

**If asked:** 75% = (16−4)/16. Lack of statistical significance is not an equivalence or non-inferiority test. No measured runtime or memory savings are claimed.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.4, Table 5. AR1339 frames copied unchanged from data/wds_out/train/shard-000015.tar through shard-000017.tar.

## 11. A higher mean TSS does not settle the operational choice (11:10–12:05)

The M-or-stronger task illustrates why we should look beyond a single score. Moving from sixteen to four frames raises mean TSS, but mean PR-AUC falls from 0.393 to 0.359 and mean F1 falls from 0.362 to 0.327. These panels show descriptive means on different vertical scales; they are not additional significance tests. TSS measures a balance of hit and false-positive rates at the chosen threshold. PR-AUC summarizes precision-recall ranking, and F1 emphasizes precision and recall at the operating point. An application concerned with missed strong flares or the reliability of alerts may therefore choose differently. Our conclusion is that sparse inputs are viable within this benchmark, with metric-dependent tradeoffs. We have not established a universal best frame count or an operational alert policy.

**Delivery:** Explicitly say the axes differ. Leave uncertainty and full tables for backup B05.

**If asked:** TSS is not alert precision. Calibration, deployment prevalence, costs of missed events and false alarms, and a pre-specified operating policy would matter for operational deployment.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §5.4, Table 5.

## 12. What the evidence supports—and what to test next (12:05–13:15)

There are three natural next steps. First, evaluate different temporal windows on a common set of prediction endpoints. Our current sequence selection requires complete history, and the number and composition of eligible examples change with cadence. A matched cohort would separate that selection effect from the value of a longer history. Second, benchmark the practical cost of the sparse configurations and test a pre-specified non-inferiority margin if the goal is to claim comparable performance. Third, validate outside this benchmark, including other solar-cycle intervals and independent magnetogram data. There are also domain issues such as overlapping active-region content and the use of natural-image or video pretraining. Finally, the image and sequence experiments do not form a fully matched architecture comparison. A clean baseline would train and evaluate a single-image model on the last frames of the exact same valid sequences.

**Delivery:** Deliver these as the next experimental design, not as an apology. Backup B06 quantifies the changing cohorts.

**If asked:** A stronger static baseline uses the final frame of the same sequence-derived examples with matched labels, splits and training budget. A solar-cycle holdout probes a different generalization question from the current AR-disjoint split.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §6; results/a3_sequence_size_counts.tsv.

## 13. Choose the history before increasing the sampling density (13:15–14:00)

I’ll leave you with three takeaways. First, extending the observed history gave our clearest improvement, especially for C-or-stronger prediction. Second, once a forty-five-hour window was available, four or eight frames remained competitive with sixteen. Third, input design needs to be evaluated directly: more pixels, more frames or more specialized training do not automatically produce a better predictor across tasks and metrics. The practical message is to establish the useful temporal coverage, then test how sparsely we can observe it. These are benchmark findings with a clear next step: matched cohorts and measured computational costs. Thank you. I’m happy to take questions.

**Delivery:** Finish here at approximately 14 minutes. Leave this slide visible during questions. Backup slides are hidden from normal slideshow navigation.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §7; Tables 2–5.

## 14. B01 — Training and evaluation details (Q&A backup)

Use this slide only for a training-protocol question. The learning rates differ between the pretrained backbone and classification head. Geometric transforms and polarity inversion are consistent across a sequence; noise can vary across frames. Early stopping and checkpoint selection use validation information. Repository configs also include negative subsampling, so do not interpret the full image-benchmark counts as the number of effective training examples for every run.

**If asked:** The code can fall back to unconstrained TSS if no threshold satisfies the constraints. The constraints describe the selection target, not a guarantee about test precision or recall.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §3, §4.2, Table 1; repository saved config.json files.

## 15. B02 — Resolution study: all reported metrics (Q&A backup)

The full-resolution M-task setting has the highest means on all four metrics, and its PR-AUC advantage over the coarsest setting is more noticeable than the TSS difference. Do not generalize a non-significant TSS test to every metric. The resolution operation changes the information available before resizing; the final network dimensions remain fixed.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 2.

## 16. B03 — Longitude: threshold-free ranking metrics (Q&A backup)

Compare inner versus outer only within the same training setup. Compare specialization using the same central test subset. The central-only model is also evaluated on the outer bucket, but those numbers do not establish that central-only training is a general improvement. PR-AUC depends on the positive fraction of the evaluated population. Longitude TSS and F1 are intentionally omitted pending resolution of the threshold-provenance discrepancy found in the repository. The analysis memo has concrete evidence and the recommended reevaluation.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 3; A2 JSON evaluation records; Analysis_and_Verification.md.

## 17. B04 — Cadence study: full results (Q&A backup)

For C-or-stronger, all four mean metrics improve across the three tested spans. For M-or-stronger, TSS and ROC-AUC peak at 180 minutes, but PR-AUC peaks at 60 minutes. This reinforces the metric-dependent interpretation. The significant tests reported here concern TSS only. Do not imply that all changes in AUC, PR-AUC or F1 were tested or found significant.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 4, §5.3.

## 18. B05 — Frame-count study: full results (Q&A backup)

This table supports a viable sparse regime rather than a unique optimum. For C-or-stronger, TSS means are close. For M-or-stronger, the larger TSS means with sparse inputs coexist with lower mean PR-AUC and F1. The eight-frame implementation selects an approximately spaced subset from the three-hour grid, with a nine-hour gap in the middle. Do not claim mathematically uniform spacing for that configuration. The endpoints remain exactly forty-five hours apart.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. Table 5; saved config.json seq_offsets fields and scripts/run_mvit_framecount_unique_5seed_a100.slurm.

## 19. B06 — Eligible sequence cohorts (Q&A backup)

This is supplementary analysis from the repository, not a table in the accepted paper. It documents how requiring longer histories changes sample eligibility. The cached audit reports 15,602 test sequences at sixty minutes versus 3,414 at a hundred and eighty minutes; M-or-stronger positives fall from 531 to 105. This is a reason to avoid attributing every observed gain solely to temporal span. These are sequence examples and can be correlated within active regions, not counts of independent flare events. The file is a cached snapshot; we did not regenerate it or train new models for the presentation.

**If asked:** Do not use these as exact counts of independent events. The 60-minute effective training count after negative subsampling is 120,203 in this snapshot, compared with 124,101 before subsampling.

**Source:** results/a3_sequence_size_counts.tsv; scripts/count_a3_sequence_sizes.py. Counts read directly from cached TSV; provenance recorded in analysis notes.

## 20. B07 — Q&A preparation (Q&A backup)

Answer the direct question first, then one supporting fact. These prompts are a rehearsal aid; normally keep the final takeaway slide visible and navigate directly to the relevant evidence table.

**If asked:** Why MViT? It supports multiscale image and temporal representations in a common family; architecture novelty is not the claim. Why ten seeds? To quantify optimization variability; this does not replace AR-level or temporal uncertainty analysis. Why TSS? It balances hit rate and false-positive rate under class imbalance, but it is not sufficient for an operational policy. Is 180 minutes optimal? It is the strongest mean TSS/AUC among the paper’s three 16-frame settings, not a global optimum. Did lower resolution save compute? No measured backbone speedup: inputs were returned to 224×224. Are four frames equivalent to sixteen? No formal equivalence test; they are competitive in these results. Is the threshold tuned on test? The reported main TSS uses a validation-selected threshold; repository metrics also contain other threshold variants, which must not be substituted. Is a 3D model proven better than a 2D model? No: use a matched last-frame baseline before making that claim.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. §6.

## 21. B08 — References and visual provenance (Q&A backup)

These references are copied from the accepted paper. The main charts are newly drawn for readability using the paper’s numeric results. The magnetograms are actual data, not generated scientific images. The cohort analysis is explicitly identified as supplementary repository material.

**Source:** Garmaev & Meunier, accepted ICANN 2026 author proof, 715691_1_En_43_Chapter_Author.pdf. references 1–3 and 17; presentation/assets/frame_provenance.json.
