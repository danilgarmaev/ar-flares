# Engaging version — ICANN 2026

**15 September update:** The PowerPoint now contains subsequent user edits and a white university logo on the title slide. It is the current source of truth; rebuilding from `build_presentation.py` would discard those manual edits. The logo was added directly to the saved PowerPoint, preserving all existing title-slide shapes and leaving every other slide and all notes unchanged. The pre-logo user-edited deck is in `archive_user_edits/20260915_before_logo.pptx`. `Title_Slide_Logo_Preview.png` previews the updated cover; the older full-deck companion PDF does not include subsequent manual edits or this logo. Logo source and verification are recorded in `logo_update_report.json`.

This is a separate alternative to the clear version in the parent directory. The earlier PowerPoint, speaker notes, sources and previews are preserved unchanged.

## Files

- `ICANN_2026_Solar_Flare_Prediction_Engaging.pptx`: editable PowerPoint, with notes embedded on every slide.
- `Speaker_Notes_Engaging.pdf`: printable speaking script, delivery cues, pronunciation and Q&A guidance.
- `Speaker_Notes_Engaging.md`: the same guide in editable text.
- `ICANN_2026_Solar_Flare_Prediction_Engaging_preview.pdf`: companion preview, including hidden slides.

The main talk remains 14 slides, targeting 14 minutes 25 seconds. The script has 1,632 words, leaving time for pauses and explaining the figures. The question period is a separate five minutes. There are now 10 hidden backup slides.

## What changed

The introduction still explains solar flares, their classes and the prediction task. The revised narrative gives more weight to the research contribution: temporal coverage and observation count are separate design choices. The sequence results are connected explicitly, so the audience can understand why longer windows and competitive sparse inputs belong in the same story.

Titles and takeaways emphasize findings. The notes explain what each finding suggests, without turning uncertainty into an unsupported performance claim. Numeric results, scientific images, significance qualifications and the PR-AUC prevalence reference remain intact. Slide 5 now gives both equivalent TSS formulas, with sensitivity, specificity and false-positive rate explained in the notes. Pronunciation guidance is retained; there is still no need to read every plotted value aloud.

## State-of-the-art question

The earlier version had general Q&A prompts but no dedicated external performance comparison. Two new hidden slides address it:

- **Slide 23 / B09:** selected published ≥M, next-24-hour TSS results, with inputs and evaluation differences visible beside the scores.
- **Slide 24 / B10:** what a fair benchmark would require, and how to explain the contribution without claiming a record score.

Use PowerPoint’s slide navigator to open these during questions. They are hidden from normal slideshow progression. The final main slide can stay on screen until a backup is useful.

Suggested short answer:

> Our four-image setting reaches a mean TSS of about 0.75 for M-class-or-stronger flares. Published scores provide useful context, but the methods were evaluated with different inputs and examples, so this study does not establish a state-of-the-art ranking. Our contribution is the input-design evidence: longer coverage helped the C task, while sparse observations remained competitive. A shared-data benchmark against strong models is the next step.

The comparison uses primary sources checked on 14 September 2026:

- [Riggi et al., arXiv:2510.23400v1, Table 6](https://arxiv.org/html/2510.23400v1): the default split and tuned-threshold column. The exact preprint version is identified to avoid mixing published and preprint values.
- [Alatoom and Nikolaou (2026), Table 5](https://doi.org/10.1029/2026SW005010): SDOBenchmark results, with active-region aggregation and multiple SDO wavelengths.
- This study: accepted ICANN author proof, Tables 2 and 5.

This is selected performance context, not an exhaustive leaderboard. The table does not establish statistical superiority. In particular, the C-task score of 0.805 cannot be compared with another paper’s M-task score. Within our own work, image and sequence performance also involve different eligible examples.

## Scientific and file checks

The existing scientific audit remains in `../Analysis_and_Verification.md`, with supporting JSON files in the parent directory. Its longitude-threshold issue remains handled by showing threshold-free ROC-AUC results; this revision does not restore the affected TSS claims. No models were retrained and no new model evaluations were run.

Validation checks the PowerPoint package, XML, notes, hidden-slide flags, embedded images, shape bounds and text height. The companion PDF was visually inspected. This environment does not have Microsoft PowerPoint or LibreOffice, so the preview is generated from the shared drawing code rather than exported by either application.

`preserved_originals_sha256.json` records the earlier files; `preservation_check.json` records the final comparison. `validation_report.json` records artifact checks. All new outputs are confined to this `engaging` directory.
