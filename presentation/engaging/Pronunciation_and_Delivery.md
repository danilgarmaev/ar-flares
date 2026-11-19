# Speaking guide

Use the full notes to rehearse the explanation. During the talk, use the short prompts below so you can look at the audience instead of reading sentences.

## Terms and numbers

| On the slide | Say aloud |
|---|---|
| ≥C | “C-class or stronger” |
| ≥M | “M-class or stronger” |
| C1 / M1 / X1 | “C-one,” “M-one,” “X-one” |
| TSS | “tee ess ess”; first time, “the True Skill Statistic” |
| ROC-AUC | “rock A-U-C” |
| PR-AUC | “P-R A-U-C”; first time, “the area under the precision–recall curve” |
| MViT | “em-vit”; first time, “Multiscale Vision Transformer” |
| Magnetogram | “mag-NEE-toh-gram” |
| 0.789 | “zero point seven eight nine”; for the main result, “about zero point seven nine” |
| 0.03 | “zero point zero three” |
| 3.1% | “about three percent” |
| ±30° | “within thirty degrees of the center” |
| 45 h | “forty-five hours”; for intuition, “almost two days” |
| 224 × 224 | “two twenty-four by two twenty-four pixels” |

Decimal scores are not automatically percentages of correct predictions. In particular, PR-AUC of 0.36 is **not** “36% accuracy.” TSS of 0.79 is not “79% accuracy” either.

You do not need to pronounce every number on a plot. On the main temporal-result slide, two rounded values are useful: “about zero point six seven to zero point seven nine.” For the other results, explain whether the lines rise, stay close, or show more variation. Leave exact means, standard deviations and p-values for questions.

## Short prompts for each slide

1. **Introduction:** who I am; what images should the model receive?
2. **Solar flares:** burst of radiation; radio disruption; A–X strength classes; predict before the event.
3. **Magnetograms:** magnetic-field images; opposite polarities; one region; two tasks for the next day.
4. **Four questions:** image detail, location, time covered, number of images.
5. **Data:** unseen regions at test time; rare strong flares; why accuracy is not enough; introduce TSS.
6. **Model:** small and large patterns; images or sequences; repeated training; explain error bars.
7. **Resolution:** coarser images; scores stay close; no measured speed claim.
8. **Disk position:** explain the two restored diagrams first; center versus edge; central-only training is mixed.
9. **Time:** more time covered versus fewer images; past observations and future forecast are different periods.
10. **Longer window:** clear improvement for the C task; M task less certain; one rounded numerical comparison.
11. **Fewer images:** same first and last times; similar scores; four is not proven better.
12. **PR-AUC:** explain warnings and missed flares; show the random reference first; not an accuracy percentage; still a trade-off.
13. **Next steps:** same examples, measured costs, other data.
14. **Close:** a longer view helped; fewer images can be enough; input choices deserve attention.

Pause after introducing the flare classes and after explaining the PR-AUC reference. Those are useful moments for the audience to catch up. Aim to finish near **14:25**, leaving a little space before the 15-minute limit. The eight backup slides are hidden from normal slideshow progression.

The longitude discussion still uses ROC-AUC, as explained in `Analysis_and_Verification.md`. The restored images illustrate geometry; they do not resolve the earlier threshold-provenance issue.
