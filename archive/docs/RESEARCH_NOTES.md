# Research Notes & Ideas

## 1. The Overfitting Problem
**Observation:** The model achieves near-perfect Train AUC (~0.999) but significantly lower Val AUC (~0.815), indicating massive overfitting/memorization.

**Root Causes:**
1.  **Hard Negative Sampling:** By balancing the training set 50/50, we discard ~75% of negatives. The model likely learns to distinguish "Flares" from "Quiet Sun" (easy) but fails on "Active Regions that don't flare" (hard), leading to high False Positives in validation.
2.  **Model Capacity vs. Data:** ConvNeXt is powerful enough to memorize the specific morphology of every sunspot in the training set, especially with small 112x112 inputs.
3.  **Loss Function:** Focal Loss is unnecessary (and potentially harmful) when the training batches are already balanced 50/50. Standard BCE or SoftTargetCrossEntropy (with Mixup) is preferred.

**Solutions:**
*   **Mixup/Cutmix:** Blending images forces the model to predict soft probabilities (e.g., "50% flare"), making memorization impossible.
*   **Hard Negative Mining:** Instead of random subsampling, select negatives that the model currently misclassifies (high probability false positives) to include in the training set.

## 2. Novel Scientific Contributions (Beyond "Just use a 3D model")

### Idea A: "Physics-Informed" Attention (High Novelty)
Instead of generic Optical Flow, use **Difference Images** as an attention mechanism.
*   **Concept:** Solar physics is about *change*.
*   **Method:** Use $(Image_t - Image_{t-1})$ to generate a spatial attention map.
*   **Story:** "We force the model to focus ONLY on pixels with rapid flux emergence, explicitly ignoring static active regions."
*   **Benefit:** Computationally cheaper than Optical Flow but captures the same "dynamics" signal.

### Idea B: "Multi-Scale Temporal" Encoding (Medium Novelty)
*   **Concept:** Flares have precursors at different time scales (slow flux emergence vs. fast shear).
*   **Method:** Input a 3-channel image where:
    *   Ch1 = Current frame ($t$)
    *   Ch2 = Frame from 1 hour ago ($t - 1h$)
    *   Ch3 = Frame from 6 hours ago ($t - 6h$)
*   **Story:** "Capturing multi-scale temporal dynamics in a single static pass."
*   **Benefit:** Efficiently encodes long-term evolution that frame-to-frame Optical Flow misses.

### Idea C: "Gradient-Boosted" Hybrid (Practical Novelty)
*   **Concept:** Deep Learning excels at morphology; Physics excels at boundary conditions.
*   **Method:**
    1.  Train CNN on images to extract an embedding vector.
    2.  Concatenate with physical parameters (Total Flux, R-value, Fracture, etc.).
    3.  Train an XGBoost/MLP classifier on the combined vector.
*   **Story:** "Deep Learning + Physics Parameters > Either alone."

## 3. Current Experiment Plan (Overnight)
Running 4 variations to establish a strong baseline and test the "Explicit Dynamics" hypothesis:
1.  **Exp1_Basic_Reg:** ConvNeXt-Tiny + Strong Regularization (Dropout 0.3, StochDepth 0.2, WD 0.05).
2.  **Exp2_Basic_Aug:** Basic + Safe Augmentations (H-Flip, Rotation, Scale).
3.  **Exp3_Flow_Reg:** Two-Stream (Image + Flow) + Strong Regularization.
4.  **Exp4_Flow_Aug:** Two-Stream (Image + Flow) + Safe Augmentations.

**Future Experiment (Mixup):**
*   Test **ConvNeXt-Atto** (smaller model) with **Mixup** (0.8) to aggressively combat memorization.
*   Script prepared: `classifier_NN/run_experiments_v2.py`
