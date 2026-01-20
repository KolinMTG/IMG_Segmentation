Voici un **prompt prêt à l’emploi**, spécifiquement conçu pour **Claude IA**, qui encadre strictement le travail attendu, évite toute dérive architecturale, et garantit la **compatibilité totale avec ton pipeline existant**.

Le prompt est volontairement **précis, contraignant et orienté implémentation**, car c’est exactement ce qu’il faut pour ce type de refactor contrôlé.

---

## Prompt à fournir à Claude IA

You are given an existing, fully functional Python codebase implementing a histogram-based KDE model for image segmentation.

Your task is to **extend and improve the existing pipeline**, while strictly respecting the constraints below.

---

### 1. General objective

The goal is to **improve the performance and stability of the histogram-based model**, in particular:

* Reduce class imbalance at inference time
* Prevent dominant classes (e.g. Building, Road) from overwhelming others
* Ensure that predicted class proportions per image are globally coherent with expected class surfaces
* Improve numerical stability of log-likelihood computation

This must be achieved **without changing the core paradigm** of the model (histogram + KDE).

---

### 2. Hard constraints (must be strictly respected)

* **Do NOT remove or rename any existing public function**
* **Do NOT break any existing functionality**
* Existing behavior must remain valid if all new options are disabled
* Only additive or optional changes are allowed

---

### 3. Mandatory preserved public API

The final pipeline must remain fully usable **using only the following functions**:
* `extract_histograms_streaming(...)` # Extract the histogram info and save it
* `run_histogram_inference(...)` # Run the inference for a given histogram model

This function:

* must keep its name unchanged
* may accept new optional parameters if needed
* must internally integrate all new logic

No additional external orchestration function is allowed.

---

### 4. What must be implemented (mandatory)

You must implement **all** of the following strategies.

---

#### 4.1 Numerically stable KDE log-likelihood

* Convert KDE values to log-probabilities
* Introduce a small epsilon floor to avoid `log(0)`
* This must be done centrally and reused everywhere

---

#### 4.2 Global dataset-level class prior

* Compute class priors from histogram statistics
* Integrate them into the per-pixel score as an additive log-prior
* This prior must be optionally enabled/disabled via parameters

---

#### 4.3 Adaptive per-image class prior (mandatory)

You must implement the **adaptive per-image prior strategy** as follows:

1. Run an initial inference pass without adaptive correction
2. Estimate predicted class ratios for the image
3. Compute a per-class bias term:

```
delta_c = log((target_ratio_c + eps) / (predicted_ratio_c + eps))
```

4. Apply this bias additively to all pixels and re-run inference
5. Allow 1–3 iterations (configurable)

This logic must:

* be fully encapsulated inside the inference pipeline
* be optional and parameter-driven
* never modify the histogram/KDE data

---

#### 4.4 Class confidence calibration (soft constraint)

* Introduce optional per-class temperature scaling
* Apply it to class scores to reduce over-confident classes
* This must be disabled by default

---

### 5. Explicit exclusions

You must **NOT** implement:

* Any spatial regularization (CRF, MRF, smoothing, morphology, etc.)
* Any deep learning model
* Any feature extraction logic
* Any change to histogram creation logic except safe extensions

The user already has a robust post-processing pipeline.

---

### 6. Code style and documentation (mandatory)

* All code must be written in **English**
* Variable names, function names, and log messages must be in English
* Comments must be **very concise**
* Important comments must start with `#!`
* Any `print()` must be replaced with `log.info(str)`
* Logging must assume `log` is already defined

---

### 7. Docstrings policy

* Small helper functions: short docstrings (1–2 lines)
* Core functions (histogram creation, inference):

  * Proper, well-structured docstrings
  * Clearly document:

    * purpose
    * parameters
    * returned values
    * behavior of new options

---

### 8. Backward compatibility requirement

* If all new options are disabled:

  * The output must be identical to the original behavior
* Default parameters must preserve legacy behavior

---

### 9. Expected output

You must output:

* The **full updated code**
* With all new logic implemented
* With no placeholder or pseudo-code
* Fully runnable and internally consistent

Do not explain the code unless explicitly requested.
Focus exclusively on **correct, clean, and robust implementation**.

---

### 10. Final reminder

You are **adapting** an existing pipeline, not redesigning it.

Think in terms of:

* calibration
* stability
* additive improvements
* controlled inference logic

---

If you follow these instructions correctly, the resulting pipeline should:

* preserve existing functionality
* significantly improve class balance
* make the histogram-based model usable at scale