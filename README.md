
# IMG_SEGMENTATION – Satellite Image Segmentation Project

- **Date:** 2026-01-21
- **Status:** In Progress
- **Author:** Colin MANYRI
- **License:** MIT – Copyright (c) 2026 Colin MANYRI
- **Version:** 10.0.26



## Project Goal

This project implements several **complete pipelines for satellite image segmentation**, including:

* Image preprocessing
* Feature extraction
* CNN-based model training (e.g., U-Net)
* Probabilistic models (histogram-based segmentation)
* Post-processing of results
* Model evaluation and reporting

The pipelines are designed to be **modular and flexible**, allowing experimentation with different features, models, and evaluation strategies.



## Dataset

**Land Cover for Aerial Imagery (Landcover AI)**

* License: Creative Commons
* **Kaggle:** [https://www.kaggle.com/datasets/aletbm/land-cover-from-aerial-imagery-landcover-ai](https://www.kaggle.com/datasets/aletbm/land-cover-from-aerial-imagery-landcover-ai)
* **Official website:** [https://landcover.ai.linuxpolska.com/](https://landcover.ai.linuxpolska.com/)

**Minimal required data:**

* `images/` – satellite images
* `labels/` – segmentation masks

**Optional:**

* `raw_data/` – allows recreating Train/Validation/Test splits using the provided `split.py` script


## Main Features

This project extracts multiple types of features from images for segmentation, grouped into **main categories**:

* **Color and spectral:** RGB, HSV, NDVI, Water Index
* **Intensity and multi-scale context:** Grayscale, Gaussian blur (multi-scale)
* **Gradient and structural:** Gradient magnitude & orientation, anisotropy, corner density
* **Texture:** Local variance, local entropy, LBP


## Repository Structure

* [src/](src) – All source code
* [data/]() – Dataset (images, labels, raw_data)
* [logs/]() – Project log files
* [.trash/]() – Old, useless or deleted files
* [documents/](documents) – Documentation files and plots

### Inside `documents/`

* [src_structure](documents/src_structure.md) – Description of all source files
* [data_structure](documents/data_structure.md) – Structure of data folders
* [FinalProject](documents/FinalProject.pdf) – Academic subject for the project
* [plot/](documents/plot/) – Plots for data analysis and evaluation
* [evaluation_strategy](documents/evaluation_strategy.md) – Explanation of evaluation methods


## Installation

**Recommended IDE:** VS Code
**AI Code Assistance:** GitHub Copilot, Claude Sonnet 4.5, ChatGPT 5.1
**Python version:** 3.10.19
**Environment Manager:** Conda

### Create Virtual Environment

```bash
conda create -n IMG_SEG python=3.10
conda activate IMG_SEG
```

### Install Dependencies

```bash
pip install -r requirements.txt
```


## Hardware Requirements

* **CPU:** Intel i5-12400f (minimum)
* **RAM:** 16 GB DDR4
* **GPU:** NVIDIA RTX 4060 Ti (optional, recommended for faster training)

**Training time estimates:**

* **CPU + downsampling, 50 epochs:** ~6–8 hours
* **GPU (50 epochs):** depends on GPU, typically 20–40 minutes


## Data Setup

Create the following folder structure in the `data/` folder:

```
data/
├─ images/
├─ labels/
├─ raw_data/   (optional)
```

Use the provided `split.py` script to create Train/Validation/Test splits from the raw data.
> If you encounter issues with paths, check the `DataPath` class in `src/cste.py`.


## External Elements / Citations

* Landcover AI dataset - *License: Creative Commons : CC BY-NC-SA 4.0*
    - **Kaggle:** [https://www.kaggle.com/datasets/aletbm/land-cover-from-aerial-imagery-landcover-ai](https://www.kaggle.com/datasets/aletbm/land-cover-from-aerial-imagery-landcover-ai)
    - **Official website:** [https://landcover.ai.linuxpolska.com/](https://landcover.ai.linuxpolska.com/)
* Autors of the dataset
    - Boguszewski Adrian
    - Batorski Dominik
    - Ziemba-Jankowska Natalia
    - Dziedzic Tomasz
    - Zambrzycka Anna
* Kaggle uploader pseudo
    - AleTBM



## Contact / Support

For questions or issues regarding code execution:

**Colin MANYRI** – [colin.manyri@etu.utc.fr](mailto:colin.manyri@etu.utc.fr)