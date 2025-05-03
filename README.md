# Paediatric_Sleep_Logs

## Overview

This repository presents an AI-driven pipeline for comprehensive analysis of paediatric sleep rhythms.  
Key features include:

* Integration of unsupervised tensor decomposition and UMAP to capture developmental stages, diurnal variations, and daily fluctuations in 14-day sleep logs  
* Visualization of typical sleep development trajectories from real-world, parent-monitored data  
* Association analysis with developmental disorders (e.g., ASD, ADHD) to identify characteristic sleep patterns  
* Introduction of a deviation metric that quantifies divergence from typical development, outperforming traditional sleep measures in detecting atypical rhythms


## Prerequisites
- **Python 3.10.6**
  - numpy, pandas, matplotlib, seaborn, tensorly, umap-learn, scipy, scikit-learn, hdbscan, shap
- **R 4.3.0**
  - tidyverse (ggplot2, dplyr, readr, tidyr, purrr), uwot, dbscan 

## Repository Structure

```
Paediatric_Sleep_Logs/
├── data/
│   ├── Discovery data/               # 14-day sleep logs for developing the method
│   ├── Discovery train data/         # Training data for the model
│   └── Validation data/              # Logs used for association analysis
├── processing/
│   ├── processing_of_discovery_data.py   # Python: undersampling & tensor preparation
│   └── processing_of_validation_data.R   # R: missing data filtering
├── Discovery_data_analysis/
│   ├── correlation.R
│   ├── non-negative_tensor_decomposition.py
│   ├── optimal_rank_selection.py
│   └── umap_visualization.R
├── Validation_data_analysis/
│   ├── apply_NTD.py
│   ├── disease_identification.py
│   └── umap_analysis.R
├── results/
│   └── Figures.pdf
├── table_data/
│   └── Tables.pdf
└── README.md
```

## Contact

For questions or issues, please contact:  
Shinji Oguchi  
RIKEN / Chiba University  
shinji.oguchi@chiba-u.jp

