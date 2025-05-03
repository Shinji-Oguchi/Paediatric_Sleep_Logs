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
│   ├── Discovery data/               # 14-day sleep logs which is used to develop a novel machine-learning approach
│   ├── Discovery train data/         # 14-day sleep logs which is used to train our machine-learning model
│   └── Validation data/              # 14-day sleep logs which is used for association analysis
├── processing/
│   ├── processing_of_discovery_data.py   # Python: undersampling & tensor preparation for discovery data
│   └── processing_of_validation_data.R   # R: missing‐data filtering and tensor preparation for validation data
├── Discovery_data_analysis/
│   ├── correlation.R                     # R: compute correlations between tensor factors & sleep outcomes
│   ├── non-negative_tensor_decomposition.py  # Python: perform NTD on discovery data
│   ├── optimal_rank_selection.py         # Python: find and select optimal tensor rank
│   └── umap_visualization.R              # R: generate UMAP plots of discovery tensor metrics
├── Validation_data_analysis/
│   ├── apply_NTD.py                      # Python: apply non-negative tensor decomposition on validation data
│   ├── disease_identification.py         # Python: identify diseases in childhood from 14-day sleep logs
│   └── umap_analysis.R                   # R: UMAP‐based visualization for validation data
├── results/
│   └── Figures.pdf                       # PDF of all generated figures which is used for the paper
├── table_data/
│   └── Tables.pdf                        # PDF of all generated tables and summaries
└── README.md                             # Project overview, setup, and usage instructions
```

## Contact

For questions or issues, please contact:  
Shinji Oguchi  
RIKEN / Chiba University  
shinji.oguchi@chiba-u.jp

