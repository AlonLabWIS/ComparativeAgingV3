# Comparative Aging Analysis V3

This repository contains the complete code and analysis for the comparative aging study across multiple species "A damage accumulation model reveals strategies of aging across species". It includes simulation tools, Bayesian MCMC analysis, figure generation code, and dataset preparation workflows.

## Quick Start

**Before running any analysis, you must first set up the repository and download the required data files.**

Run the setup script to install dependencies and download posterior distribution data:

```bash
./setup.sh
```

Or manually:

```bash
# Install SRtools package and requirements
cd SRtools
pip install -r requirements.txt
pip install -e .
cd ..

# Download posterior distribution data
python download_posterior_data.py
```

The setup script will:
1. Install the `SRtools` package and all required dependencies
2. Download posterior distribution files from Zenodo to the appropriate directories

**Note**: The posterior distribution files are large (~851 MB) and are stored in git-ignored directories. They will be automatically downloaded to:
- `random_posterior_sampling_analysis/posteriors/`
- `Baysian03/analysis/products_baysian/posteriors/`

## Repository Structure

### Datasets Preparation

The `Datasets_preperation/` folder contains all raw and processed datasets used in this study, along with processing notebooks that document the data cleaning and preparation steps.

**Contents:**
- `Rawfiles/`: Original raw data files from various sources
- `cleaned_datasets/`: Processed datasets ready for analysis
- `Cleanup_notebooks/`: Jupyter notebooks documenting the data processing pipeline for each dataset
- `Lifetables/`: Life table data for human populations

Each cleanup notebook includes detailed documentation explaining the data processing steps, transformations applied, and any filtering or quality control measures taken.

**Note**: Datasets for yeast and C. elegans have been removed from this repository. Raw and processed versions will be made available separately upon approval from the data owners.

### SRtools

The `SRtools/` package contains the core simulation and analysis tools used throughout this project. This package includes:

- **Simulation tools**: Code for running survival/mortality simulations using the SR model
- **Analysis tools**: Statistical analysis functions for mortality data
- **MCMC tools**: Bayesian MCMC sampling and posterior analysis utilities
- **Visualization utilities**: Plotting functions for figures and analysis

The package is installed as a Python package and can be imported as:
```python
import SRtools
```

See `SRtools/README.md` for detailed documentation of the package components.

### Bayesian Analysis

The `Baysian03/` folder contains the complete results of all MCMC runs and their analysis.

**Key components:**
- `analysis/`: Full analysis notebooks for each MCMC run, including:
  - Posterior distribution analysis
  - Parameter estimation results
  - Likelihood statistics
  - Diagnostic plots 
- `datasets/`: Processed datasets used for MCMC analysis
- `configurations_baysian.xlsx`: Configuration file specifying all MCMC run parameters
- `run_*.py`: Scripts for running MCMC analyses

Each analysis notebook in the `analysis/` folder provides a complete workflow for a specific dataset, including data loading, model fitting, posterior sampling, and result visualization.

### Figures

The `Figures/` folder contains Jupyter notebooks with code to reproduce all main and supplementary figures from the publication.

**Figure notebooks:**
- `Fig_2_datasets_vs_sim.ipynb`: Figure 2 - Comparison of datasets with simulations, and supplementary figure S2
- `FIg_3_production_vs_LS.ipynb`: Figure 3a - Production vs. lifespan analysis
- `Fig_3_balistic_vs_ss.ipynb`: Figure 3c - Ballistic vs. steady-state comparison
- `Fig_4_invariants_in_mammals.ipynb`: Figure 4 - Invariant relationships in mammals
- `Fig_5_Yeast.ipynb`: Figure 5 - Yeast analysis
- `Fig_6_dimensionlessgroups.ipynb`: Figure 6 - Dimensionless group analysis
- `Suplementary_Fig_3_all params_and_trends.ipynb`: Supplementary Figure S3
- `Supplementary_Fig_4_Weibull_and_Gompertz_fits.ipynb`: Supplementary Figure S4

**Important**: Some figure notebooks (Fig_2_datasets_vs_sim.ipynb,Supplementary_Fig_4_Weibull_and_Gompertz_fits.ipyn) require mock datasets for yeast and C. elegans. If these datasets are not available, we recommend duplicating a mice dataset and naming it:
- `Yeast_ds.csv`
- `Celegans_ds.csv`

Place these files in the `Figures/datasets/` directory.

**Additional notebooks:**
- `QSS_explanation_figure.ipynb`: Explanation figure for quasi-steady-state
- `params_vs_LS.ipynb`: Parameter vs. lifespan analysis

**Results folder (`results/`):**
The `results/` folder contains three types of parameter estimation tables with full parameter estimates for all datasets:

- **`summery_max_likelihood.csv`**: Contains the single sample from each MCMC run with the highest likelihood. This represents the best-fit parameter set based on maximum likelihood.

- **`summery_mode_overall.csv`**: After binning the MCMC samples (with averaging of likelihoods that fall in the same bin), this contains the sample with the highest likelihood within the mode bin (the bin with highest posterior probability). Either `summery_max_likelihood.csv` or `summery_mode_overall.csv` should be used for simulations, as they contain complete parameter sets.

- **`summery_mode.csv`**: Contains the modes (highest probabilities) for marginalized posterior distributions over different parameters and parameter groups, including 95% confidence intervals. **Important**: Since these are marginalized distributions, the values represent modes of individual parameters rather than a coherent parameter set. Therefore, this file should **not** be used for simulations, but is useful for understanding the distribution of individual parameters and their uncertainties. These are the values in tables 2,3 in the paper.

### Different Noises

The `Different_Noises/` folder contains code for **Supplementary Information Figure 9**, which analyzes the effects of different noise types on the model results.

**Contents:**
- `SR_noises.py`: Core noise analysis functions
- `Noise_tests.ipynb`: Notebook running noise sensitivity analysis
- Generated plots showing noise effects

### Random Posterior Sampling Analysis

The `random_posterior_sampling_analysis/` folder contains code for ANOVA analysis used in **Figure 3b** and **Supplementary Tables 7 and 8**.

**Contents:**
- `random_sampling.ipynb`: Main analysis notebook performing:
  - Random sampling from posterior distributions
  - ANOVA variance decomposition
  - Statistical analysis of parameter contributions
- `download_posterior_data.py`: Script to download posterior data (also available in root)
- `summery_mode_no_CI.csv`: Summary statistics used for the best fit values.

This analysis identifies which model parameters (eta, beta, epsilon, xc) contribute most to explaining variance in median lifetimes across species.

### Performance Tests

The `performence tests/` folder contains Excel files documenting all test runs and their configurations. Some of this analysis is presented in SI 6.

**Contents:**
- `configurations_for_tests.xlsx`: Full configuration specifications for all test runs
- `summery_of_error_analysis.xlsx`: Summary of error analysis results

These files provide complete documentation of the testing methodology and results used to validate the analysis pipeline.

## Requirements

All Python package requirements are specified in `SRtools/requirements.txt`. The main dependencies include:

- numpy, pandas, scipy
- matplotlib, seaborn, plotly
- emcee (MCMC sampling)
- jupyter, ipykernel
- lifelines (survival analysis)
- corner (posterior visualization)
- And others (see `SRtools/requirements.txt` for complete list)

Install all requirements using:
```bash
cd SRtools
pip install -r requirements.txt
pip install -e .
```

## Data Availability

Posterior distribution files are available from Zenodo: [10.5281/zenodo.17804233](https://doi.org/10.5281/zenodo.17804233)

The download script (`download_posterior_data.py`) automatically retrieves these files during setup.

## Citation

If you use this code, please cite the associated publication (citation to be added upon publication).

## License

[Add license information]

## Contact

[Add]

