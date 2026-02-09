# SI 11 and SHAP Analysis

This directory contains analysis code for studying aging and mortality patterns across multiple species using Bayesian posterior sampling, statistical variance decomposition (ANOVA), and SHAP (SHapley Additive exPlanations) analysis.

## ⚠️ Important: Download Data First

**Before running the notebooks, you must download the posterior distribution data files.**

The ANOVA analysis notebooks (`random_sampling_ANOVA_type_I.ipynb` and `random_sampling_ANOVA_type_III.ipynb`) require posterior distribution files that are not included in this repository due to their large size (~851 MB). Download the data first using:

```bash
python download_posterior_data.py --output-dir posteriors
```

This will download all required files from Zenodo ([10.5281/zenodo.17804233](https://doi.org/10.5281/zenodo.17804233)) to the `posteriors/` directory. See the [Data Download](#data-download) section below for more details.

## Overview

This project performs statistical analysis on posterior distributions obtained from MCMC sampling of survival/mortality models. The analysis focuses on understanding which model parameters (eta, beta, epsilon, xc) contribute most to explaining variance in median lifetimes across diverse species.

**This analysis is used for:**
- **Figure 3b**: SHAP analysis visualization
- **Figure 2**: ANOVA variance decomposition analysis showing which parameters explain the most variance in median lifetimes
- **Supplementary Figure S9**: SHAP analysis results
- **Supplementary Figure S10**: Single parameter substitution 
- **Supplementary Table 8**: Single parameter substitution correlations
- **Supplementary Tables 9-12**: ANOVA analysis for best fits, number of parameter sets used for ANOVA validation, and 1000 random parameter sets validation of ANOVA Type I and Type III

## Species Dataset

The analysis includes data from multiple species and populations:

- **Mammals**: Mice (M/F), Cats, Dogs (Staffy, Labradors, Jack Russell, German Shepherd), Guinea Pigs, Humans (Sweden 1910, Denmark 1890/1900)
- **Invertebrates**: Drosophila (multiple strains: 217, 441, 707, 853), C. elegans
- **Microorganisms**: Yeast, E. coli

## Methodology

1. **Posterior Distribution Loading**: Loads posterior samples from MCMC analysis stored in CSV and npz formats
2. **Sample Filtering**: 
   - Removes duplicate samples
   - Filters out lowest 5% of samples by log-probability
3. **Random Sampling**: Performs random sampling from filtered posterior distributions
4. **Variance Decomposition**: Uses ANOVA analysis to determine which parameters explain the most variance in median lifetimes
5. **Statistical Analysis**: Tests all parameter orderings to find optimal variance explanation

## Key Parameters

The survival model uses four main parameters:
- **eta** (η): Related to baseline mortality rate
- **beta** (β): Related to mortality acceleration
- **epsilon** (ε): Related to mortality variability
- **xc**: Critical age parameter

## Files

- `shap_analysis_4.py`: Python script for SHAP (SHapley Additive exPlanations) analysis
  - Generates **Figure 3b**: SHAP analysis visualization
  - Generates **Supplementary Figure S9**: SHAP analysis results
  - Analyzes feature importance and contributions to model predictions

- `random_sampling_ANOVA_type_I.ipynb`: Jupyter notebook for ANOVA Type I analysis
  - Generates **Supplementary Tables 9-12**: 
    - Table 9: ANOVA for best fits
    - Table 10: Number of parameter sets used for ANOVA validation
    - Table 11: 1000 random parameter sets validation of ANOVA Type I
    - Table 12: 1000 random parameter sets validation of ANOVA Type III
  - Performs ANOVA Type I variance decomposition analysis
  - Random sampling procedures and statistical validation

- `random_sampling_ANOVA_type_III.ipynb`: Jupyter notebook for ANOVA Type III analysis
  - Generates **Supplementary Tables 9-12**: 
    - Table 9: ANOVA for best fits
    - Table 10: Number of parameter sets used for ANOVA validation
    - Table 11: 1000 random parameter sets validation of ANOVA Type I
    - Table 12: 1000 random parameter sets validation of ANOVA Type III
  - Performs ANOVA Type III variance decomposition analysis
  - Random sampling procedures and statistical validation

- `Single_parameter_substitution_test.ipynb`: Jupyter notebook for single parameter substitution analysis
  - Generates **Supplementary Figure S10**: Single parameter substitution test visualization
  - Generates **Supplementary Table 8**: Single parameter substitution test correlations 
  - Tests the effect of substituting individual parameters across species
  - Analyzes how parameter substitutions affect model predictions
  
- `download_posterior_data.py`: Script to download all posterior distribution files from Zenodo
  - Automatically fetches all files from the dataset
  - Includes progress tracking and error handling
  - Can resume interrupted downloads
  
- `summery_mode_no_CI.csv`: Summary statistics for all species including:
  - Parameter values (xc, eta, beta, epsilon)
  - Derived quantities (xc/eta, beta/eta, etc.)
  - Median and maximum lifetimes (best fit and data)
  - Maximum likelihood log-probabilities

- `SHAP_outputs/`: Directory containing SHAP analysis output files
  - `shap_values4.npy`: SHAP values array
  - `shap_data4.csv`: SHAP data in CSV format
  - `shap_results4.csv`: SHAP results summary
  - `base_value4.npy`: Base values for SHAP analysis

- `posteriors/`: Directory containing posterior distribution files (excluded from git due to large file sizes)
  - CSV files with posterior samples
  - NPZ files with additional data
  - Files should be downloaded using `download_posterior_data.py`

## Dependencies

The analysis requires:
- `SRtools`: Custom library for survival analysis and MCMC sampling
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical functions
- `statsmodels`: ANOVA analysis
- `matplotlib`: Plotting
- `emcee`: MCMC sampling
- `corner`: Corner plots for posterior visualization

## Usage

1. **Download the data first** (see [Data Download](#data-download) section below):
   ```bash
   python download_posterior_data.py --output-dir posteriors
   ```

2. Ensure all dependencies are installed (see [Dependencies](#dependencies) section)

3. Run the analysis scripts/notebooks:
   - **For SHAP analysis**: Run `shap_analysis_4.py` to generate Figure 3b and Supplementary Figure S9
   - **For ANOVA Type I analysis**: Open `random_sampling_ANOVA_type_I.ipynb` and run cells sequentially
   - **For ANOVA Type III analysis**: Open `random_sampling_ANOVA_type_III.ipynb` and run cells sequentially
   - **For single parameter substitution test**: Open `Single_parameter_substitution_test.ipynb` and run cells sequentially to generate Supplementary Figure S10 and Supplementary Table 8

4. The notebooks will:
   - Load and process posterior distributions
   - Perform random sampling and statistical analysis
   - Generate variance decomposition results
   - Produce supplementary tables (9-12) for ANOVA validation

## Results

The analysis identifies which parameters contribute most to explaining variance in median lifetimes across species. Results are used to understand the relative importance of different biological mechanisms in aging and mortality.

The results from this analysis are presented in:
- **Figure 3b**: SHAP analysis visualization
- **Figure 2**: ANOVA analysis showing variance decomposition across species
- **Supplementary Figure S9**: SHAP analysis results
- **Supplementary Figure S10**: Single parameter substitution test results
- **Supplementary Table 8**: Single parameter substitution test
- **Supplementary Tables 9-12**: ANOVA analysis for best fits, number of parameter sets used for ANOVA validation, and 1000 random parameter sets validation of ANOVA Type I and Type III

## Data Download

Posterior distribution files are available from Zenodo: [10.5281/zenodo.17804233](https://doi.org/10.5281/zenodo.17804233)

To download all files automatically, use the provided script:

```bash
python download_posterior_data.py --output-dir posteriors
```

This will download all posterior distribution files (CSV and NPZ formats) to the specified directory. The script includes:
- Progress bars (if `tqdm` is installed)
- Automatic resumption (skips already downloaded files)
- Error handling and summary statistics

**Note**: The total dataset size is approximately 851 MB.

## Note

The `posteriors/` folder is excluded from version control due to large file sizes. Posterior distribution files should be downloaded using the script above or obtained from the Zenodo repository.

## Citation

If you use this code, please cite the associated publication (to be added).

## License

[Add license information]

