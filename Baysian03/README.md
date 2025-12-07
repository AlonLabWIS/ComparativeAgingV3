# Bayesian MCMC Survival Analysis

A Bayesian Markov Chain Monte Carlo (MCMC) framework for analyzing survival and mortality data across multiple species. This repository contains tools for parameter estimation, posterior distribution analysis, and visualization of survival models.

## Overview

This project implements Bayesian MCMC methods to estimate parameters of survival models from mortality data. It supports both homogeneous and heterogeneous (individual variation) models, and can analyze datasets from various organisms including:

- **Mammals**: Cats, Dogs, Mice, Guinea Pigs, Humans
- **Invertebrates**: Drosophila (multiple strains), C. elegans
- **Microorganisms**: Yeast, E. coli

## Repository Structure

```
Baysian03/
├── datasets/                    # Input mortality datasets (CSV format)
│   │                            # Contains datasets for multiple species:
│   ├── Cats_ds.csv              # (examples shown - many more files exist)
│   ├── Cats_VP_F.csv
│   ├── Cats_VP_M.csv
│   ├── Mice_F_ds.csv
│   ├── Mice_M_ds.csv
│   ├── Celegans_ds.csv
│   ├── Ecoli_ds.csv
│   ├── Yeast_ds.csv
│   ├── Guineapig_VC_data.csv
│   ├── DOGS/                    # Dog breed datasets
│   ├── DROSOPHILA/              # Multiple Drosophila strains
│   └── HUMANS/                  # Historical human mortality data
├── analysis/                    # Analysis notebooks and results
│   ├── mcmc_analyisis_template_baysian.ipynb  # Template notebook
│   ├── mcmc_analysis_*_baysian.ipynb          # Analysis notebooks
│   ├── mcmc_analysis_*_baysian.pdf             # PDF reports
│   └── products_baysian/        # Analysis outputs (CSVs, HTML plots)
├── simulation_results/          # MCMC simulation outputs (H5 files)
├── configurations_*.xlsx        # Excel configuration files
├── run_file_mcmc_excel.py      # Main MCMC execution script
├── run_manager.py              # Cluster job submission manager
├── run_multiple_configs.py     # Batch notebook execution
└── configs_to_run.txt          # List of configurations to process
```

**Note**: The structure above shows only a subset of files. The `datasets/` folder contains many more CSV files for various species and conditions.

## Data Format

Input datasets should be CSV files with two columns:
- `death times`: Time of death (or censoring)
- `events`: Event indicator (1 = death, 0 = censored)

Example:
```csv
death times,events
130.8571429,1
157,1
95.14285714,1
```

## Dependencies

This project requires the `SRtools` Python package, which provides:
- `sr_mcmc`: MCMC sampling functionality
- `config_lib`: Configuration file parsing
- `deathTimesDataSet`: Dataset handling
- `SR_hetro`: Heterogeneous model support
- `readResults`: Result reading utilities
- `cluster_utils`: Cluster job management

Additional dependencies:
- NumPy
- Jupyter/IPython
- nbconvert (for PDF generation)
- pandas (for data handling)

## Configuration

Configurations are stored in Excel files (`configurations_*.xlsx`). Each configuration specifies:

- **Model parameters**: `eta`, `beta`, `epsilon`, `xc` (and optionally `ExtH` for external hazard)
- **MCMC settings**: Number of walkers, steps, chains
- **Data settings**: Dataset path, time range, step multiplier
- **Cluster settings**: Job name, memory, queue, number of jobs
- **Model type**: Homogeneous vs. heterogeneous (`hetro` flag)

## Usage

### Running MCMC Analysis

1. **Prepare configuration file**: Edit `configurations_*.xlsx` with your settings

2. **List configurations to run**: Edit `configs_to_run.txt` with comma-separated config names:
   ```
   yeast, yeast_dir
   ```

3. **Submit cluster jobs** (for LSF cluster):
   ```bash
   python run_manager.py configurations_baysian.xlsx configs_to_run.txt
   ```

   To preview without submitting:
   ```bash
   python run_manager.py configurations_baysian.xlsx configs_to_run.txt --no-submit
   ```

4. **Run MCMC** (individual job):
   ```bash
   python run_file_mcmc_excel.py <config_path> <h5_folder> <index> <config_name> <test_idx> <datasets_folder>
   ```

### Analyzing Results

1. **Generate analysis notebooks**:
   ```bash
   python run_multiple_configs.py configs_to_run.txt
   ```

   This script:
   - Creates analysis notebooks from the template
   - Executes them to generate results
   - Converts them to PDF reports

2. **Manual analysis**: Open individual notebooks in `analysis/` directory:
   - `mcmc_analysis_<config_name>_baysian.ipynb`

### Output Files

MCMC runs produce:
- **H5 files**: Sampler chains stored in HDF5 format (`simulation_results/`)
- **Posterior distributions**: CSV and NPZ files (`analysis/products_baysian/posteriors/`)
- **Summary tables**: CSV files with parameter estimates (`analysis/products_baysian/results_csvs/`)
- **3D plots**: Interactive HTML visualizations (`analysis/products_baysian/html_3d_plots/`)
- **PDF reports**: Executed notebooks converted to PDF (`analysis/mcmc_analysis_*_baysian.pdf`)

## Models

### Homogeneous Model
Standard survival model assuming identical parameters across all individuals.

### Heterogeneous Model (`hetro=True`)
Accounts for individual variation in survival parameters, allowing for population heterogeneity.

## Cluster Setup

The project is designed for LSF (Load Sharing Facility) cluster systems. The `run_mcmc_excel.csh` script:
- Loads conda environment (`srtools`)
- Activates the environment
- Runs MCMC jobs with proper resource allocation

Ensure your cluster environment has:
- Conda/miniconda installed
- `srtools` conda environment configured
- Python 3.x with required packages

## File Ignoring

The following are ignored by git (see `.gitignore`):
- `analysis/products/` and `analysis/products_baysian_old/` folders
- All contents of `posteriors/` folders (folders are preserved but empty)
- `*.h5` files (simulation results)
- `*.out`, `*.o`, `*.e` files (cluster outputs)
- Python cache files (`__pycache__/`, `*.pyc`)

## Notes

- Large simulation result files (H5) are not tracked in git
- Posterior distribution files are excluded but folder structure is preserved
- Analysis products can be regenerated from simulation results
- PDF reports are included for documentation purposes

## License

[Add your license information here]

## Contact

[Add contact information here]
