

from SRtools import sr_mcmc as srmc
from SRtools import config_lib as cl
from SRtools import deathTimesDataSet as dtds
from SRtools import readResults as rr

import matplotlib.pyplot as plt

from SRtools import SRmodellib as sr
from SRtools import SRmodellib_lifelines as srl
from SRtools import samples_utils as su
from SRtools import SR_hetro as srh
from SRtools import joint_posterior as jp
from SRtools import presets
import shap
import pandas as pd
import numpy as np

paths = ['mice_F','mice_M','yeast','ecoli','cats_vp_M', 'cats_vp_F',
        'drosophila_853','drosophila_707','drosophila_441','drosophila_217',
        'Sweden_F_1910_homo','Denmark_M_1900_homo','Denmark_M_1890_homo',
        'Staffy_vetCompass','Labradors_vetCompass','Jack_Russell_vetCompass','German_Shepherd_vetCompass',
        'celegans','Guiniea_pig_VC'
        ]


def run_shap_analysis():
    """Run full SHAP analysis: load data, build explainer, compute SHAP values, save results."""
    # Load the summary CSV, keeping only the columns in `paths` and the indices of interest
    summary_file = 'summery_mode_overall.csv'

    # Indices (row labels) we want, in order:
    summary_indices = ['xc/eta', 'beta/eta', 'xc^2/epsilon', 'xc']

    # Load entire table, first row as header, first column as index
    summary_df = pd.read_csv(summary_file, index_col=0)

    # Keep only desired indices and columns from the `paths` list
    summary_filtered = summary_df.loc[summary_indices, paths]

    # Calculate eta, beta, epsilon using formulas:
    xc_over_eta = summary_filtered.loc['xc/eta'].astype(float)
    beta_over_eta = summary_filtered.loc['beta/eta'].astype(float)
    xc_sq_over_epsilon = summary_filtered.loc['xc^2/epsilon'].astype(float)
    xc = summary_filtered.loc['xc'].astype(float)

    # eta = (xc/eta) / xc
    eta = xc_over_eta / xc

    # beta = (beta/eta) * eta
    beta = beta_over_eta * eta

    # epsilon = xc / (xc^2/epsilon)
    epsilon = xc / xc_sq_over_epsilon

    # Take log for each parameter and xc
    log_df = pd.DataFrame({
        'log_eta': np.log(eta),
        'log_beta': np.log(beta),
        'log_epsilon': np.log(epsilon),
        'log_xc': np.log(xc)
    })

    # Transpose so parameters are columns, paths are rows
    log_df = log_df.transpose()
    log_df.index = ['log_eta', 'log_beta', 'log_epsilon', 'log_xc']
    log_df = log_df.transpose()



    # ---------------------------------------------------------
    # 1. SETUP YOUR DATA
    # ---------------------------------------------------------
    # Load your 25 calibrated parameter sets.
    # Ensure column names are meaningful for the plots later.

    X = log_df

    # ---------------------------------------------------------
    # 2. DEFINE THE MODEL WRAPPER
    # ---------------------------------------------------------
    # This function connects SHAP to your heavy SDE solver.
    # SHAP passes 'X_batch' as a numpy array or dataframe.
    # We must loop through it because your SDE runs one at a time.

    def sde_wrapper(X_batch):
        # If X_batch comes as a numpy array, convert to DataFrame for easier handling
        if isinstance(X_batch, np.ndarray):
            X_batch = pd.DataFrame(X_batch, columns=X.columns)

        results = []

        for index, row in X_batch.iterrows():
            p1 = row['log_eta']
            p2 = row['log_beta']
            p3 = row['log_epsilon']
            p4 = row['log_xc']


            p1 = np.exp(p1)
            p2 = np.exp(p2)
            p3 = np.exp(p3)
            p4 = np.exp(p4)
            theta = np.array([p1, p2, p3, p4])
            config = presets.get_config_params('yeast', time_unit='days', config_params =['nsteps', 'time_step_multiplier', 'npeople', 't_end','hetro'])
            config['npeople'] = 700
            sim  = srh.getSrHetro(theta,**config, parallel=False)
            lifespan =sim.getMedianLifetime()
            if lifespan is None or np.isinf(lifespan):
                config['t_end'] *= 20
                config['time_step_multiplier'] *= 20
                sim  = srh.getSrHetro(theta,**config, parallel=False)
                lifespan =sim.getMedianLifetime()
                if lifespan is None or np.isinf(lifespan):
                    config['t_end'] *= 20
                    sim  = srh.getSrHetro(theta,**config, parallel=False)
                    lifespan =sim.getMedianLifetime()
                    if lifespan is None or np.isinf(lifespan):
                        config['t_end'] *= 20
                        sim  = srh.getSrHetro(theta,**config, parallel=False)
                        lifespan =sim.getMedianLifetime()
                        if lifespan is None or np.isinf(lifespan):
                            config['t_end'] *= 20
                            sim  = srh.getSrHetro(theta,**config, parallel=False)
                            lifespan =sim.getMedianLifetime()
                            if lifespan is None or np.isinf(lifespan):
                                lifespan = config['t_end']

            predicted_lifespan = np.log(lifespan)
            results.append(predicted_lifespan)

        return np.array(results)

    # ---------------------------------------------------------
    # 3. RUN SHAP (The Heavy Lift)
    # ---------------------------------------------------------
    explainer = shap.KernelExplainer(model=sde_wrapper, data=X)
    shap_values = explainer.shap_values(X)

    # ---------------------------------------------------------
    # 4. SAVE RESULTS IMMEDIATELY
    # ---------------------------------------------------------
    df_shap = pd.DataFrame(shap_values, columns=X.columns)
    df_shap.to_csv("shap_results4.csv", index=False)

    np.save("shap_values4.npy", shap_values)
    np.save("base_value4.npy", explainer.expected_value)
    X.to_csv("shap_data4.csv", index=False)

    print("Results saved: shap_values4.npy, base_value4.npy, shap_data4.csv")
    print("SHAP analysis complete. Results saved.")


if __name__ == "__main__":
    run_shap_analysis()