import numpy as np
#adding baysian01 folder directory to path
import sys
from SRtools import sr_mcmc as srmc
import argparse
from SRtools import config_lib as cl
import os
from SRtools import deathTimesDataSet as dtds
import ast
from SRtools import readResults as rr
from SRtools import SR_hetro as srh


def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments.")
    parser.add_argument("config_path", type=str, help="the path to the config file to get params from.")
    parser.add_argument("folder", type=str, help="The path of the output h5 folder")
    parser.add_argument("index", type=int, help="The index of array job")
    parser.add_argument("config_name", type=str, help="The name of the config")
    parser.add_argument("test_idx", type=int, help="The index of the test")
    parser.add_argument("datasets_folder", type=str, help="The path of the datasets folder")

    args = parser.parse_args()
    # Determine if config_path is Excel or not
    is_excel = args.config_path.lower().endswith('.xlsx') or args.config_path.lower().endswith('.xls')
    if is_excel:
        # Use ExcelConfigParser (config_name is required)
        config = cl.read_excel_config(args.config_path, args.config_name)
        cfg = cl.config_to_dict(config, mcmc_convert=True)
        name = str(cfg.get('name'))
        nsteps = int(cfg.get('nsteps'))
        npeople = int(cfg.get('npeople'))
        t_end = int(cfg.get('t_end'))
        nwalkers = int(cfg.get('nwalkers'))
        h5_file = str(cfg.get('h5_file_name'))
        num_mcmc_steps = int(cfg.get('n_mcmc_steps'))
        metric = str(cfg.get('metric'))
        time_range = cfg.get('time_range')
        time_step_multiplier = int(cfg.get('time_step_multiplier'))
        data_file = str(cfg.get('data_file'))
        variations = cfg.get('variations')
        prior = cfg.get('prior')
        transform = bool(cfg.get('transform'))
        external_hazard = cfg.get('external_hazard',None)
        data_dt = float(cfg.get('data_dt', 1))
        ndims = int(cfg.get('ndims', 4))
        hetro = bool(cfg.get('hetro', False))
        test_mode = bool(int(cfg.get('test', 0)))
    else:
        raise RuntimeError("Non-Excel config files are not supported in this script. Please provide an Excel config file.")


    if external_hazard is None:
        external_hazard = np.inf
    else:
        external_hazard = float(external_hazard)


    if test_mode:
        data_file = os.path.join(args.datasets_folder, f"{name}_dataset_{args.test_idx}.csv")
    h5_file_path = os.path.join(args.folder, f"{h5_file}_{args.index}.h5")
    ds = dtds.dsFromFile(data_file, properties=['death dt'])
    ds.external_hazard = external_hazard
    eta_seed = float(cfg.get('eta'))
    beta_seed = float(cfg.get('beta'))
    epsilon_seed = float(cfg.get('epsilon'))
    xc_seed = float(cfg.get('xc'))
    exth_seed = cfg.get('ExtH',None)
    if exth_seed is not None:
        exth_seed = float(exth_seed)
    seed = np.array([eta_seed, beta_seed, epsilon_seed, xc_seed])
    if transform:
        seed = srmc.transform(seed)
    seed =np.array(seed)
    if len(seed==4) and ndims == 5:
        if exth_seed is None:
            raise RuntimeError("External hazard seed is not provided but ndims is 5.")
        seed = np.append(seed,exth_seed)
    bins =srmc.get_bins_from_seed(seed, ndims =ndims, variations = variations)

    if hetro:
        model = srh.model
        print('Using hetro model')
    else:
        model = srmc.model  
        print('Using homogenous model')   

    sampler = srmc.getSampler(nwalkers=nwalkers,
                                  num_mcmc_steps=num_mcmc_steps, dataSet= ds,back_end_file=h5_file_path,npeople=npeople,nsteps=nsteps,
                                  t_end=t_end,metric=metric,time_range=time_range, time_step_multiplier=time_step_multiplier,
                                  bins=bins, prior=prior, ndim=ndims,
                                  transformed=True, dt=data_dt, model_func=model
                                  )

    print(f"h5_file: {h5_file_path}")
    print(f"metric: {metric}")


if __name__ == "__main__":
    main()