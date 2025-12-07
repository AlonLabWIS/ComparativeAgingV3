
import numpy as np
from SRtools import sr_mcmc as srmc
import argparse
from SRtools import config_lib as cl
import os
import subprocess
import datetime
import re
from SRtools import cluster_utils as cu
from SRtools import SR_hetro as srh

def main():
    parser = argparse.ArgumentParser(description="A script that processes command line arguments.")
    parser.add_argument("config_path", type=str, help="the path to the config file to get params from. Supports folder with .ini files or a .xlsx Excel file.")
    parser.add_argument("specific_configs_path", type=str, help="path to a file with comma-separated configuration names (used for Excel).")
    parser.add_argument("--no-submit", action="store_true", help="do not submit cluster jobs; only parse and print config values")

    args = parser.parse_args()
    with open(args.specific_configs_path, 'r') as f:
        specific_configs_content = f.read().strip()
        if specific_configs_content:
            specific_configs_list = [name.strip() for name in specific_configs_content.split(',')]
        else:
            specific_configs_list = []


    
    config_path = args.config_path

    # Support both legacy INI configs (folder of .ini files) and Excel .xlsx configs
    is_excel = config_path.lower().endswith('.xlsx') or config_path.lower().endswith('.xls')

    if is_excel:
        if not specific_configs_list:
            raise ValueError("When using an Excel config, specific_configs_path must contain at least one configuration name (column header).")
        # Loop over all configuration names provided
        for config_name in specific_configs_list:
            config = cl.read_excel_config(config_path, config_name)
            cfg = cl.config_to_dict(config, mcmc_convert=True)
            # Extract values from dict to mirror the legacy behavior
            nsteps = int(cfg.get('nsteps'))
            npeople = int(cfg.get('npeople'))
            t_end = int(cfg.get('t_end'))
            n_jobs = int(cfg.get('n_jobs'))
            nwalkers = int(cfg.get('nwalkers'))
            job_name = str(cfg.get('job_name'))
            memory = str(cfg.get('initial_memory'))
            h5_file = str(cfg.get('h5_file_name'))
            folder = str(cfg.get('folder'))
            name = str(cfg.get('name'))
            run_file_mcmc = str(cfg.get('run_file_mcmc'))
            n_mcmc_steps = int(cfg.get('n_mcmc_steps'))
            queue = str(cfg.get('queue'))
            # mcmc may come as bool via config_to_dict, or str
            mcmc = bool(cfg.get('mcmc', True)) if isinstance(cfg.get('mcmc', True), bool) else str(cfg.get('mcmc', True)).lower() == 'true'
            test_mode = bool(int(cfg.get('test', 0)))
            n_tests = int(cfg.get('n_tests', 1))
            if test_mode:
                print(f"Running in test mode with {n_tests} tests")
            else:
                print("Running in normal mode")

            # Create submission structure unless running with --no-submit
            if not args.no_submit:
                # Create a subfolder for the current submission date
                submission_date = datetime.datetime.now().strftime("%d_%m_%Y")
                submission_folder = os.path.join(folder, f"{name}_submit_{submission_date}")
                cl.add_submition_folder(config, submission_folder, config_path)
                os.makedirs(submission_folder, exist_ok=True)
                
                # Create datasets folder if in test mode
                if test_mode:
                    datasets_folder = os.path.join(submission_folder, "datasets")
                    os.makedirs(datasets_folder, exist_ok=True)
                    n_dataset = int(cfg.get('n_dataset', 10000))
                    eta = float(cfg.get('eta'))
                    beta = float(cfg.get('beta'))
                    epsilon = float(cfg.get('epsilon'))
                    xc = float(cfg.get('xc'))
                    ExtH = cfg.get('ExtH',None)
                    if ExtH is not None and ExtH != '':
                        ExtH = float(ExtH)
                    else:
                        ExtH = None
                    time_step_multiplier = int(cfg.get('time_step_multiplier'))
                    # Create n_tests dataset folders 
                    for i in range(n_tests):
                        dataset_path = os.path.join(datasets_folder, f"{name}_dataset_{i}.csv")
                        sim_ds=srh.getSrHetro([eta, beta, epsilon, xc], npeople=n_dataset, nsteps=nsteps, t_end=t_end,time_step_multiplier=time_step_multiplier,external_hazard=ExtH)
                        sim_ds.toCsv(dataset_path)
                
                if test_mode:
                    # In test mode, create separate subfolders for each test
                    for test_idx in range(n_tests):
                        test_subfolder = os.path.join(submission_folder, f"test_{test_idx}")
                        os.makedirs(test_subfolder, exist_ok=True)
                        
                        # Create subfolders for output files for this test
                        out_folder = os.path.join(test_subfolder, "out_files")
                        os.makedirs(out_folder, exist_ok=True)
                        e_folder = os.path.join(test_subfolder, "e_files")
                        os.makedirs(e_folder, exist_ok=True)
                        log_folder = os.path.join(test_subfolder, "log")
                        os.makedirs(log_folder, exist_ok=True)
                        
                        if mcmc:
                            # Create MCMC specific folders for this test
                            h5_folder = os.path.join(test_subfolder, "h5_files")
                            os.makedirs(h5_folder, exist_ok=True)
                            out_mcmc = os.path.join(out_folder, f"out_files_mcmc")
                            os.makedirs(out_mcmc, exist_ok=True)
                            e_mcmc = os.path.join(e_folder, f"e_files_mcmc")
                            os.makedirs(e_mcmc, exist_ok=True)
                            
                            # Send array job to LSF cluster with test index and dataset folder
                            job = f"bsub -J \"{job_name}_test{test_idx}[1-{n_jobs}]\" -R 'rusage[mem={memory}GB]' -oo {out_mcmc}/%J_%I.o -eo {e_mcmc}/%J_%I.e -q {queue} {run_file_mcmc} {log_folder} {config_path} {h5_folder} {config_name} {test_idx} {datasets_folder}" 
                            output = subprocess.run(job, shell=True, capture_output=True, text=True)
                            print(f'mcmc test {test_idx}')
                            print(output.stdout)
                            job_id = re.search(r'Job <(\d+)>', output.stdout).group(1)
                            subject = f"Job {job_name}_test{test_idx} {job_id} ended" 
                else:
                    # Normal mode - single submission structure
                    out_folder = os.path.join(submission_folder, "out_files")
                    os.makedirs(out_folder, exist_ok=True)
                    e_folder = os.path.join(submission_folder, "e_files")
                    os.makedirs(e_folder, exist_ok=True)
                    log_folder = os.path.join(submission_folder, "log")
                    os.makedirs(log_folder, exist_ok=True)
                    
                    if mcmc:
                        # Create MCMC specific folders
                        h5_folder = os.path.join(submission_folder, "h5_files")
                        os.makedirs(h5_folder, exist_ok=True)
                        out_mcmc = os.path.join(out_folder, f"out_files_mcmc")
                        os.makedirs(out_mcmc, exist_ok=True)
                        e_mcmc = os.path.join(e_folder, f"e_files_mcmc")
                        os.makedirs(e_mcmc, exist_ok=True)

                        # Send array job to LSF cluster
                        job = f"bsub -J \"{job_name}[1-{n_jobs}]\" -R 'rusage[mem={memory}GB]' -oo {out_mcmc}/%J_%I.o -eo {e_mcmc}/%J_%I.e -q {queue} {run_file_mcmc} {log_folder} {config_path} {h5_folder} {config_name} {0} {0}" 
                        output = subprocess.run(job, shell=True, capture_output=True, text=True)
                        print('mcmc')
                        print(output.stdout)
                        job_id = re.search(r'Job <(\d+)>', output.stdout).group(1)
                        subject = f"Job {job_name} {job_id} ended" 
            else:
                submission_folder = None
                out_folder = None
                e_folder = None
                log_folder = None
                if test_mode:
                    datasets_folder = None

            if mcmc and args.no_submit:
                if test_mode:
                    print(f"--no-submit specified; would submit {n_tests} test jobs (skipping).")
                else:
                    print("--no-submit specified; skipping cluster submission.")

            print("config_path: ", args.config_path)
            print("excel_config_name: ", config_name)
            print("nsteps: ", nsteps)
            print("npeople: ", npeople)
            print("nsteps: ", nsteps)
            print("t_end: ", t_end)
            print("n_jobs: ", n_jobs)
            print("nwalkers: ", nwalkers)
            print("n_mcmc_steps: ", n_mcmc_steps)
            print("job_name: ", job_name)
            print("memory: ", memory)
            print("h5_file: ", h5_file)
            print("folder: ", folder)
        return
    else:
        config = cl.read_configs(config_path)
        nsteps = int(config.get('DEFAULT', 'nsteps'))
        npeople = int(config.get('DEFAULT', 'npeople'))
        t_end = int(config.get('DEFAULT', 't_end'))
        n_jobs = int(config.get('DEFAULT', 'n_jobs'))
        nwalkers = int(config.get('DEFAULT', 'nwalkers'))
        nsteps = int(config.get('DEFAULT', 'nsteps'))
        job_name = config.get('DEFAULT', 'job_name')
        memory = config.get('DEFAULT', 'initial_memory')
        h5_file = config.get('DEFAULT', 'h5_file_name')
        folder = config.get('DEFAULT', 'folder')
        name = config.get('DEFAULT', 'name')
        run_file_mcmc = config.get('DEFAULT', 'run_file_mcmc')
        n_mcmc_steps = int(config.get('DEFAULT', 'n_mcmc_steps'))
        queue = config.get('DEFAULT', 'queue')
        mcmc = config.getboolean('DEFAULT', 'mcmc')



    # Create submission structure unless running with --no-submit
    if not args.no_submit:
        # Create a subfolder for the current submission date
        submission_date = datetime.datetime.now().strftime("%d_%m_%Y")
        submission_folder = os.path.join(folder, f"{name}_submit_{submission_date}")
        cl.add_submition_folder(config, submission_folder, config_path)
        os.makedirs(submission_folder, exist_ok=True)
        
        # Create subfolders for output files
        out_folder = os.path.join(submission_folder, "out_files")
        os.makedirs(out_folder, exist_ok=True)
        e_folder = os.path.join(submission_folder, "e_files")
        os.makedirs(e_folder, exist_ok=True)
        
        log_folder = os.path.join(submission_folder, "log")
        os.makedirs(log_folder, exist_ok=True)
    else:
        submission_folder = None
        out_folder = None
        e_folder = None
        log_folder = None


    if mcmc and not args.no_submit:
        #create MCMC specific folders
        h5_folder = os.path.join(submission_folder, "h5_files")
        os.makedirs(h5_folder, exist_ok=True)
        out_mcmc = os.path.join(out_folder, f"out_files_mcmc")
        os.makedirs(out_mcmc, exist_ok=True)
        e_mcmc = os.path.join(e_folder, f"e_files_mcmc")
        os.makedirs(e_mcmc, exist_ok=True)

        # Send array job to LSF cluster
        job = f"bsub -J \"{job_name}[1-{n_jobs}]\" -R 'rusage[mem={memory}GB]' -oo {out_mcmc}/%J_%I.o -eo {e_mcmc}/%J_%I.e -q {queue} {run_file_mcmc} {log_folder} {config_path} {h5_folder}" 
        output = subprocess.run(job, shell=True, capture_output=True, text=True)
        print('mcmc')
        print(output.stdout)
        job_id = re.search(r'Job <(\d+)>', output.stdout).group(1)
        subject = f"Job {job_name} {job_id} ended" 
        cu.send_email(subject=subject, message=f"stdout: {output.stdout}", when_a_jobe_ends=job_id)
    elif mcmc and args.no_submit:
        print("--no-submit specified; skipping cluster submission.")


    print("config_path: ",args.config_path) 
    if is_excel:
        print("excel_config_name: ", specific_configs_list[0] if specific_configs_list else "")
    print("nsteps: ",nsteps)
    print("npeople: ",npeople)
    print("nsteps: ",nsteps)
    print("t_end: ",t_end)
    print("n_jobs: ",n_jobs)
    print("nwalkers: ",nwalkers)
    print("n_mcmc_steps: ",n_mcmc_steps)
    print("job_name: ",job_name)
    print("memory: ",memory)
    print("h5_file: ",h5_file)
    print("folder: ",folder)
    


if __name__ == "__main__":
    main()