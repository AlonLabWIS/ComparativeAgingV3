#!/bin/bash
# Check the CPU model
cpu_model=$(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)

output_path=$1
config_path=$2
h5_path=$3
config_name=$4
test_idx=$5
datasets_folder=$6

module load miniconda/23.3.1-0_environmentally
conda activate srtools 
head /proc/cpuinfo 
python_check=$(python3 -c "import sys; print(sys.version)" 2>&1)
                
if [[ $? -ne 0 ]]; then
    echo "Error: Python check failed on $(hostname). Python output:"
    echo "$python_check"
    exit 1
else
    echo "Python check passed on $(hostname). Python version:"
    echo "$python_check"
fi

echo "$output_path"
python3 ~/Baysian03/run_file_mcmc_excel.py "$config_path" "$h5_path" "$LSB_JOBINDEX" "$config_name" "$test_idx" "$datasets_folder" > "$output_path/"${LSB_JOBID}"_"${LSB_JOBINDEX}".out"
