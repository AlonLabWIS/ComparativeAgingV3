#!/usr/bin/env python3
"""
Script to create multiple copies of the MCMC analysis notebook with different config names,
execute them, and convert to PDF.

Usage:
    python run_multiple_configs.py [configs_file]

If configs_file is not provided, defaults to 'configs_to_run.txt'
"""

import os
import sys
import subprocess
import re
import json
from pathlib import Path

try:
    import nbformat
except ImportError:
    print("Error: nbformat is required. Install with: pip install nbformat")
    sys.exit(1)


def read_config_names(config_file):
    """Read config names from file. Supports comma-separated or one per line."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        content = f.read().strip()
    
    # Try comma-separated first
    if ',' in content:
        configs = [name.strip() for name in content.split(',') if name.strip()]
    else:
        # One per line
        configs = [line.strip() for line in content.split('\n') if line.strip()]
    
    return configs


def modify_notebook_config_name(notebook_path, new_config_name):
    """Load notebook and modify config_name in Cell 1."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Cell 1 (index 1) contains the config_name assignment
    if len(nb.cells) < 2:
        raise ValueError(f"Notebook {notebook_path} doesn't have enough cells")
    
    cell = nb.cells[1]
    source = cell['source']
    
    # Replace config_name assignment using regex
    # Pattern matches: config_name = 'old_name' or config_name = "old_name"
    pattern = r"config_name\s*=\s*['\"]([^'\"]*)['\"]"
    replacement = f"config_name = '{new_config_name}'"
    
    if not re.search(pattern, source):
        raise ValueError(f"Could not find config_name assignment in Cell 1 of {notebook_path}")
    
    cell['source'] = re.sub(pattern, replacement, source)
    
    return nb


def create_notebook_copy(template_path, config_name, output_dir):
    """Create a copy of the notebook with modified config_name."""
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output filename: mcmc_analysis_<config_name>_baysian.ipynb
    output_filename = f"mcmc_analysis_{config_name}_baysian.ipynb"
    output_path = output_dir / output_filename
    
    # Load and modify notebook
    nb = modify_notebook_config_name(template_path, config_name)
    
    # Save modified notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Created notebook: {output_path}")
    return output_path


def find_nbconvert_command():
    """Find the nbconvert command, trying multiple methods."""
    # Try different ways to invoke nbconvert
    candidates = [
        [sys.executable, '-m', 'nbconvert'],
        ['python3', '-m', 'nbconvert'],
        ['python', '-m', 'nbconvert'],
        ['jupyter', 'nbconvert'],
        ['jupyter-nbconvert'],
    ]
    
    for cmd in candidates:
        try:
            # Test if the command works by checking help
            result = subprocess.run(
                cmd + ['--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 or 'nbconvert' in result.stdout.lower() or 'nbconvert' in result.stderr.lower():
                print(f"Using nbconvert command: {' '.join(cmd)}")
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    # If none worked, return default and let it fail with a better error
    return [sys.executable, '-m', 'nbconvert']


# Cache the nbconvert command
_nbconvert_cmd = None

def get_nbconvert_command():
    """Get the nbconvert command, caching the result."""
    global _nbconvert_cmd
    if _nbconvert_cmd is None:
        _nbconvert_cmd = find_nbconvert_command()
    return _nbconvert_cmd


def execute_notebook(notebook_path):
    """Execute a notebook using nbconvert."""
    notebook_path = Path(notebook_path)
    print(f"Executing notebook: {notebook_path}")
    
    nbconvert_cmd = get_nbconvert_command()
    
    try:
        result = subprocess.run(
            nbconvert_cmd + ['--execute', '--inplace', str(notebook_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Successfully executed: {notebook_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Could not find nbconvert command. Tried: {' '.join(nbconvert_cmd)}")
        print("Please ensure nbconvert is installed and accessible.")
        print("You can install it with: pip install nbconvert")
        print("Or try: python -m pip install nbconvert")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing {notebook_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def convert_to_pdf(notebook_path):
    """Convert executed notebook to PDF, showing only outputs and markdown (no code cells)."""
    notebook_path = Path(notebook_path)
    print(f"Converting to PDF: {notebook_path}")
    
    nbconvert_cmd = get_nbconvert_command()
    
    try:
        result = subprocess.run(
            nbconvert_cmd + ['--to', 'pdf', '--no-input', str(notebook_path)],
            capture_output=True,
            text=True,
            check=True
        )
        pdf_path = notebook_path.with_suffix('.pdf')
        print(f"Successfully created PDF: {pdf_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Could not find nbconvert command. Tried: {' '.join(nbconvert_cmd)}")
        print("Please ensure nbconvert is installed and accessible.")
        print("You can install it with: pip install nbconvert")
        print("Or try: python -m pip install nbconvert")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error converting {notebook_path} to PDF:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        configs_file = sys.argv[1]
    else:
        configs_file = 'configs_to_run.txt'
    
    # Get script directory and set paths
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    
    configs_file = Path(configs_file)
    template_notebook = script_dir / 'analysis' / 'mcmc_analyisis_template_baysian.ipynb'
    output_dir = script_dir / 'analysis'
    
    # Validate inputs
    if not configs_file.exists():
        print(f"Error: Config file not found: {configs_file}")
        sys.exit(1)
    
    if not template_notebook.exists():
        print(f"Error: Template notebook not found: {template_notebook}")
        sys.exit(1)
    
    # Read config names
    print(f"Reading config names from: {configs_file}")
    config_names = read_config_names(configs_file)
    print(f"Found {len(config_names)} config(s): {config_names}")
    
    # Process each config
    successful = []
    failed = []
    
    for config_name in config_names:
        print(f"\n{'='*60}")
        print(f"Processing config: {config_name}")
        print(f"{'='*60}")
        
        try:
            # Create notebook copy
            notebook_path = create_notebook_copy(template_notebook, config_name, output_dir)
            
            # Execute notebook
            if not execute_notebook(notebook_path):
                failed.append(config_name)
                print(f"Failed to execute notebook for {config_name}, skipping PDF conversion")
                continue
            
            # Convert to PDF
            if not convert_to_pdf(notebook_path):
                failed.append(config_name)
                print(f"Failed to convert {config_name} to PDF")
                continue
            
            successful.append(config_name)
            print(f"Successfully completed: {config_name}")
            
        except Exception as e:
            print(f"Error processing {config_name}: {e}")
            failed.append(config_name)
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(successful)}")
    if successful:
        for name in successful:
            print(f"  - {name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for name in failed:
            print(f"  - {name}")
    
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

