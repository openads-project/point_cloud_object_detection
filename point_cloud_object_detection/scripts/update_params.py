import os
import shutil
from pathlib import Path
import sys

import yaml


# constants
package_path = Path(__file__).parent.parent.absolute()

# inputs
if len(sys.argv) < 3:
    raise RuntimeError(
        'Usage: python update_params.py <INPUT_FILE> <OUTPUT_FILE> [<NODE_NAME> <NAMESPACE>]'
    )
params_file = Path(sys.argv[1])
combined_params_file = Path(sys.argv[2])

if not params_file.is_absolute():
    params_file = package_path / params_file

if not combined_params_file.is_absolute():
    combined_params_file = package_path / combined_params_file

if not params_file.exists():
    raise FileNotFoundError('File ' + str(params_file) + ' not found')

# create folder if needed
if not combined_params_file.parent.exists():
    combined_params_file.parent.mkdir(parents=True)

# processing params file
with open(params_file, 'r') as infile:
    params = yaml.safe_load(infile)

# define wildcard prefix
prefix = '/**'

# check arguments if wildcard prefix is not contained in params file
if prefix not in params.keys():
    prefix = '/'
    if len(sys.argv) == 4:
        prefix = os.path.join(prefix, sys.argv[3])
    if len(sys.argv) == 5:
        prefix = os.path.join(prefix, sys.argv[4], sys.argv[3])
    if prefix not in params.keys():
        raise KeyError('Prefix ' + prefix + ' not found in ' + str(params_file))

# no model_config merging; the node loads the manifest directly
if 'model_config' in params[prefix]['ros__parameters']:
    raise RuntimeError('model_config is deprecated; use model_manifest_path instead')

shutil.copyfile(params_file, combined_params_file)
