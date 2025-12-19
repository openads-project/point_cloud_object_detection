import io
import os
from pathlib import Path
import shutil
import sys

import yaml


def slikaf2ros(config):
    if 'predicted_class_names' in config:
        # Discard label class names per predicted class name
        if type(config['predicted_class_names']) in [list, set]:
            config['predicted_class_names'] = list(
                config['predicted_class_names'])
        else:
            config['predicted_class_names'] = list(
                config['predicted_class_names'].keys())
    if 'anchors' in config:
        # Unravel anchors
        old_anchors = config['anchors']
        config['anchors'] = {'size': len(old_anchors)}
        for i in range(len(old_anchors)):
            config['anchors']['anchor_' + str(i)] = [
                float(j.strip()) for j in old_anchors[i].split(',')
            ]
    if 'nms_score_threshold' in config and not isinstance(
            config['nms_score_threshold'], list):
        # For scalar score thresholds, transform it into list.
        config['nms_score_threshold'] = [config['nms_score_threshold']]
    return config


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

    # check if nodename and namespace is provided
    if len(sys.argv) == 4:
        prefix = os.path.join(prefix, sys.argv[3])

    if len(sys.argv) == 5:
        prefix = os.path.join(prefix, sys.argv[4], sys.argv[3])

    # check if processed prefix is contained in params file
    if prefix not in params.keys():
        raise KeyError('Prefix ' + prefix + ' not found in ' + str(params_file))

# only process if model_config exist
if 'model_config' in params[prefix]['ros__parameters'].keys():

    config_file = os.path.join(
        package_path, params[prefix]['ros__parameters']['model_config'])

    # processing model config file
    with open(config_file, 'r') as configfile:
        config = yaml.safe_load(configfile)
        config = slikaf2ros(config)

        # modify config
        if prefix not in config.keys():

            modified_config: dict = {}
            modified_config[prefix] = {}
            modified_config[prefix]['ros__parameters'] = config

            config = modified_config

        # add config to params
        for key, value in config[prefix]['ros__parameters'].items():
            if key != 'model_config':
                params[prefix]['ros__parameters'][key] = value

    # save new params file
    with io.open(combined_params_file, 'w', encoding='utf8') as outfile:
        yaml.dump(params, outfile, default_flow_style=False, allow_unicode=True)

else:
    # If no model_config in params, just copy params
    shutil.copyfile(params_file, combined_params_file)
