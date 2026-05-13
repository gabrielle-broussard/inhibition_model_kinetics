# inhibition_model_kinetics
Python workflow used for "Microbial Substrate Inhibition Model Robustness: Structural Failure Analysis Under Gaussian Noise Conditions." Originally coded and employed in Spyder Version 6.

### Inhibition Kinetics Data-Generation Code
This repository contains data-generation code for the following substrate-inhibition model recovery study: 'Microbial Substrate Inhibition Model Robustness: Structural Failure Analysis Under Gaussian Noise Conditions.'

### Files
'generate_model_recovery_data.py' generates seeded synthetic model-recovery datasets and CSV summaries.
'generate_timeseries_data.py' generates optional all-model batch time-series CSV datasets.
'run_data_generation_workflow.py' runs the data-generation workflow.
'literature_parameter_sets.csv' stores the literature-derived parameter sets used by the scripts.
'.gitignore' excludes generated outputs, caches, and local environment files from GitHub.

### Basic run
python run_data_generation_workflow.py
This creates seeded synthetic model outputs under: outputs/

Optional time-series CSV run: python run_data_generation_workflow.py --run_timeseries
This creates time-series CSV outputs under: timeseries_all_models_data/csv/
