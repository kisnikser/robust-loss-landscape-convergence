# Code

This repository contains source code for our paper. The code includes data processing, model building, and visualization of results.

## Installation

Clone the repo:

```bash
git clone https://github.com/kisnisker/robust-loss-landscape-convergence.git
cd robust-loss-landscape-convergence/code
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Pick a Hydra config.** Every experiment is described by a YAML file under [code/configs](code/configs). Select the dataset split you need (for example [code/configs/CIFAR10](code/configs/CIFAR10)) and note the file name you want to run (such as `conv_channels_sigm8.yaml`).
2. **Run a single visualization job.** Execute the trainer/visualizer directly from the `code` directory by pointing Hydra to the config you selected:

   ```bash
   python models_delta_visualize.py --config-path configs/CIFAR10 --config-name conv_channels_sigm8
   ```

   The script trains the specified model, computes the $\Delta_k$ statistics, and stores the generated figures wherever `logging.savefig_path` points inside the config.

3. **Sweep across a folder of configs.** To reproduce the paper sweeps, use [code/run_exps.sh](code/run_exps.sh). The script iterates over all YAML files inside the `config_path` you set at the top and launches `models_delta_visualize.py` for each one:

   ```bash
   # edit config_path inside the script or export it before running
   bash run_exps.sh
   ```

4. **Compare trained models or explore deltas programmatically.**
   - [code/delta_models_compare.py](code/delta_models_compare.py) runs the same Hydra-powered pipeline but focuses on comparing mean $k\Delta_k$ curves across architectures.
   - [code/src/calc_delta_visualize.py](code/src/calc_delta_visualize.py) exposes a `DeltaCalcVisualizer` utility that lets you vary estimator parameters (number of samples, integration mode, etc.) and render the corresponding $\Delta_k$ trajectories inside custom Python scripts or notebooks.

Interactive notebooks inside `code/*.ipynb` mirror these steps if you prefer exploratory runs in Jupyter.
