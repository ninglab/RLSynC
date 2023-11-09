# RLSynC

This is the official repository for RLSynC, a reinforcement-learning framework for synthon completion in semi-template-based retrosynthesis.

[ArXiv Preprint](https://arxiv.org/abs/2309.02671)

## Setup

Our tool comes with a Docker image for easy installation with dependencies.  To build, run:
```
docker build -t rlsync .
```

To instantiate and run a container of this image, we recommend mounting the data directory as a volume using the following command:
```
docker run -v ${PWD}/data:/rlsync/data -v -it rlsync
```

### Alternative Self-Install

Alternatively, users can reproduce the installation steps from our [Dockerfile](Dockerfile) in their own environment.
It should be sufficient to `pip install -r requirements.txt`, install the MolecularTransformer submodule, and `pip install -e .` from the root folder of the repository.

## Training, Augmentation, and Evaluation Scripts

All scripts required to train, augment the training set, and evaluate test results are in the `scripts` directory.

### `generate_ground_truth_original.py`

This script generates the initial training data based on the ground truth reactants in the provided dataset.

### `generate_agent_actions.py`

This script generates training data augmentations through online interactions of previously trained agents which select exactly one action at each step.

### `generate_topn_actions.py`

This script generates training data augmentations through online interactions of previously trained agents which select k actions at each step and produce the top-N reactions at the terminal step.

### `offline_train.py`

This script trains an agent using offline training datasets.

### `apply_agents.py` / `apply_agents_parallel.py`

This script computes predictions for an evaluation dataset using trained agents.  The `_parallel` version uses multiple processes to parallelize the computation, leverageing the `pqdm` library to track progress.

### `metrics.py`

This script computes metrics (NDCG@N, MAP@, Diversity@N, Validity, etc.) on the output CSV of `apply_agents.py`.  It formats its output as a JSON file, and includes comprehensive records of rewards for further evaluation purposes.

### `compute_results.sh`

This script generates the results for later sections of this README.md file, provided that the best
model parameters are located at `data/rlsync/best.pt`.  If using Docker, avoid expanding the Docker image size by mounting the data directory into the container using the Docker volume feature.  By default, this script generates predictions by parallelizing over multiple CPUs.  You may consider changing the number of CPUs depending on your hardware.

Here is an example command which generates the results sequentially in the background using Docker:

```
docker run -d -v ${PWD}/data:/rlsync/data rlsync /rlsync/scripts/compute_results.sh
```

### `test.sh`

This script runs a simple set of integration tests on the codebase.  Every command in this script should exit with a zero return code.  Please note, the standard error and standard output should both see activity on a successful test.

### `keymetrics.py`

This script extracts the high-level metrics presented in the Results section of this README from the results JSON file computed by `metrics.py`.

## Results

These are the results of our final model, whose parameters can be found in `data/rlsync/best.pt`.

To compute these numbers, you can use this command:
```
docker run -d -v ${PWD}/data:/rlsync/data rlsync /rlsync/scripts/compute_results.sh
```

| **N** | **MAP@N** | **NDCG@N** | **Diversity@N** |
|:-----:|----------:|-----------:|----------------:|
|   1   |     0.927 |      0.927 |             N/A |
|   2   |     0.898 |      0.905 |           0.193 |
|   3   |     0.874 |      0.886 |           0.205 |
|   4   |     0.845 |      0.865 |           0.231 |
|   5   |     0.822 |      0.847 |           0.243 |
|   6   |     0.803 |      0.832 |           0.251 |
|   7   |     0.784 |      0.817 |           0.255 |
|   8   |     0.769 |      0.805 |           0.260 |
|   9   |     0.754 |      0.793 |           0.265 |
|   10  |     0.741 |      0.782 |           0.272 |
