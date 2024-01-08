"""
Global configs for running code of time-series imputation survey.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# random seeds for five rounds of experiments
RANDOM_SEEDS = [2023, 2024, 2025, 2026, 2027]
# number of threads for pytorch
TORCH_N_THREADS = 1

# whether to apply lazy-loading strategy (help save memory) to read datasets
LAZY_LOAD_DATA = False

# set general configs for model training
BATCH_SIZE = 32
MAX_N_EPOCHS = 300  # max epochs for training
PATIENCE = 10  # patience for early stopping
DEVICE = "cuda:0"

# set the path for saving all models
RESULTS_SAVING_PATH = "saved_results"
