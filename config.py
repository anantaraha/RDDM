DATA_ROOT = '../datasets/'                                      # root dir for datasets
WEIGHTS_DIR = '../weights/'                                     # root dir for weights
DATASETS = ['WESAD', 'CAPNO', 'DALIA', 'BIDMC', 'MIMIC-AFib']   # list of dataset dirnames
SAVE_AFTER_EPOCHS = 8#80                                        # save weights every ? epochs

RDDM_TRAIN_CONFIG = {
    "n_epoch": 5,#1000,
    "batch_size": 64,#128*4,
    "nT":10,
    "device": "cuda",
    "attention_heads": 8,
    "cond_mask": 0.0,
    "alpha1": 100,
    "alpha2": 1,
    "PATH": WEIGHTS_DIR
}

DDPM_TRAIN_CONFIG = {
    "n_epoch": 5,#1000,
    "batch_size": 64,#128*4,
    "nT":10,
    "device": "cuda",
    "attention_heads": 8,
    "PATH": WEIGHTS_DIR
}