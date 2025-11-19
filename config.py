DATA_ROOT = '../datasets/'                                      # root dir for datasets
WEIGHTS_DIR = '../weights/'                                     # root dir for weights
DATASETS = ['WESAD', 'CAPNO', 'DALIA', 'BIDMC', 'MIMIC-AFib']   # list of dataset dirnames
SAVE_AFTER_EPOCHS = 4                                           # save weights every ? epochs (80 previously)

RDDM_TRAIN_CONFIG = {
    "n_epoch": 5,           #1000, -> Total epochs to train
    "batch_size": 64,       #128*4,-> Batch size
    "nT":10,
    "device": "cuda",
    "attention_heads": 8,
    "cond_mask": 0.0,
    "alpha1": 100,
    "alpha2": 1,
    "PATH": WEIGHTS_DIR     # -> Where to save weights ?
}

DDPM_TRAIN_CONFIG = {
    "n_epoch": 5,#1000,
    "batch_size": 64,#128*4,
    "nT":10,
    "device": "cuda",
    "attention_heads": 8,
    "PATH": WEIGHTS_DIR
}