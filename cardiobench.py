from std_eval import eval_diffusion
from zipfile import ZipFile
import os

import torch
torch.autograd.set_detect_anomaly(True)
import random
from tqdm import tqdm
import warnings
from metrics import *
warnings.filterwarnings("ignore")
import numpy as np
from diffusion import load_pretrained_DPM
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data import get_datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import WEIGHTS_DIR, DATA_ROOT, DATASETS

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from torchvision.models import vgg13

def prep_afib_dataset(window, EVAL_DATASETS, nT=10, batch_size=512, PATH=WEIGHTS_DIR, save_path=DATA_ROOT, device="cuda"):

    dataset_train, dataset_test = get_datasets(mode='ecg+labels', datasets=EVAL_DATASETS, window=window)   ###

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=64)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=64)

    dpm, Conditioning_network1, Conditioning_network2 = load_pretrained_DPM(
        PATH=PATH,
        nT=nT,
        type="RDDM",
        device="cuda"
    )
    
    dpm = nn.DataParallel(dpm)
    Conditioning_network1 = nn.DataParallel(Conditioning_network1)
    Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    dpm.eval()
    Conditioning_network1.eval()
    Conditioning_network2.eval()

    with torch.no_grad():

        for idx, loader in enumerate([trainloader, testloader]):
            fake_ecgs = np.zeros((1, 128*window))
            real_ecgs = np.zeros((1, 128*window))
            real_ppgs = np.zeros((1, 128*window))
            labels = np.zeros(1)

            for y_ecg, x_ppg, ecg_roi, label in tqdm(loader):

                x_ppg = x_ppg.float().to(device)
                y_ecg = y_ecg.float().to(device)
                ecg_roi = ecg_roi.float().to(device)

                generated_windows = []

                for ppg_window in torch.split(x_ppg, 128*4, dim=-1):
                    
                    if ppg_window.shape[-1] != 128*4:
                        
                        ppg_window = F.pad(ppg_window, (0, 128*4 - ppg_window.shape[-1]), "constant", 0)

                    ppg_conditions1 = Conditioning_network1(ppg_window)
                    ppg_conditions2 = Conditioning_network2(ppg_window)

                    xh = dpm(
                        cond1=ppg_conditions1, 
                        cond2=ppg_conditions2, 
                        mode="sample", 
                        window_size=128*4
                    )
                    
                    generated_windows.append(xh.cpu().numpy())

                xh = np.concatenate(generated_windows, axis=-1)[:, :, :128*window]

                fake_ecgs = np.concatenate((fake_ecgs, xh.reshape(-1, 128*window)))
                real_ecgs = np.concatenate((real_ecgs, y_ecg.reshape(-1, 128*window).cpu().numpy()))
                real_ppgs = np.concatenate((real_ppgs, x_ppg.reshape(-1, 128*window).cpu().numpy()))
                labels = np.concatenate((labels, label.cpu().numpy()))
            
            #### Here, we save the fake_ecgs, real_ecgs, real_ppgs, labels.
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_fake_ecgs_{window}sec.npy"),  fake_ecgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_real_ecgs_{window}sec.npy"),  real_ecgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_real_ppgs_{window}sec.npy"),  real_ppgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_labels_{window}sec.npy"),  labels[1:])

from sklearn.metrics import f1_score  # if you prefer not to depend on sklearn, we can do it manually
# or we can compute F1 manually below; I'll do manual F1 to avoid extra deps.

def eval_afib(
    datasets,
    PATH=DATA_ROOT,
    window=4,
    batch_size=64,
    epochs=10,
    lr=1e-4,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load datasets: STFT-based spectrogram datasets
    train_ds, test_ds_real_ecg, test_ds_real_ppg, test_ds_fake_ecg = get_datasets(
        mode='afib',
        data_root=PATH,
        datasets=datasets,
        window=window
    )

    # Quick class counts (0, 1)
    def print_counts(name, ds):
        y = ds.labels
        zeros, ones = (y == 0).sum(), (y == 1).sum()
        print(f"{name}: total={len(y)}, class0={zeros}, class1={ones}")
    print_counts("Train",          train_ds)
    print_counts("Test real ECG",  test_ds_real_ecg)
    print_counts("Test real PPG",  test_ds_real_ppg)
    print_counts("Test fake ECG",  test_ds_fake_ecg)
    # -------------------------------------

    train_loader         = DataLoader(train_ds,           batch_size=batch_size, shuffle=True,  num_workers=4)
    test_loader_real_ecg = DataLoader(test_ds_real_ecg,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_real_ppg = DataLoader(test_ds_real_ppg,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_fake_ecg = DataLoader(test_ds_fake_ecg,   batch_size=batch_size, shuffle=False, num_workers=4)

    # VGG-13 model (binary head)
    model = vgg13(weights=None)   # or weights="IMAGENET1K_V1" if you want ImageNet init
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)   # 1 logit for binary AFib / non-AFib
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- accuracy + F1 for a loader ----
    def eval_loader(loader):
        model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device)                 # (B, 3, H, W)
                y = y.to(device).view(-1)        # (B,)
                logits = model(X).squeeze(1)     # (B,)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                correct += (preds == y).sum().item()
                total   += y.numel()

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())

        acc = correct / total if total > 0 else 0.0
        if total > 0:
            all_preds    = torch.cat(all_preds).numpy()
            all_targets  = torch.cat(all_targets).numpy()
            # manual F1 (binary)
            tp = ((all_preds == 1) & (all_targets == 1)).sum()
            fp = ((all_preds == 1) & (all_targets == 0)).sum()
            fn = ((all_preds == 0) & (all_targets == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            f1 = 0.0
        return acc, f1
    # ------------------------------------

    # train loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)               # (B, 3, H, W)
            y = y.to(device).view(-1, 1)   # (B, 1)

            optimizer.zero_grad()
            logits = model(X)              # (B, 1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        avg_loss = running_loss / len(train_ds)

        # validation on real ECG each epoch (upper limit)
        acc_real_ecg, f1_real_ecg = eval_loader(test_loader_real_ecg)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
              f"val_acc (real ecg): {acc_real_ecg:.4f} - val_f1: {f1_real_ecg:.4f}")

    # Final evaluation on all three test sets
    acc_real_ecg_final,  f1_real_ecg_final  = eval_loader(test_loader_real_ecg)
    acc_real_ppg_final,  f1_real_ppg_final  = eval_loader(test_loader_real_ppg)
    acc_fake_ecg_final,  f1_fake_ecg_final  = eval_loader(test_loader_fake_ecg)

    metrics = {
        "acc_real_ecg":  acc_real_ecg_final,
        "f1_real_ecg":   f1_real_ecg_final,
        "acc_real_ppg":  acc_real_ppg_final,
        "f1_real_ppg":   f1_real_ppg_final,
        "acc_fake_ecg":  acc_fake_ecg_final,
        "f1_fake_ecg":   f1_fake_ecg_final,
    }
    print("Final AFib metrics:", metrics)

    return metrics, model









if __name__ == "__main__":
    # TABLE 3 results
    print("\n******* Heart Rate estimation (Table 3) results *******")
    for dataset_name in ["WESAD", "DALIA"]:
        
        tracked_metrics = eval_diffusion(
            window=8,
            EVAL_DATASETS=[dataset_name],
            nT=10,
        )
        print(f"\n{dataset_name}: Mean Absolute Error (BPM) is {tracked_metrics['MAE_HR_ECG']}")
        print("-"*1000)

