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
import torch.nn.functional as F
from data import get_datasets
from config import WEIGHTS_DIR, DATA_ROOT, DATASETS
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from torchvision.models import vgg13
from metrics import eval_loader
from std_eval import set_deterministic

set_deterministic(31)

def prep_afib_dataset(dataset_name, window, nT=10, batch_size=512, PATH=WEIGHTS_DIR, device="cuda"):

    dataset_train, dataset_test = get_datasets(mode='ecg+labels', datasets=[dataset_name,], window=window)   ###

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
            save_path = os.path.join(DATA_ROOT, dataset_name)
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_fake_ecgs_{window}sec.npy"),  fake_ecgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_real_ecgs_{window}sec.npy"),  real_ecgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_real_ppgs_{window}sec.npy"),  real_ppgs[1:])
            np.save(os.path.join(save_path, f"afib_{'test' if idx else 'train'}_labels_{window}sec.npy"),  labels[1:])

def eval_afib(
    EVAL_DATASETS,
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
        datasets=EVAL_DATASETS,
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

    # Train loop
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
        acc_real_ecg, f1_real_ecg = eval_loader(test_loader_real_ecg, model, device)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
              f"val_acc (real ecg): {acc_real_ecg:.4f} - val_f1: {f1_real_ecg:.4f}")

    # Final evaluation on all three test sets
    acc_real_ecg_final,  f1_real_ecg_final  = eval_loader(test_loader_real_ecg, model, device)
    acc_real_ppg_final,  f1_real_ppg_final  = eval_loader(test_loader_real_ppg, model, device)
    acc_fake_ecg_final,  f1_fake_ecg_final  = eval_loader(test_loader_fake_ecg, model, device)

    metrics = {
        "acc_real_ecg":  acc_real_ecg_final,
        "f1_real_ecg":   f1_real_ecg_final,
        "acc_real_ppg":  acc_real_ppg_final,
        "f1_real_ppg":   f1_real_ppg_final,
        "acc_fake_ecg":  acc_fake_ecg_final,
        "f1_fake_ecg":   f1_fake_ecg_final,
    }
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

    # TABLE 4 results: AFib detection
    print("\n******* Atrial Fibrillation Detection (Table 4) results *******")
    for dataset_name in ['MIMIC-AFib',]:    # We can add more datasets later, each for afib detection.

        # Prepare AFib dataset for every dataset (for VGG13, training + evaluation)
        prep_afib_dataset(dataset_name, window=4, nT=10, batch_size=512, device="cuda")
            
        tracked_metrics, _ = eval_afib(EVAL_DATASETS=[dataset_name,],
                                    window=4,
                                    batch_size=64,
                                    epochs=25,
                                    lr=1e-4,
                                    device='cuda')
        print(f"\n{dataset_name} (Real ECG): Accuracy={tracked_metrics['acc_real_ecg_final']}, F1={tracked_metrics['f1_real_ecg_final']}")
        print(f"{dataset_name} (Real PPG): Accuracy={tracked_metrics['acc_real_ppg_final']}, F1={tracked_metrics['f1_real_ppg_final']}")
        print(f"{dataset_name} (Fake ECG): Accuracy={tracked_metrics['acc_fake_ecg_final']}, F1={tracked_metrics['f1_fake_ecg_final']}")
        print("-"*1000)

