import torch
import numpy as np
from tqdm import tqdm
import neurokit2 as nk
import sklearn.preprocessing as skp
from torch.utils.data import Dataset, DataLoader
from config import DATA_ROOT, DATASETS
import os
from preprocessing import _convert_freq_and_align, _convert_ecg_to_spectrogram_batch

class ECGDataset(Dataset):
    def __init__(self, ecg_data, ppg_data, label_data):
        self.ecg_data = ecg_data
        self.ppg_data = ppg_data
        self.label_data = label_data

    def __getitem__(self, index):

        ecg = self.ecg_data[index]
        ppg = self.ppg_data[index]
        
        window_size = ecg.shape[-1]

        ppg = nk.ppg_clean(ppg.reshape(window_size), sampling_rate=128)
        ecg = nk.ecg_clean(ecg.reshape(window_size), sampling_rate=128, method="pantompkins1985")
        _, info = nk.ecg_peaks(ecg, sampling_rate=128, method="pantompkins1985", correct_artifacts=True, show=False)

        # Create a numpy array for ROI regions with the same shape as ECG
        ecg_roi_array = np.zeros_like(ecg.reshape(1, window_size))

        # Iterate through ECG R peaks and set values to 1 within the ROI regions
        roi_size = 32
        for peak in info["ECG_R_Peaks"]:
            roi_start = max(0, peak - roi_size // 2)
            roi_end = min(roi_start + roi_size, window_size)
            ecg_roi_array[0, roi_start:roi_end] = 1

        if self.label_data is not None:
            return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy(), self.label_data[index].copy() #, ppg_cwt.copy()
        else:
            return ecg.reshape(1, window_size).copy(), ppg.reshape(1, window_size).copy(), ecg_roi_array.copy() #, ppg_cwt.copy()

    def __len__(self):
        return len(self.ecg_data)

class ECGSpectDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels.astype(np.float32)

    def __getitem__(self, idx):
        s = self.specs[idx]             # (F, T)
        # per-sample normalization
        mu, sd = s.mean(), s.std()
        sd = sd if sd > 1e-6 else 1.0
        s = (s - mu) / sd

        # VGG expects 3xHxW -> tile spectrogram to 3 channels
        s3 = np.stack([s, s, s], axis=0).astype(np.float32) # (3, F, T)
        y = self.labels[idx]
        return torch.from_numpy(s3), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.specs)

def get_datasets(
    mode='ecg',     # 'ecg', 'ecg+labels', 'afrib'
    data_root=DATA_ROOT, 
    datasets=DATASETS,
    window=4,
    ):
    
    if mode == 'afrib':
        train_x_list = []
        train_y_list = []
        test_x_real_ppg_list = []
        test_x_real_ecg_list = []
        test_x_fake_ecg_list = []
        test_y_list = []
        for dataset in datasets:
            train_x = np.load(os.path.join(data_root, dataset, f'afrib_train_real_ecgs_{window}sec.npy'))
            train_y = np.load(os.path.join(data_root, dataset, f'afrib_train_labels_{window}sec.npy'))
            test_x_fake_ecg = np.load(os.path.join(data_root, dataset, f'afrib_test_fake_ecgs_{window}sec.npy'))
            test_x_real_ecg = np.load(os.path.join(data_root, dataset, f'afrib_test_real_ecgs_{window}sec.npy'))
            test_x_real_ppg = np.load(os.path.join(data_root, dataset, f'afrib_test_real_ppgs_{window}sec.npy'))
            test_y = np.load(os.path.join(data_root, dataset, f'afrib_test_labels_{window}sec.npy'))

            train_x_list.append(train_x)
            train_y_list.append(train_y)
            test_x_fake_ecg_list.append(test_x_fake_ecg)
            test_x_real_ecg_list.append(test_x_real_ecg)
            test_x_real_ppg_list.append(test_x_real_ppg)
            test_y_list.append(test_y)

        #### Here, we will convert to STFT (short time fourier transform)
        train_specs = _convert_ecg_to_spectrogram_batch(np.concatenate(train_x_list), fs=128)
        test_specs_real_ecg  = _convert_ecg_to_spectrogram_batch(np.concatenate(test_x_real_ecg_list),  fs=128)
        test_specs_fake_ecg = _convert_ecg_to_spectrogram_batch(np.concatenate(test_x_fake_ecg_list),  fs=128)
        test_specs_real_ppg = _convert_ecg_to_spectrogram_batch(np.concatenate(test_x_real_ppg_list),  fs=128)

        # Build datasets/loaders
        dataset_train = ECGSpectDataset(train_specs, np.concatenate(train_y_list))
        dataset_test_real_ecg  = ECGSpectDataset(test_specs_real_ecg, np.concatenate(test_y_list))
        dataset_test_real_ppg  = ECGSpectDataset(test_specs_real_ppg, np.concatenate(test_y_list))
        dataset_test_fake_ecg  = ECGSpectDataset(test_specs_fake_ecg, np.concatenate(test_y_list))
        
        return dataset_train, dataset_test_real_ecg, dataset_test_real_ppg, dataset_test_fake_ecg
    else:
        ecg_train_list = []
        ppg_train_list = []
        ecg_test_list = []
        ppg_test_list = []
        labels_train_list = []
        labels_test_list = []
        for dataset in datasets:

            ecg_train = np.load(data_root + dataset + f"/ecg_train_{window}sec.npy", allow_pickle=True).reshape(-1, 128*window)
            ppg_train = np.load(data_root + dataset + f"/ppg_train_{window}sec.npy", allow_pickle=True).reshape(-1, 128*window)
            
            ecg_test = np.load(data_root + dataset + f"/ecg_test_{window}sec.npy", allow_pickle=True).reshape(-1, 128*window)
            ppg_test = np.load(data_root + dataset + f"/ppg_test_{window}sec.npy", allow_pickle=True).reshape(-1, 128*window)

            ecg_train_list.append(ecg_train)
            ppg_train_list.append(ppg_train)
            ecg_test_list.append(ecg_test)
            ppg_test_list.append(ppg_test)

            if mode == 'ecg+labels':    # Finds labels npy in the same directory.
                labels_train = np.load(data_root + dataset + f"/labels_train_{window}sec.npy", allow_pickle=True).reshape(-1)
                labels_test = np.load(data_root + dataset + f"/labels_test_{window}sec.npy", allow_pickle=True).reshape(-1)
                labels_train_list.append(labels_train)
                labels_test_list.append(labels_test)

        ecg_train = np.nan_to_num(np.concatenate(ecg_train_list).astype("float32"))
        ppg_train = np.nan_to_num(np.concatenate(ppg_train_list).astype("float32"))

        ecg_test = np.nan_to_num(np.concatenate(ecg_test_list).astype("float32"))
        ppg_test = np.nan_to_num(np.concatenate(ppg_test_list).astype("float32"))

        labels_train = np.nan_to_num(np.concatenate(labels_train_list).astype("float32")) if labels_train_list else None
        labels_test = np.nan_to_num(np.concatenate(labels_test_list).astype("float32")) if labels_test_list else None

        dataset_train = ECGDataset(
            skp.minmax_scale(ecg_train, (-1, 1), axis=1),
            skp.minmax_scale(ppg_train, (-1, 1), axis=1),
            labels_train
        )
        dataset_test = ECGDataset(
            skp.minmax_scale(ecg_test, (-1, 1), axis=1),
            skp.minmax_scale(ppg_test, (-1, 1), axis=1),
            labels_test
        )
        return dataset_train, dataset_test