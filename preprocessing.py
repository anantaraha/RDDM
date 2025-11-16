import os, pickle, mat73, numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from zipfile import ZipFile
from scipy.io import loadmat

def _resamp(x, fs_src, fs_tgt=128):
    fs_src = int(float(np.asarray(fs_src)))
    return x.astype(np.float32) if fs_src == fs_tgt else resample_poly(x, fs_tgt, fs_src).astype(np.float32)

def _butter_hp(x, fs, fc=0.5, order=4):
    '''
    x: 1D numpy array of the signal to filter
    fs: Sampling rate of x in Hz, used to normalize the cutoff.
    fc: High-pass cutoff frequency in Hz, frequencies below this are attenuated.
    order: Filter order. Higher = sharper cutoff but more phase ripple/instability risk.
    '''
    np.nan_to_num(x, copy=False)
    b, a = butter(order, fc/(fs/2), btype="highpass")
    return filtfilt(b, a, x).astype(np.float32)

def _butter_bp(x, fs, f1=0.5, f2=8.0, order=4):
    '''
    Nyquist constraint: 0 < f1 < f2 < fs/2
    x: 1D numpy array of the signal to filter
    fs: Sampling rate of x in Hz, used to normalize the cutoff.
    f1: Band-pass lower cutoff frequency in Hz, frequencies below this are attenuated.
    f2: Band-pass higher cutoff frequency in Hz, frequencies above this are attenuated.
    order: Filter order. Higher = sharper cutoff but more phase ripple/instability risk.
    '''
    np.nan_to_num(x, copy=False)
    b, a = butter(order, [f1/(fs/2), f2/(fs/2)], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float32)

def _zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    sd = sd if sd > 1e-8 else 1.0
    return ((x - mu) / sd).astype(np.float32)

def _win(x, win, stride=None):
    stride = win if stride is None else stride
    n = (len(x) - win) // stride + 1
    if n <= 0: return np.empty((0, win), np.float32)
    idx = np.arange(win)[None, :] + np.arange(0, n*stride, stride)[:, None]
    return x[idx].astype(np.float32)

def _save_splits(out, tag, ecg_all, ppg_all, train_ratio=0.8, seed=128):
    idx = np.random.RandomState(seed).permutation(len(ecg_all))
    ntrain = int(len(idx) * train_ratio)
    train, test = idx[:ntrain], idx[ntrain:]
    # save npy to storage
    np.save(os.path.join(out, f"ecg_train_{tag}.npy"), ecg_all[train])
    np.save(os.path.join(out, f"ppg_train_{tag}.npy"), ppg_all[train])
    np.save(os.path.join(out, f"ecg_test_{tag}.npy"),  ecg_all[test])
    np.save(os.path.join(out, f"ppg_test_{tag}.npy"),  ppg_all[test])
    print("Saved:", [os.path.join(out, f) for f in os.listdir(out) if f.endswith(f"_{tag}.npy")])

def _convert_freq_and_align(ecg_raw, ppg_raw, 
                    ecg_fs_src=700, ppg_fs_src=64, fs_tgt=128,
                    win_sec=4, stride_sec=None):
    WIN_SIZE  = int(fs_tgt * win_sec)
    STR_SIZE  = None if stride_sec is None else int(fs_tgt * stride_sec)

    # 1) resample to target frequency
    ecg = _resamp(ecg_raw, fs_src=ecg_fs_src, fs_tgt=fs_tgt)
    ppg = _resamp(ppg_raw,  fs_src=ppg_fs_src, fs_tgt=fs_tgt)

    # 2) filters
    ecg = _butter_hp(ecg, fs=fs_tgt, fc=0.5, order=4)
    ppg = _butter_bp(ppg, fs=fs_tgt, f1=0.5, f2=8.0, order=4)

    # 3) per-subject z-score
    ecg = _zscore(ecg)
    ppg = _zscore(ppg)

    # 4) align + window
    min_len = min(len(ecg), len(ppg))
    ecg, ppg = ecg[:min_len], ppg[:min_len]
    e, p = _win(ecg, WIN_SIZE, STR_SIZE), _win(ppg, WIN_SIZE, STR_SIZE)
    n = min(len(e), len(p))

    return (e[:n], p[:n]) if n else None

def build_splits_from_pickles(zip_path, pkl_paths, out, 
                 ecg_raw_fn, ppg_raw_fn,
                 win_sec, stride_sec, 
                 ecg_fs_src, ppg_fs_src, fs_tgt,
                 train_ratio, seed=128):
    # Create dir
    os.makedirs(out, exist_ok=True)
    
    # Gather pkl files, process and delete
    ecg_all, ppg_all = [], []
    for pkl_path in pkl_paths:
        # Extract single file
        with ZipFile(zip_path, "r") as zf:
            with zf.open(pkl_path, "r") as fh:
                # Read pickle
                pkl = pickle.load(fh, encoding="latin1")

                # Process pkl
                ecg_raw = ecg_raw_fn(pkl)
                ppg_raw = ppg_raw_fn(pkl)
                ecg_and_ppg = _convert_freq_and_align(ecg_raw, ppg_raw,
                                                    ecg_fs_src, ppg_fs_src, fs_tgt, win_sec, stride_sec)
                
                if ecg_and_ppg:
                    ecg_all.append(ecg_and_ppg[0])
                    ppg_all.append(ecg_and_ppg[1])

    # Stack, save
    ecg_all, ppg_all = np.vstack(ecg_all), np.vstack(ppg_all)
    _save_splits(out, f'{int(win_sec)}sec',
                 ecg_all, ppg_all, train_ratio, seed)

def build_splits_from_mat(zip_path, mat_paths, out, # ** If zip_path is None, mat_paths will be considered as list of matlab structures, where each struct denotes one case.
                 ecg_raw_fn, ppg_raw_fn,
                 win_sec, stride_sec, 
                 ecg_fs_src_fn, ppg_fs_src_fn, fs_tgt,
                 train_ratio, seed=128):
    # Create dir
    os.makedirs(out, exist_ok=True)
    
    # Gather pkl files, process and delete
    ecg_all, ppg_all = [], []
    for mat_path in mat_paths:
        try:
            if zip_path:
                # Extract single file
                with ZipFile(zip_path, "r") as zf:
                    with zf.open(mat_path, "r") as fh:
                        # Read mat
                        m = mat73.loadmat(fh)
            else:
                # Consider single matlab data
                m = mat_path
            
            # Process mat
            ecg_raw = ecg_raw_fn(m)
            ppg_raw = ppg_raw_fn(m)
            ecg_and_ppg = _convert_freq_and_align(ecg_raw, ppg_raw,
                                                ecg_fs_src_fn(m), ppg_fs_src_fn(m), fs_tgt, win_sec, stride_sec)
            
            if ecg_and_ppg:
                ecg_all.append(ecg_and_ppg[0])
                ppg_all.append(ecg_and_ppg[1])
        except Exception as e:
            print("skip:", os.path.basename(mat_path), e)

    # Stack, save
    ecg_all, ppg_all = np.vstack(ecg_all), np.vstack(ppg_all)
    _save_splits(out, f'{int(win_sec)}sec',
                 ecg_all, ppg_all, train_ratio, seed)

def unzip_single_file(zip_path, extract_path, file_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        if file_path in zip_ref.namelist():
            zip_ref.extract(file_path, path=extract_path)
            print(f'Extracted to: {os.path.join(extract_path, file_path)}')
        else:
            print(f'Error!! Unsuccessful: {file_path}')



if __name__ == "__main__":

    print('WESAD (Expects wesad.zip in current directory): https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download/WESAD.zip')
    print('-'*80)
    # list .pkl paths inside the zip
    wesad_pkl_paths = []
    with ZipFile('wesad.zip', 'r') as zf:
        for fpath in zf.namelist():
            name = fpath.strip()
            if name.startswith('WESAD/S') and name.endswith('.pkl'):
                print(name)
                wesad_pkl_paths.append(name)

    # WESAD Dataset: 4s window
    build_splits_from_pickles(zip_path='wesad.zip', pkl_paths=wesad_pkl_paths, out='../datasets/WESAD/',
                        ecg_raw_fn=lambda x:np.asanyarray(x['signal']['chest']['ECG']).squeeze(), 
                        ppg_raw_fn=lambda x:np.asanyarray(x['signal']['wrist']['BVP']).squeeze(),
                        win_sec=4, stride_sec=None, ecg_fs_src=700, ppg_fs_src=64, fs_tgt=128,
                        train_ratio=0.8, seed=128)

    # WESAD Dataset: 8s window
    build_splits_from_pickles(zip_path='wesad.zip', pkl_paths=wesad_pkl_paths, out='../datasets/WESAD/',
                        ecg_raw_fn=lambda x:np.asanyarray(x['signal']['chest']['ECG']).squeeze(), 
                        ppg_raw_fn=lambda x:np.asanyarray(x['signal']['wrist']['BVP']).squeeze(),
                        win_sec=8, stride_sec=None, ecg_fs_src=700, ppg_fs_src=64, fs_tgt=128,
                        train_ratio=0.8, seed=128)

    print('CAPNO (Expects capno.zip in current directory): https://drive.google.com/uc?id=1HWTAWT1phMQYZlnKJ160FnEPAaRECbk2')
    print('-'*80)
    capno_m_paths = []
    with ZipFile('capno.zip', 'r') as zf:
        for fpath in zf.namelist():
            name = fpath.strip()
            if name.startswith('data/mat/') and name.endswith('_8min.mat'):
                print(name)
                capno_m_paths.append(name)
    
    # CAPNO Dataset
    build_splits_from_mat(zip_path='capno.zip', mat_paths=capno_m_paths, out='../datasets/CAPNO/',
                        ecg_raw_fn=lambda x:np.asarray(x['signal']['ecg']['y']).squeeze().astype(np.float32),
                        ppg_raw_fn=lambda x:np.asarray(x['signal']['pleth']['y']).squeeze().astype(np.float32),
                        win_sec=4, stride_sec=None,
                        ecg_fs_src_fn=lambda x:np.asarray(x['param']['samplingrate']['ecg']).squeeze(),
                        ppg_fs_src_fn=lambda x:np.asarray(x['param']['samplingrate']['pleth']).squeeze(),
                        fs_tgt=128, train_ratio=0.8, seed=128)

    print('DALIA (Expects dalia.zip in current directory): https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4jP/download/PPG_FieldStudy.zip')
    print('-'*80)
    dalia_pkl_paths = []
    with ZipFile('dalia.zip', 'r') as zf:
        for fpath in zf.namelist():
            name = fpath.strip()
            if name.startswith('PPG_FieldStudy/S') and name.endswith('.pkl'):
                print(name)
                dalia_pkl_paths.append(name)

    # DALIA Dataset: 4s window
    build_splits_from_pickles(zip_path='dalia.zip', pkl_paths=dalia_pkl_paths, out='../datasets/DALIA/',
                        ecg_raw_fn=lambda x:np.asanyarray(x['signal']['chest']['ECG']).squeeze(), 
                        ppg_raw_fn=lambda x:np.asanyarray(x['signal']['wrist']['BVP']).squeeze(),
                        win_sec=4, stride_sec=None, ecg_fs_src=700, ppg_fs_src=64, fs_tgt=128,
                        train_ratio=0.8, seed=128)

    # DALIA Dataset: 8s window
    build_splits_from_pickles(zip_path='dalia.zip', pkl_paths=dalia_pkl_paths, out='../datasets/DALIA/',
                        ecg_raw_fn=lambda x:np.asanyarray(x['signal']['chest']['ECG']).squeeze(), 
                        ppg_raw_fn=lambda x:np.asanyarray(x['signal']['wrist']['BVP']).squeeze(),
                        win_sec=8, stride_sec=None, ecg_fs_src=700, ppg_fs_src=64, fs_tgt=128,
                        train_ratio=0.8, seed=128)

    print('BIDMC (Expects bidmc.zip in current directory): https://physionet.org/content/bidmc/get-zip/1.0.0/')
    print('-'*80)
    with ZipFile('bidmc.zip', 'r') as zp:
        zp.extract('bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_data.mat', '../datasets/BIDMC/')
        print('Extracted:', '../datasets/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_data.mat')

    # BIDMC Dataset
    mat_path = "../datasets/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_data.mat"
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    build_splits_from_mat(zip_path=None, mat_paths=m['data'], out='../datasets/BIDMC/',
                        ecg_raw_fn=lambda x:np.asarray(x.ekg.v).squeeze().astype(np.float32),
                        ppg_raw_fn=lambda x:np.asarray(x.ppg.v).squeeze().astype(np.float32),
                        win_sec=4, stride_sec=None,
                        ecg_fs_src_fn=lambda x:float(np.asarray(x.ekg.fs).squeeze()),
                        ppg_fs_src_fn=lambda x:float(np.asarray(x.ppg.fs).squeeze()),
                        fs_tgt=128, train_ratio=0.8, seed=128)

    print('MIMIC (Expects mimic.zip in current directory): https://zenodo.org/record/6807403/files/mimic_perform_af_data.mat?download=1')
    print('-'*80)
    
    # MIMIC Dataset
    mat_path = "/kaggle/working/RDDM/mimic.mat"
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    build_splits_from_mat(zip_path=None, mat_paths=m['data'], out='../datasets/MIMIC-AFib/',
                        ecg_raw_fn=lambda x:np.asarray(x.ekg.v).squeeze().astype(np.float32),
                        ppg_raw_fn=lambda x:np.asarray(x.ppg.v).squeeze().astype(np.float32),
                        win_sec=4, stride_sec=None,
                        ecg_fs_src_fn=lambda x:float(np.asarray(x.ekg.fs).squeeze()),
                        ppg_fs_src_fn=lambda x:float(np.asarray(x.ppg.fs).squeeze()),
                        fs_tgt=128, train_ratio=0.8, seed=128)

