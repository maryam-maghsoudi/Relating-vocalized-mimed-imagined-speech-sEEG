import os
import sys
import glob
import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pdb
import matplotlib.pyplot as plt

SPECT_TOOLS_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/spectrogram_mapping/spectrogram"
COCHBA_MAT_PATH  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/cochba_v7.mat"

AUDIO_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/cmsc828/data/Original_Audio_Sentence"
V_SEEG_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/cmsc828/data/Processed_sEEG_Vocalized_Sentence"
M_SEEG_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/cmsc828/data/Processed_sEEG_Mimed_Sentence"
I_SEEG_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/cmsc828/data/Processed_sEEG_Imagined_Sentence"

OUT_DIR   = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/per_sentence_results"
os.makedirs(OUT_DIR, exist_ok=True)

FRL_MS = 10
TAU_MS = 16

N_FREQ_KEEP = 128     # keep first N_FREQ_KEEP out of 128
N_CH_KEEP   = 110 

# Sliding window parameters
WINDOW_DURATION_MS = 300  # 300ms window duration
HOP_SIZE_MS = 10          # 10ms hop size

if SPECT_TOOLS_PATH not in sys.path:
    sys.path.insert(0, SPECT_TOOLS_PATH)
sys.path.insert(0, "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stimReconstruction")
from spect_tools import cortical_model_python, aud_plot_python  # noqa: E402
from gv_per_sentence import load_cochba, load_wav_mono_norm, parse_audio_fname, corresponding_seeg_path, align_seeg_to_spect_time


def create_sliding_windows(seeg_data, fs_seeg, window_duration_ms, hop_size_ms):
    """
    Create sliding windows over sEEG data.
    
    Parameters:
    -----------
    seeg_data : np.ndarray, shape (T, C)
        sEEG data with T time points and C channels
    fs_seeg : float
        Sampling frequency of sEEG in Hz
    window_duration_ms : float
        Duration of sliding window in milliseconds
    hop_size_ms : float
        Hop size between windows in milliseconds
    
    Returns:
    --------
    windows : np.ndarray, shape (N_windows, window_samples, C)
        Sliding windows of neural data
    window_starts : np.ndarray, shape (N_windows,)
        Start indices of each window in samples
    """
    T, C = seeg_data.shape
    
    # Convert to samples
    window_samples = int(np.round(window_duration_ms / 1000.0 * fs_seeg))
    hop_samples = int(np.round(hop_size_ms / 1000.0 * fs_seeg))
    
    # Calculate number of windows
    n_windows = (T - window_samples) // hop_samples + 1
    
    windows = []
    window_starts = []
    
    for i in range(n_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        
        if end_idx <= T:
            window = seeg_data[start_idx:end_idx, :]  # (window_samples, C)
            windows.append(window)
            window_starts.append(start_idx)
    
    if len(windows) == 0:
        return np.array([]), np.array([])
    
    return np.array(windows), np.array(window_starts)


def train_linear_decoder(seeg_windows, spectrogram_frames, alpha=1.0):
    """
    Train a linear decoder to map sEEG sliding windows to spectrogram frames.
    
    Parameters:
    -----------
    seeg_windows : np.ndarray, shape (N, window_samples, C)
        Sliding windows of sEEG data
    spectrogram_frames : np.ndarray, shape (N, F)
        Corresponding spectrogram frames (frequency bins)
    alpha : float
        Regularization strength for Ridge regression
    
    Returns:
    --------
    decoder : Ridge
        Trained linear decoder
    scaler : StandardScaler
        Scaler fitted on input features
    """
    N, window_samples, C = seeg_windows.shape
    F = spectrogram_frames.shape[1]
    
    # Flatten windows: (N, window_samples * C)
    X = seeg_windows.reshape(N, window_samples * C)
    y = spectrogram_frames  # (N, F)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train separate Ridge regressors for each frequency bin
    # This is equivalent to training one decoder that outputs all frequencies
    decoder = Ridge(alpha=alpha, fit_intercept=True)
    
    # Fit the decoder (Ridge handles multi-output regression)
    decoder.fit(X_scaled, y)
    
    return decoder, scaler


def reconstruct_spectrogram(seeg_windows, decoder, scaler):
    """
    Reconstruct spectrogram from sEEG windows using trained decoder.
    
    Parameters:
    -----------
    seeg_windows : np.ndarray, shape (N, window_samples, C)
        Sliding windows of sEEG data
    decoder : Ridge
        Trained linear decoder
    scaler : StandardScaler
        Scaler fitted on training data
    
    Returns:
    --------
    reconstructed : np.ndarray, shape (N, F)
        Reconstructed spectrogram frames
    """
    N, window_samples, C = seeg_windows.shape
    
    # Flatten windows
    X = seeg_windows.reshape(N, window_samples * C)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict spectrogram
    reconstructed = decoder.predict(X_scaled)
    
    return reconstructed


def main():  
    cochba = load_cochba(COCHBA_MAT_PATH)
    print(f"[INFO] cochba shape: {cochba.shape}")

    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")

    fs_seeg = 200
    fs_spect = 1 / FRL_MS * 1000.0  # 100 Hz (10ms per frame)

    for i, wav_path in enumerate(wav_files[:5], 1):
        sentence, rep = parse_audio_fname(wav_path)
        modality = "Vocalized"
        v_seeg_path = corresponding_seeg_path(sentence, rep, modality)
        if not os.path.exists(v_seeg_path):
            print(f"[WARN] Missing sEEG for {sentence} rep{rep}: {v_seeg_path} (skipping)")
            continue

        print(f"[{i}/{len(wav_files)}] {sentence} rep{rep}")

        # ---- 1) Load audio + compute stimulus spectrogram ----
        wav, fs_audio = load_wav_mono_norm(wav_path)
        fs_old = fs_audio
        fs_new = 8000
        wav = resample_poly(wav, up=fs_new, down=fs_old)

        v_spect = cortical_model_python(wav, fs_new, FRL_MS, TAU_MS,
            pplot=False, which="wav", cochba=cochba
        )
        v_spect = np.sqrt(v_spect)   # power to amplitude
        v_spect = v_spect[:, :N_FREQ_KEEP]  # (T_spec, F)

        # ---- 2) Load sEEG (vocalized), keep first 110 channels ----
        v_X = np.load(v_seeg_path)  # (T_seeg, C_all) = (1000, 220)
        v_X = v_X[:, -N_CH_KEEP:]   # (T_seeg, C) = (1000, 110)

        # ---- 3) Align sEEG to spectrogram time ----
        # Resample sEEG to match spectrogram time resolution
        v_X_aligned = align_seeg_to_spect_time(v_X, T_spec=v_spect.shape[0])  # => (T_spec, C)
        
        print(f"[INFO] Spectrogram shape: {v_spect.shape}")
        print(f"[INFO] sEEG aligned shape: {v_X_aligned.shape}")

        # ---- 4) Create sliding windows from aligned sEEG ----
        # Create windows at spectrogram time resolution
        # Window duration: 300ms, Hop size: 10ms
        # At spectrogram resolution (100 Hz = 10ms per frame):
        #   - 300ms window = 30 frames
        #   - 10ms hop = 1 frame
        
        window_frames = int(np.round(WINDOW_DURATION_MS / 1000.0 * fs_spect))  # 30 frames
        hop_frames = int(np.round(HOP_SIZE_MS / 1000.0 * fs_spect))  # 1 frame
        
        T_spec, C = v_X_aligned.shape
        F = v_spect.shape[1]
        
        # Create sliding windows from aligned sEEG
        windows = []
        spect_frame_indices = []
        
        for t_start in range(0, T_spec - window_frames + 1, hop_frames):
            t_end = t_start + window_frames
            window = v_X_aligned[t_start:t_end, :]  # (window_frames, C)
            windows.append(window)
            # Each window predicts the spectrogram frame at the end of the window
            # This accounts for neural processing delay: neural activity in the window
            # (spanning 300ms) predicts the stimulus at the end of that window
            spect_frame_indices.append(t_end - 1)
        
        if len(windows) == 0:
            print(f"[WARN] No windows created for {sentence} rep{rep} (skipping)")
            continue
        
        windows = np.array(windows)  # (N_windows, window_frames, C)
        spect_frame_indices = np.array(spect_frame_indices)
        
        # Ensure indices don't exceed spectrogram length
        valid_mask = spect_frame_indices < T_spec
        windows = windows[valid_mask]
        spect_frame_indices = spect_frame_indices[valid_mask]
        
        # Get corresponding spectrogram frames
        spect_frames = v_spect[spect_frame_indices, :]  # (N_windows, F)
        
        print(f"[INFO] Created {len(windows)} sliding windows")
        print(f"[INFO] Window shape: {windows.shape} (each window: {window_frames} frames x {C} channels)")
        print(f"[INFO] Spectrogram frames shape: {spect_frames.shape}")

        # ---- 5) Train linear decoder ----
        print("[INFO] Training linear decoder...")
        decoder, scaler = train_linear_decoder(
            windows,
            spect_frames,
            alpha=1.0
        )
        
        # ---- 6) Reconstruct spectrogram ----
        print("[INFO] Reconstructing spectrogram...")
        reconstructed = reconstruct_spectrogram(
            windows,
            decoder,
            scaler
        )
        if True:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
            im0 = axs[0].imshow(spect_frames.T, aspect='auto', origin='lower', interpolation='none')
            axs[0].set_title('True Spectrogram Frames')
            axs[0].set_ylabel('Frequency bin')
            fig.colorbar(im0, ax=axs[0], orientation='vertical', pad=0.02)
            im1 = axs[1].imshow(reconstructed.T, aspect='auto', origin='lower', interpolation='none')
            axs[1].set_title('Reconstructed Spectrogram Frames')
            axs[1].set_xlabel('Sliding window frame')
            axs[1].set_ylabel('Frequency bin')
            fig.colorbar(im1, ax=axs[1], orientation='vertical', pad=0.02)
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{sentence}_rep{rep}_true_vs_rcnst_spect.png"))
        
        # ---- 8) Evaluate reconstruction ----
        mse = mean_squared_error(spect_frames, reconstructed)
        print(f"[INFO] Reconstruction MSE: {mse:.6f}")
        
        # Compute correlation per frequency bin
        correlations = []
        for f in range(spect_frames.shape[1]):
            corr = np.corrcoef(spect_frames[:, f], reconstructed[:, f])[0, 1]
            correlations.append(corr)
        mean_corr = np.mean(correlations)
        print(f"[INFO] Mean correlation per frequency bin: {mean_corr:.4f}")
        
        # Save results
        out_file = os.path.join(OUT_DIR, f"{sentence}_rep{rep}_sliding_window.npz")
        np.savez(
            out_file,
            original_spectrogram=spect_frames,
            reconstructed_spectrogram=reconstructed,
            mse=mse,
            correlations=correlations,
            mean_correlation=mean_corr,
            window_duration_ms=WINDOW_DURATION_MS,
            hop_size_ms=HOP_SIZE_MS
        )
        print(f"[INFO] Saved results to {out_file}")


if __name__ == "__main__":
    main()