import os
import sys
import glob
import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pdb

# Import from the per-sentence file
sys.path.insert(0, "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2")
from stimrcnt_per_sentence import (
    SPECT_TOOLS_PATH, COCHBA_MAT_PATH, AUDIO_DIR, V_SEEG_DIR,
    FRL_MS, TAU_MS, N_FREQ_KEEP, N_CH_KEEP,
    WINDOW_DURATION_MS, HOP_SIZE_MS,
    load_cochba, load_wav_mono_norm, parse_audio_fname, 
    corresponding_seeg_path, align_seeg_to_spect_time,
    train_linear_decoder, reconstruct_spectrogram
)

OUT_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Ridge regression regularization parameter
# Higher values = more regularization (prevents overfitting)
# Lower values = less regularization (may overfit)
# Set to None to use grid search to find optimal value
RIDGE_ALPHA = 100.0  # Default value, adjust as needed
# RIDGE_ALPHA = None  # Uncomment to use grid search
N_PERM = 20
# Alpha values to test in grid search (if RIDGE_ALPHA is None)
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

if SPECT_TOOLS_PATH not in sys.path:
    sys.path.insert(0, SPECT_TOOLS_PATH)

from spect_tools import cortical_model_python  # noqa: E402


def create_sliding_windows_from_aligned(seeg_aligned, spectrogram, fs_spect):
    """
    Create sliding windows from aligned sEEG data and corresponding spectrogram frames.
    
    Parameters:
    -----------
    seeg_aligned : np.ndarray, shape (T_spec, C)
        sEEG data aligned to spectrogram time resolution
    spectrogram : np.ndarray, shape (T_spec, F)
        Spectrogram data
    fs_spect : float
        Spectrogram frame rate in Hz
    
    Returns:
    --------
    windows : np.ndarray, shape (N_windows, window_frames, C)
        Sliding windows of sEEG data
    spect_frames : np.ndarray, shape (N_windows, F)
        Corresponding spectrogram frames
    """
    window_frames = int(np.round(WINDOW_DURATION_MS / 1000.0 * fs_spect))  # 30 frames
    hop_frames = int(np.round(HOP_SIZE_MS / 1000.0 * fs_spect))  # 1 frame
    
    T_spec, C = seeg_aligned.shape
    F = spectrogram.shape[1]
    
    windows = []
    spect_frame_indices = []
    
    for t_start in range(0, T_spec - window_frames + 1, hop_frames):
        t_end = t_start + window_frames
        window = seeg_aligned[t_start:t_end, :]  # (window_frames, C)
        windows.append(window)
        # Each window predicts the spectrogram frame at the end of the window
        spect_frame_indices.append(t_end - 1)
    
    if len(windows) == 0:
        return np.array([]), np.array([])
    
    windows = np.array(windows)  # (N_windows, window_frames, C)
    spect_frame_indices = np.array(spect_frame_indices)
    
    # Ensure indices don't exceed spectrogram length
    valid_mask = spect_frame_indices < T_spec
    windows = windows[valid_mask]
    spect_frame_indices = spect_frame_indices[valid_mask]
    
    # Get corresponding spectrogram frames
    spect_frames = spectrogram[spect_frame_indices, :]  # (N_windows, F)
    
    return windows, spect_frames


def normalize_spectrograms(y_train, y_test=None):
    """
    Normalize spectrogram targets using StandardScaler.
    Fits on training data only, applies to both train and test.
    
    Parameters:
    -----------
    y_train : np.ndarray, shape (N_train, F)
        Training spectrogram frames
    y_test : np.ndarray, shape (N_test, F) or None
        Test spectrogram frames (optional)
    
    Returns:
    --------
    y_train_norm : np.ndarray, shape (N_train, F)
        Normalized training spectrogram frames
    y_test_norm : np.ndarray, shape (N_test, F) or None
        Normalized test spectrogram frames (if provided)
    spect_scaler : StandardScaler
        Fitted scaler for denormalization
    """
    spect_scaler = StandardScaler()
    y_train_norm = spect_scaler.fit_transform(y_train)
    
    if y_test is not None:
        y_test_norm = spect_scaler.transform(y_test)
        return y_train_norm, y_test_norm, spect_scaler
    else:
        return y_train_norm, spect_scaler


def denormalize_spectrograms(y_norm, spect_scaler):
    """
    Denormalize spectrogram predictions back to original scale.
    
    Parameters:
    -----------
    y_norm : np.ndarray, shape (N, F)
        Normalized spectrogram frames
    spect_scaler : StandardScaler
        Fitted scaler used for normalization
    
    Returns:
    --------
    y : np.ndarray, shape (N, F)
        Denormalized spectrogram frames
    """
    return spect_scaler.inverse_transform(y_norm)


def compute_spectrogram_envelope(spectrogram):
    """
    Compute the envelope of a spectrogram by averaging along the frequency axis.
    
    Parameters:
    -----------
    spectrogram : np.ndarray, shape (N, F)
        Spectrogram frames where N is number of windows and F is number of frequency bins
    
    Returns:
    --------
    envelope : np.ndarray, shape (N,)
        Envelope (average across frequency bins) for each window
    """
    return np.mean(spectrogram, axis=1)  # Average along frequency axis


def compute_chance_performance(y_true, y_pred, n_permutations=N_PERM, random_state=42):
    """
    Compute chance-level performance by shuffling predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray, shape (N, F)
        True spectrogram frames
    y_pred : np.ndarray, shape (N, F)
        Predicted spectrogram frames
    n_permutations : int
        Number of permutations for null distribution
    random_state : int
        Random seed
    
    Returns:
    --------
    chance_correlations : np.ndarray
        Correlation values from shuffled predictions
    chance_mean : float
        Mean correlation from shuffled predictions
    chance_std : float
        Std of correlations from shuffled predictions
    p_value : float
        P-value (proportion of shuffled correlations >= actual correlation)
    """
    np.random.seed(random_state)
    
    actual_corrs = []
    for f in range(y_true.shape[1]):
        corr = np.corrcoef(y_true[:, f], y_pred[:, f])[0, 1]
        if not np.isnan(corr):
            actual_corrs.append(corr)
    actual_mean_corr = np.mean(actual_corrs)
    
    # Shuffle predictions
    chance_corrs = []
    for _ in range(n_permutations):
        y_pred_shuffled = y_pred.copy()
        np.random.shuffle(y_pred_shuffled)
        
        perm_corrs = []
        for f in range(y_true.shape[1]):
            corr = np.corrcoef(y_true[:, f], y_pred_shuffled[:, f])[0, 1]
            if not np.isnan(corr):
                perm_corrs.append(corr)
        if len(perm_corrs) > 0:
            chance_corrs.append(np.mean(perm_corrs))
    
    chance_corrs = np.array(chance_corrs)
    chance_mean = np.mean(chance_corrs)
    chance_std = np.std(chance_corrs)
    
    # P-value: proportion of shuffled correlations >= actual
    p_value = np.mean(chance_corrs >= actual_mean_corr)
    
    return chance_corrs, chance_mean, chance_std, p_value


def evaluate_cross_modality_performance(train_results, test_results, test_modality_name="Test", n_permutations=N_PERM):
    """
    Comprehensive evaluation of cross-modality performance.
    
    Parameters:
    -----------
    train_results : dict
        Dictionary with 'y_true', 'y_pred', 'envelope_true', 'envelope_pred'
    test_results : dict
        Dictionary with 'y_true', 'y_pred', 'envelope_true', 'envelope_pred'
    test_modality_name : str
        Name of test modality (e.g., "Mimed", "Imagined")
    n_permutations : int
        Number of permutations for chance-level testing
    
    Returns:
    --------
    eval_dict : dict
        Comprehensive evaluation metrics
    """
    print("\n" + "="*60)
    print(f"COMPREHENSIVE EVALUATION: {test_modality_name}")
    print("="*60)
    
    eval_dict = {}
    
    # 1. Full spectrogram performance
    print(f"\n[1] Full Spectrogram Performance:")
    print(f"    Training correlation: {train_results['mean_correlation']:.4f}")
    print(f"    Test correlation: {test_results['mean_correlation']:.4f}")
    print(f"    Performance drop: {train_results['mean_correlation'] - test_results['mean_correlation']:.4f}")
    print(f"    Relative performance: {test_results['mean_correlation'] / train_results['mean_correlation']:.2%} of training")
    
    eval_dict['spectrogram'] = {
        'train_corr': train_results['mean_correlation'],
        'test_corr': test_results['mean_correlation'],
        'drop': train_results['mean_correlation'] - test_results['mean_correlation'],
        'relative_performance': test_results['mean_correlation'] / train_results['mean_correlation']
    }
    
    # 2. Envelope performance
    print(f"\n[2] Envelope Performance:")
    print(f"    Training envelope correlation: {train_results['envelope_correlation']:.4f}")
    print(f"    Test envelope correlation: {test_results['envelope_correlation']:.4f}")
    print(f"    Performance drop: {train_results['envelope_correlation'] - test_results['envelope_correlation']:.4f}")
    print(f"    Relative performance: {test_results['envelope_correlation'] / train_results['envelope_correlation']:.2%} of training")
    
    eval_dict['envelope'] = {
        'train_corr': train_results['envelope_correlation'],
        'test_corr': test_results['envelope_correlation'],
        'drop': train_results['envelope_correlation'] - test_results['envelope_correlation'],
        'relative_performance': test_results['envelope_correlation'] / train_results['envelope_correlation']
    }
    
    # 3. Chance-level performance (statistical significance)
    # print(f"\n[3] Statistical Significance (Chance-Level Test):")
    # print(f"    Computing chance-level performance with {n_permutations} permutations...")
    # 
    # chance_corrs_spec, chance_mean_spec, chance_std_spec, p_value_spec = compute_chance_performance(
    #     test_results['y_true'], test_results['y_pred'], n_permutations=N_PERM
    # )
    # 
    # chance_corrs_env, chance_mean_env, chance_std_env, p_value_env = compute_chance_performance(
    #     test_results['envelope_true'].reshape(-1, 1), 
    #     test_results['envelope_pred'].reshape(-1, 1), 
    #     n_permutations=N_PERM
    # )
    # 
    # print(f"    Spectrogram:")
    # print(f"      Actual correlation: {test_results['mean_correlation']:.4f}")
    # print(f"      Chance mean: {chance_mean_spec:.4f} ± {chance_std_spec:.4f}")
    # print(f"      Z-score: {(test_results['mean_correlation'] - chance_mean_spec) / chance_std_spec:.2f}")
    # print(f"      P-value: {p_value_spec:.4f} {'***' if p_value_spec < 0.001 else '**' if p_value_spec < 0.01 else '*' if p_value_spec < 0.05 else 'ns'}")
    # 
    # print(f"    Envelope:")
    # print(f"      Actual correlation: {test_results['envelope_correlation']:.4f}")
    # print(f"      Chance mean: {chance_mean_env:.4f} ± {chance_std_env:.4f}")
    # print(f"      Z-score: {(test_results['envelope_correlation'] - chance_mean_env) / chance_std_env:.2f}")
    # print(f"      P-value: {p_value_env:.4f} {'***' if p_value_env < 0.001 else '**' if p_value_env < 0.01 else '*' if p_value_env < 0.05 else 'ns'}")
    # 
    # eval_dict['significance'] = {
    #     'spectrogram': {
    #         'chance_mean': chance_mean_spec,
    #         'chance_std': chance_std_spec,
    #         'z_score': (test_results['mean_correlation'] - chance_mean_spec) / chance_std_spec,
    #         'p_value': p_value_spec
    #     },
    #     'envelope': {
    #         'chance_mean': chance_mean_env,
    #         'chance_std': chance_std_env,
    #         'z_score': (test_results['envelope_correlation'] - chance_mean_env) / chance_std_env,
    #         'p_value': p_value_env
    #     }
    # }
    
    # 4. Performance benchmarks
    print(f"\n[4] Performance Benchmarks:")
    
    # Correlation thresholds (commonly used in neuroscience)
    corr_excellent = 0.7
    corr_good = 0.5
    corr_moderate = 0.3
    corr_weak = 0.2
    
    spec_corr = test_results['mean_correlation']
    env_corr = test_results['envelope_correlation']
    
    def get_rating(corr):
        if corr >= corr_excellent:
            return "Excellent"
        elif corr >= corr_good:
            return "Good"
        elif corr >= corr_moderate:
            return "Moderate"
        elif corr >= corr_weak:
            return "Weak"
        else:
            return "Poor"
    
    print(f"    Spectrogram correlation: {spec_corr:.4f} ({get_rating(spec_corr)})")
    print(f"    Envelope correlation: {env_corr:.4f} ({get_rating(env_corr)})")
    
    eval_dict['benchmarks'] = {
        'spectrogram_rating': get_rating(spec_corr),
        'envelope_rating': get_rating(env_corr)
    }
    
    # 5. Cross-modality comparison
    print(f"\n[5] Cross-Modality Analysis:")
    if spec_corr > 0.5:
        print(f"    ✓ Spectrogram: Above 'good' threshold (r > 0.5)")
        print(f"      → Model generalizes well to {test_modality_name} for full spectrogram")
    elif spec_corr > 0.3:
        print(f"    ⚠ Spectrogram: Moderate performance (0.3 < r < 0.5)")
        print(f"      → Model shows some generalization to {test_modality_name}")
    else:
        print(f"    ✗ Spectrogram: Weak performance (r < 0.3)")
        print(f"      → Limited generalization to {test_modality_name}")
    
    if env_corr > 0.5:
        print(f"    ✓ Envelope: Above 'good' threshold (r > 0.5)")
        print(f"      → Temporal dynamics well captured for {test_modality_name}")
    elif env_corr > 0.3:
        print(f"    ⚠ Envelope: Moderate performance (0.3 < r < 0.5)")
        print(f"      → Some temporal structure captured for {test_modality_name}")
    else:
        print(f"    ✗ Envelope: Weak performance (r < 0.3)")
        print(f"      → Limited temporal structure capture for {test_modality_name}")
    
    # 6. Practical interpretation
    print(f"\n[6] Practical Interpretation:")
    # if p_value_spec < 0.05 and spec_corr > 0.3:
    if spec_corr > 0.5:
        print(f"    ✓ Predictions are practically meaningful")
        print(f"      → The model can reliably decode {test_modality_name} spectrograms")
        print(f"      → Cross-modality transfer is successful")
    elif spec_corr > 0.3:
        print(f"    ⚠ Predictions show moderate performance")
        print(f"      → Model shows signal, but may need improvement")
    else:
        print(f"    ✗ Predictions show weak performance")
        print(f"      → Model does not reliably decode {test_modality_name}")
    
    return eval_dict


def find_best_alpha(X_train, y_train_norm, X_val, y_val_norm, spect_scaler, alpha_grid):
    """
    Find the best Ridge alpha value using validation set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training windows
    y_train_norm : np.ndarray
        Normalized training spectrograms
    X_val : np.ndarray
        Validation windows
    y_val_norm : np.ndarray
        Normalized validation spectrograms
    spect_scaler : StandardScaler
        Scaler for denormalization
    alpha_grid : list
        List of alpha values to test
    
    Returns:
    --------
    best_alpha : float
        Best alpha value
    best_score : float
        Best validation correlation score
    results : dict
        Dictionary with alpha values and their scores
    """
    print("\n" + "="*60)
    print("GRID SEARCH FOR OPTIMAL RIDGE ALPHA")
    print("="*60)
    
    results = {}
    best_alpha = None
    best_score = -np.inf
    
    # Get original scale validation data for evaluation
    y_val = denormalize_spectrograms(y_val_norm, spect_scaler)
    
    for alpha in alpha_grid:
        print(f"\n[INFO] Testing alpha = {alpha:.2f}")
        
        # Train decoder
        decoder, scaler = train_linear_decoder(X_train, y_train_norm, alpha=alpha)
        
        # Evaluate on validation set
        y_val_pred_norm = reconstruct_spectrogram(X_val, decoder, scaler)
        y_val_pred = denormalize_spectrograms(y_val_pred_norm, spect_scaler)
        
        # Compute correlation (use correlation as metric)
        correlations = []
        for f in range(y_val.shape[1]):
            corr = np.corrcoef(y_val[:, f], y_val_pred[:, f])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        mean_corr = np.mean(correlations) if correlations else 0.0
        mse = mean_squared_error(y_val, y_val_pred)
        
        results[alpha] = {'correlation': mean_corr, 'mse': mse}
        
        print(f"  Validation correlation: {mean_corr:.4f}, MSE: {mse:.6f}")
        
        if mean_corr > best_score:
            best_score = mean_corr
            best_alpha = alpha
    
    print(f"\n[INFO] Best alpha: {best_alpha:.2f} (correlation: {best_score:.4f})")
    
    return best_alpha, best_score, results


def collect_all_data(wav_files, modality="Vocalized", max_files=None):
    """
    Collect all windows and spectrogram frames from multiple files.
    
    Parameters:
    -----------
    wav_files : list
        List of audio file paths
    modality : str
        Modality to use ("Vocalized", "Mimed", "Imagined")
    max_files : int or None
        Maximum number of files to process (None for all)
    
    Returns:
    --------
    all_windows : np.ndarray, shape (N_total, window_frames, C)
        All collected windows
    all_spect_frames : np.ndarray, shape (N_total, F)
        All corresponding spectrogram frames
    file_info : list
        List of (sentence, rep) tuples for each window
    """
    cochba = load_cochba(COCHBA_MAT_PATH)
    fs_seeg = 200
    fs_spect = 1 / FRL_MS * 1000.0  # 100 Hz
    
    all_windows = []
    all_spect_frames = []
    file_info = []
    
    if max_files is not None:
        wav_files = wav_files[:max_files]
    
    for i, wav_path in enumerate(wav_files, 1):
        sentence, rep = parse_audio_fname(wav_path)
        v_seeg_path = corresponding_seeg_path(sentence, rep, modality)
        
        if not os.path.exists(v_seeg_path):
            print(f"[WARN] Missing sEEG for {sentence} rep{rep}: {v_seeg_path} (skipping)")
            continue
        
        print(f"[{i}/{len(wav_files)}] Collecting data from {sentence} rep{rep}")
        
        # Load audio and compute spectrogram
        wav, fs_audio = load_wav_mono_norm(wav_path)
        fs_old = fs_audio
        fs_new = 8000
        wav = resample_poly(wav, up=fs_new, down=fs_old)
        
        v_spect = cortical_model_python(wav, fs_new, FRL_MS, TAU_MS,
            pplot=False, which="wav", cochba=cochba
        )
        v_spect = np.sqrt(v_spect)  # power to amplitude
        v_spect = v_spect[:, :N_FREQ_KEEP]  # (T_spec, F)
        
        # Load sEEG
        v_X = np.load(v_seeg_path)  # (T_seeg, C_all)
        v_X = v_X[:, -N_CH_KEEP:]   # (T_seeg, C)
        
        # Align sEEG to spectrogram time
        v_X_aligned = align_seeg_to_spect_time(v_X, T_spec=v_spect.shape[0])
        
        # Create sliding windows
        windows, spect_frames = create_sliding_windows_from_aligned(
            v_X_aligned, v_spect, fs_spect
        )
        
        if len(windows) == 0:
            print(f"[WARN] No windows created for {sentence} rep{rep} (skipping)")
            continue
        
        # Append to collections
        all_windows.append(windows)
        all_spect_frames.append(spect_frames)
        
        # Track which file each window comes from
        n_windows = len(windows)
        file_info.extend([(sentence, rep)] * n_windows)
        
        print(f"  -> Collected {n_windows} windows")
    
    if len(all_windows) == 0:
        return np.array([]), np.array([]), []
    
    # Concatenate all windows
    all_windows = np.concatenate(all_windows, axis=0)
    all_spect_frames = np.concatenate(all_spect_frames, axis=0)
    
    print(f"\n[INFO] Total collected: {len(all_windows)} windows")
    print(f"[INFO] Window shape: {all_windows.shape}")
    print(f"[INFO] Spectrogram frames shape: {all_spect_frames.shape}")
    
    return all_windows, all_spect_frames, file_info


def split_by_sentence(file_info, test_size=0.2, random_state=42):
    """
    Split data by sentence (leave some sentences out for testing).
    
    Parameters:
    -----------
    file_info : list
        List of (sentence, rep) tuples
    test_size : float
        Proportion of unique sentences to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_mask : np.ndarray
        Boolean mask for training samples
    test_mask : np.ndarray
        Boolean mask for test samples
    train_sentences : set
        Set of sentences in training set
    test_sentences : set
        Set of sentences in test set
    """
    # Get unique sentences
    unique_sentences = sorted(set([s for s, _ in file_info]))
    
    # Split sentences
    train_sentences, test_sentences = train_test_split(
        unique_sentences, test_size=test_size, random_state=random_state
    )
    
    train_sentences = set(train_sentences)
    test_sentences = set(test_sentences)
    
    # Create masks
    train_mask = np.array([s in train_sentences for s, _ in file_info])
    test_mask = np.array([s in test_sentences for s, _ in file_info])
    
    print(f"\n[INFO] Split by sentence:")
    print(f"  Training sentences ({len(train_sentences)}): {sorted(train_sentences)}")
    print(f"  Test sentences ({len(test_sentences)}): {sorted(test_sentences)}")
    print(f"  Training samples: {train_mask.sum()}")
    print(f"  Test samples: {test_mask.sum()}")
    
    return train_mask, test_mask, train_sentences, test_sentences


def split_random(file_info, test_size=0.2, random_state=42):
    """
    Split data randomly (mixing all sentences).
    
    Parameters:
    -----------
    file_info : list
        List of (sentence, rep) tuples
    test_size : float
        Proportion of samples to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_mask : np.ndarray
        Boolean mask for training samples
    test_mask : np.ndarray
        Boolean mask for test samples
    """
    n_samples = len(file_info)
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    print(f"\n[INFO] Random split:")
    print(f"  Training samples: {train_mask.sum()}")
    print(f"  Test samples: {test_mask.sum()}")
    
    return train_mask, test_mask

def split_file_info(file_info):
    sents = [s for s, r in file_info]
    reps  = [r for s, r in file_info]
    return np.array(sents, dtype=object), np.array(reps, dtype=int)

def main():
    # Collect all data
    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")
    
    print(f"[INFO] Found {len(wav_files)} audio files")
    
    # Collect all windows and spectrogram frames
    all_windows, all_spect_frames, file_info = collect_all_data(
        wav_files, modality="Vocalized", max_files=None
    )
    
    if len(all_windows) == 0:
        raise ValueError("No data collected!")
    
    # Split into train/test sets
    # Option 1: Split by sentence (leave some sentences out)
    print("\n" + "="*60)
    print("SPLITTING DATA BY SENTENCE (generalization across sentences)")
    print("="*60)
    train_mask, test_mask, train_sentences, test_sentences = split_by_sentence(
        file_info, test_size=0.2, random_state=42
    )
    
    # Option 2: Random split (uncomment to use instead)
    # print("\n" + "="*60)
    # print("RANDOM SPLIT")
    # print("="*60)
    # train_mask, test_mask = split_random(file_info, test_size=0.2, random_state=42)
    
    # Get train/test sets
    X_train = all_windows[train_mask]
    y_train = all_spect_frames[train_mask]
    X_test = all_windows[test_mask]
    y_test = all_spect_frames[test_mask]
    
    print(f"\n[INFO] Training set: {X_train.shape[0]} windows")
    print(f"[INFO] Test set: {X_test.shape[0]} windows")
    
    # Normalize spectrograms (fit on training data only)
    print("\n" + "="*60)
    print("NORMALIZING SPECTROGRAMS")
    print("="*60)
    y_train_norm, y_test_norm, spect_scaler = normalize_spectrograms(y_train, y_test)
    print(f"[INFO] Normalized training spectrograms: mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")
    print(f"[INFO] Normalized test spectrograms: mean={y_test_norm.mean():.4f}, std={y_test_norm.std():.4f}")
    
    # Determine alpha value
    alpha_results = None
    best_alpha = None
    best_score = None
    
    if RIDGE_ALPHA is None:
        # Use grid search to find best alpha
        # Split training data into train/validation for grid search
        X_train_gs, X_val_gs, y_train_gs_norm, y_val_gs_norm = train_test_split(
            X_train, y_train_norm, test_size=0.2, random_state=42
        )
        best_alpha, best_score, alpha_results = find_best_alpha(
            X_train_gs, y_train_gs_norm, X_val_gs, y_val_gs_norm, spect_scaler, ALPHA_GRID
        )
        ridge_alpha = best_alpha
        print(f"\n[INFO] Using optimal alpha from grid search: {ridge_alpha:.2f}")
    else:
        ridge_alpha = RIDGE_ALPHA
        print(f"\n[INFO] Using specified alpha: {ridge_alpha:.2f}")
    
    # Train decoder on training set (using normalized targets)
    print("\n" + "="*60)
    print(f"TRAINING LINEAR DECODER (alpha={ridge_alpha:.2f})")
    print("="*60)
    decoder, scaler = train_linear_decoder(
        X_train, y_train_norm, alpha=ridge_alpha
    )
    
    # Evaluate on training set
    print("\n" + "="*60)
    print("EVALUATING ON TRAINING SET")
    print("="*60)
    y_train_pred_norm = reconstruct_spectrogram(X_train, decoder, scaler)
    # Denormalize predictions for evaluation
    y_train_pred = denormalize_spectrograms(y_train_pred_norm, spect_scaler)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_correlations = []
    for f in range(y_train.shape[1]):
        corr = np.corrcoef(y_train[:, f], y_train_pred[:, f])[0, 1]
        train_correlations.append(corr)
    train_mean_corr = np.mean(train_correlations)
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Training mean correlation: {train_mean_corr:.4f}")
    
    # Evaluate on test set (generalization)
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET (GENERALIZATION)")
    print("="*60)
    y_test_pred_norm = reconstruct_spectrogram(X_test, decoder, scaler)
    # Denormalize predictions for evaluation
    y_test_pred = denormalize_spectrograms(y_test_pred_norm, spect_scaler)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_correlations = []
    for f in range(y_test.shape[1]):
        corr = np.corrcoef(y_test[:, f], y_test_pred[:, f])[0, 1]
        test_correlations.append(corr)
    test_mean_corr = np.mean(test_correlations)
    
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test mean correlation: {test_mean_corr:.4f}")
    
    # Plot comparison
    print("\n[INFO] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Training set: true vs reconstructed
    im0 = axes[0, 0].imshow(y_train[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[0, 0].set_title(f'Training Set: True Spectrogram (first 200 windows)')
    axes[0, 0].set_ylabel('Frequency bin')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(y_train_pred[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[0, 1].set_title(f'Training Set: Reconstructed (MSE={train_mse:.4f}, Corr={train_mean_corr:.3f})')
    axes[0, 1].set_ylabel('Frequency bin')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Test set: true vs reconstructed
    im2 = axes[1, 0].imshow(y_test[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[1, 0].set_title(f'Test Set: True Spectrogram (first 200 windows)')
    axes[1, 0].set_xlabel('Window index')
    axes[1, 0].set_ylabel('Frequency bin')
    plt.colorbar(im2, ax=axes[1, 0])
    
    im3 = axes[1, 1].imshow(y_test_pred[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[1, 1].set_title(f'Test Set: Reconstructed (MSE={test_mse:.4f}, Corr={test_mean_corr:.3f})')
    axes[1, 1].set_xlabel('Window index')
    axes[1, 1].set_ylabel('Frequency bin')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "generalization_train_test_comparison.png"), dpi=150)
    print(f"[INFO] Saved visualization to {OUT_DIR}/generalization_train_test_comparison.png")
    
    # Compute and plot spectrogram envelopes
    print("\n[INFO] Computing spectrogram envelopes...")
    train_envelope_true = compute_spectrogram_envelope(y_train)
    train_envelope_pred = compute_spectrogram_envelope(y_train_pred)
    test_envelope_true = compute_spectrogram_envelope(y_test)
    test_envelope_pred = compute_spectrogram_envelope(y_test_pred)
    # Compute envelope correlations
    train_envelope_corr = np.corrcoef(train_envelope_true, train_envelope_pred)[0, 1]
    test_envelope_corr = np.corrcoef(test_envelope_true, test_envelope_pred)[0, 1]
    train_envelope_mse = mean_squared_error(train_envelope_true, train_envelope_pred)
    test_envelope_mse = mean_squared_error(test_envelope_true, test_envelope_pred)
    
    print(f"Training envelope correlation: {train_envelope_corr:.4f}, MSE: {train_envelope_mse:.6f}")
    print(f"Test envelope correlation: {test_envelope_corr:.4f}, MSE: {test_envelope_mse:.6f}")
    
    # Plot envelopes
    if True:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        # Training set envelope
        axes[0].plot(train_envelope_true[:500], label='True', linewidth=2, alpha=0.8)
        axes[0].plot(train_envelope_pred[:500], label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        axes[0].set_title(f'Training Set: Spectrogram Envelope (first 500 windows)\nCorr={train_envelope_corr:.3f}, MSE={train_envelope_mse:.4f}')
        axes[0].set_ylabel('Average Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # Test set envelope
        axes[1].plot(test_envelope_true[:500], label='True', linewidth=2, alpha=0.8)
        axes[1].plot(test_envelope_pred[:500], label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        axes[1].set_title(f'Test Set: Spectrogram Envelope (first 500 windows)\nCorr={test_envelope_corr:.3f}, MSE={test_envelope_mse:.4f}')
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('Average Amplitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "gen_envelopes.png"), dpi=150)
        print(f"[INFO] Saved envelope visualization to {OUT_DIR}/generalization_envelopes.png")
    
    # Save results
    out_file = os.path.join(OUT_DIR, "generalization_results.npz")
    save_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_correlations': train_correlations,
        'test_correlations': test_correlations,
        'train_mean_correlation': train_mean_corr,
        'test_mean_correlation': test_mean_corr,
        'train_sentences': list(train_sentences),
        'test_sentences': list(test_sentences),
        'window_duration_ms': WINDOW_DURATION_MS,
        'hop_size_ms': HOP_SIZE_MS,
        'ridge_alpha': ridge_alpha,
        'train_envelope_true': train_envelope_true,
        'train_envelope_pred': train_envelope_pred,
        'test_envelope_true': test_envelope_true,
        'test_envelope_pred': test_envelope_pred,
        'train_envelope_correlation': train_envelope_corr,
        'test_envelope_correlation': test_envelope_corr,
        'train_envelope_mse': train_envelope_mse,
        'test_envelope_mse': test_envelope_mse
    }
    
    # Add grid search results if used
    if RIDGE_ALPHA is None:
        save_dict['alpha_grid_search_results'] = alpha_results
        save_dict['best_alpha'] = best_alpha
        save_dict['best_validation_score'] = best_score
    
    np.savez(out_file, **save_dict)
    print(f"[INFO] Saved results to {out_file}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Ridge alpha: {ridge_alpha:.2f}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training correlation: {train_mean_corr:.4f}")
    print(f"Test correlation: {test_mean_corr:.4f}")
    print(f"Generalization gap (MSE): {test_mse - train_mse:.6f}")
    print(f"Generalization gap (Corr): {train_mean_corr - test_mean_corr:.4f}")
    print(f"\nEnvelope Performance:")
    print(f"Training envelope correlation: {train_envelope_corr:.4f}")
    print(f"Test envelope correlation: {test_envelope_corr:.4f}")
    print(f"Envelope generalization gap (Corr): {train_envelope_corr - test_envelope_corr:.4f}")


def main_cross_modality():
    """
    Train on all vocalized data and test on both mimed and imagined data (cross-modality evaluation).
    Uses alpha=100 as determined from previous grid search.
    """
    cochba = load_cochba(COCHBA_MAT_PATH)
    print(f"[INFO] cochba shape: {cochba.shape}")
    
    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")
    
    print(f"\n" + "="*60)
    print("CROSS-MODALITY EVALUATION")
    print("="*60)
    print(f"[INFO] Found {len(wav_files)} audio files")
    
    # Use alpha=100 as determined from grid search
    ridge_alpha = 100.0
    
    # ---- 1) Collect all VOCALIZED data for training ----
    print("\n" + "="*60)
    print("COLLECTING VOCALIZED DATA (TRAINING SET)")
    print("="*60)
    X_train, y_train, file_info_train = collect_all_data(
        wav_files, modality="Vocalized", max_files=None
    )
    
    if len(X_train) == 0:
        raise ValueError("No vocalized data collected!")
    
    print(f"[INFO] Collected {len(X_train)} windows from vocalized data")
    
    # ---- 2) Collect all MIMED data for testing ----
    print("\n" + "="*60)
    print("COLLECTING MIMED DATA (TEST SET)")
    print("="*60)
    X_test_mimed, y_test_mimed, file_info_mimed = collect_all_data(
        wav_files, modality="Mimed", max_files=None
    )
    
    if len(X_test_mimed) == 0:
        print("[WARN] No mimed data collected!")
        X_test_mimed, y_test_mimed = None, None
    else:
        print(f"[INFO] Collected {len(X_test_mimed)} windows from mimed data")
    
    # ---- 3) Collect all IMAGINED data for testing ----
    print("\n" + "="*60)
    print("COLLECTING IMAGINED DATA (TEST SET)")
    print("="*60)
    X_test_imagined, y_test_imagined, file_info_imagined = collect_all_data(
        wav_files, modality="Imagined", max_files=None
    )
    
    if len(X_test_imagined) == 0:
        print("[WARN] No imagined data collected!")
        X_test_imagined, y_test_imagined = None, None
    else:
        print(f"[INFO] Collected {len(X_test_imagined)} windows from imagined data")
    
    if X_test_mimed is None and X_test_imagined is None:
        raise ValueError("No test data collected! Need at least mimed or imagined data.")
    
    # ---- 4) Normalize spectrograms (fit on training data only) ----
    print("\n" + "="*60)
    print("NORMALIZING SPECTROGRAMS")
    print("="*60)
    
    # Normalize training data
    y_train_norm, spect_scaler = normalize_spectrograms(y_train, None)
    print(f"[INFO] Normalized training spectrograms: mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")
    
    # Normalize test data using training scaler
    if X_test_mimed is not None:
        y_test_mimed_norm = spect_scaler.transform(y_test_mimed)
        print(f"[INFO] Normalized mimed spectrograms: mean={y_test_mimed_norm.mean():.4f}, std={y_test_mimed_norm.std():.4f}")
    
    if X_test_imagined is not None:
        y_test_imagined_norm = spect_scaler.transform(y_test_imagined)
        print(f"[INFO] Normalized imagined spectrograms: mean={y_test_imagined_norm.mean():.4f}, std={y_test_imagined_norm.std():.4f}")
    
    # ---- 5) Train decoder on ALL vocalized data ----
    print("\n" + "="*60)
    print(f"TRAINING LINEAR DECODER ON ALL VOCALIZED DATA (alpha={ridge_alpha:.2f})")
    print("="*60)
    decoder, scaler = train_linear_decoder(
        X_train, y_train_norm, alpha=ridge_alpha
    )
    
    # ---- 6) Evaluate on training set (vocalized) ----
    print("\n" + "="*60)
    print("EVALUATING ON TRAINING SET (VOCALIZED)")
    print("="*60)
    y_train_pred_norm = reconstruct_spectrogram(X_train, decoder, scaler)
    y_train_pred = denormalize_spectrograms(y_train_pred_norm, spect_scaler)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_correlations = []
    for f in range(y_train.shape[1]):
        corr = np.corrcoef(y_train[:, f], y_train_pred[:, f])[0, 1]
        train_correlations.append(corr)
    train_mean_corr = np.mean(train_correlations)
    
    train_envelope_true = compute_spectrogram_envelope(y_train)
    train_envelope_pred = compute_spectrogram_envelope(y_train_pred)
    train_envelope_corr = np.corrcoef(train_envelope_true, train_envelope_pred)[0, 1]
    train_envelope_mse = mean_squared_error(train_envelope_true, train_envelope_pred)
    
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Training mean correlation: {train_mean_corr:.4f}")
    print(f"Training envelope correlation: {train_envelope_corr:.4f}, MSE: {train_envelope_mse:.6f}")
    
    # ---- 7) Evaluate on test set (MIMED - cross-modality) ----
    results_mimed = None
    if X_test_mimed is not None:
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET (MIMED - CROSS-MODALITY)")
        print("="*60)
        y_test_mimed_pred_norm = reconstruct_spectrogram(X_test_mimed, decoder, scaler)
        y_test_mimed_pred = denormalize_spectrograms(y_test_mimed_pred_norm, spect_scaler)
        
        test_mimed_mse = mean_squared_error(y_test_mimed, y_test_mimed_pred)
        test_mimed_correlations = []
        for f in range(y_test_mimed.shape[1]):
            corr = np.corrcoef(y_test_mimed[:, f], y_test_mimed_pred[:, f])[0, 1]
            test_mimed_correlations.append(corr)
        test_mimed_mean_corr = np.mean(test_mimed_correlations)
        
        test_mimed_envelope_true = compute_spectrogram_envelope(y_test_mimed)
        test_mimed_envelope_pred = compute_spectrogram_envelope(y_test_mimed_pred)
        test_mimed_envelope_corr = np.corrcoef(test_mimed_envelope_true, test_mimed_envelope_pred)[0, 1]
        test_mimed_envelope_mse = mean_squared_error(test_mimed_envelope_true, test_mimed_envelope_pred)
        
        print(f"Test MSE (Mimed): {test_mimed_mse:.6f}")
        print(f"Test mean correlation (Mimed): {test_mimed_mean_corr:.4f}")
        print(f"Test envelope correlation (Mimed): {test_mimed_envelope_corr:.4f}, MSE: {test_mimed_envelope_mse:.6f}")
        
        results_mimed = {
            'X_test': X_test_mimed,
            'y_test': y_test_mimed,
            'y_test_pred': y_test_mimed_pred,
            'mse': test_mimed_mse,
            'correlations': test_mimed_correlations,
            'mean_correlation': test_mimed_mean_corr,
            'envelope_true': test_mimed_envelope_true,
            'envelope_pred': test_mimed_envelope_pred,
            'envelope_correlation': test_mimed_envelope_corr,
            'envelope_mse': test_mimed_envelope_mse
        }
    
    # ---- 8) Evaluate on test set (IMAGINED - cross-modality) ----
    results_imagined = None
    if X_test_imagined is not None:
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET (IMAGINED - CROSS-MODALITY)")
        print("="*60)
        y_test_imagined_pred_norm = reconstruct_spectrogram(X_test_imagined, decoder, scaler)
        y_test_imagined_pred = denormalize_spectrograms(y_test_imagined_pred_norm, spect_scaler)
        
        test_imagined_mse = mean_squared_error(y_test_imagined, y_test_imagined_pred)
        test_imagined_correlations = []
        for f in range(y_test_imagined.shape[1]):
            corr = np.corrcoef(y_test_imagined[:, f], y_test_imagined_pred[:, f])[0, 1]
            test_imagined_correlations.append(corr)
        test_imagined_mean_corr = np.mean(test_imagined_correlations)
        
        test_imagined_envelope_true = compute_spectrogram_envelope(y_test_imagined)
        test_imagined_envelope_pred = compute_spectrogram_envelope(y_test_imagined_pred)
        test_imagined_envelope_corr = np.corrcoef(test_imagined_envelope_true, test_imagined_envelope_pred)[0, 1]
        test_imagined_envelope_mse = mean_squared_error(test_imagined_envelope_true, test_imagined_envelope_pred)
        
        print(f"Test MSE (Imagined): {test_imagined_mse:.6f}")
        print(f"Test mean correlation (Imagined): {test_imagined_mean_corr:.4f}")
        print(f"Test envelope correlation (Imagined): {test_imagined_envelope_corr:.4f}, MSE: {test_imagined_envelope_mse:.6f}")
        
        results_imagined = {
            'X_test': X_test_imagined,
            'y_test': y_test_imagined,
            'y_test_pred': y_test_imagined_pred,
            'mse': test_imagined_mse,
            'correlations': test_imagined_correlations,
            'mean_correlation': test_imagined_mean_corr,
            'envelope_true': test_imagined_envelope_true,
            'envelope_pred': test_imagined_envelope_pred,
            'envelope_correlation': test_imagined_envelope_corr,
            'envelope_mse': test_imagined_envelope_mse
        }
    
    # ---- 9) Visualizations ----
    print("\n[INFO] Creating visualizations...")
    
    # Determine number of test modalities for plotting
    n_test_modalities = sum([X_test_mimed is not None, X_test_imagined is not None])
    
    # Spectrogram comparison
    n_rows = 1 + n_test_modalities
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 + n_test_modalities * 4), sharex='col', sharey='row')
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Training set (always present)
    row = 0
    im0 = axes[row, 0].imshow(y_train[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[row, 0].set_title(f'Training (Vocalized): True Spectrogram (first 200 windows)')
    axes[row, 0].set_ylabel('Frequency bin')
    plt.colorbar(im0, ax=axes[row, 0])
    
    im1 = axes[row, 1].imshow(y_train_pred[:200, :].T, aspect='auto', origin='lower', interpolation='none')
    axes[row, 1].set_title(f'Training (Vocalized): Reconstructed (MSE={train_mse:.4f}, Corr={train_mean_corr:.3f})')
    axes[row, 1].set_ylabel('Frequency bin')
    plt.colorbar(im1, ax=axes[row, 1])
    
    # Mimed test set
    if results_mimed is not None:
        row += 1
        im2 = axes[row, 0].imshow(results_mimed['y_test'][:200, :].T, aspect='auto', origin='lower', interpolation='none')
        axes[row, 0].set_title(f'Test (Mimed): True Spectrogram (first 200 windows)')
        axes[row, 0].set_ylabel('Frequency bin')
        plt.colorbar(im2, ax=axes[row, 0])
        
        im3 = axes[row, 1].imshow(results_mimed['y_test_pred'][:200, :].T, aspect='auto', origin='lower', interpolation='none')
        axes[row, 1].set_title(f'Test (Mimed): Reconstructed (MSE={results_mimed["mse"]:.4f}, Corr={results_mimed["mean_correlation"]:.3f})')
        axes[row, 1].set_ylabel('Frequency bin')
        plt.colorbar(im3, ax=axes[row, 1])
    
    # Imagined test set
    if results_imagined is not None:
        row += 1
        im4 = axes[row, 0].imshow(results_imagined['y_test'][:200, :].T, aspect='auto', origin='lower', interpolation='none')
        axes[row, 0].set_title(f'Test (Imagined): True Spectrogram (first 200 windows)')
        axes[row, 0].set_xlabel('Window index')
        axes[row, 0].set_ylabel('Frequency bin')
        plt.colorbar(im4, ax=axes[row, 0])
        
        im5 = axes[row, 1].imshow(results_imagined['y_test_pred'][:200, :].T, aspect='auto', origin='lower', interpolation='none')
        axes[row, 1].set_title(f'Test (Imagined): Reconstructed (MSE={results_imagined["mse"]:.4f}, Corr={results_imagined["mean_correlation"]:.3f})')
        axes[row, 1].set_xlabel('Window index')
        axes[row, 1].set_ylabel('Frequency bin')
        plt.colorbar(im5, ax=axes[row, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cross_modality_vocalized_to_all.png"), dpi=150)
    print(f"[INFO] Saved spectrogram comparison to {OUT_DIR}/cross_modality_vocalized_to_all.png")
    
    # Envelope comparison
    fig, axes = plt.subplots(1 + n_test_modalities, 1, figsize=(16, 3 + n_test_modalities * 3), sharex=True)
    if 1 + n_test_modalities == 1:
        axes = [axes]
    
    row = 0
    axes[row].plot(train_envelope_true[:500], label='True (Vocalized)', linewidth=2, alpha=0.8)
    axes[row].plot(train_envelope_pred[:500], label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
    axes[row].set_title(f'Training (Vocalized): Spectrogram Envelope (first 500 windows)\nCorr={train_envelope_corr:.3f}, MSE={train_envelope_mse:.4f}')
    axes[row].set_ylabel('Average Amplitude')
    axes[row].legend()
    axes[row].grid(True, alpha=0.3)
    
    if results_mimed is not None:
        row += 1
        axes[row].plot(results_mimed['envelope_true'][:500], label='True (Mimed)', linewidth=2, alpha=0.8)
        axes[row].plot(results_mimed['envelope_pred'][:500], label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        axes[row].set_title(f'Test (Mimed): Spectrogram Envelope (first 500 windows)\nCorr={results_mimed["envelope_correlation"]:.3f}, MSE={results_mimed["envelope_mse"]:.4f}')
        axes[row].set_ylabel('Average Amplitude')
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
    
    if results_imagined is not None:
        row += 1
        axes[row].plot(results_imagined['envelope_true'][:500], label='True (Imagined)', linewidth=2, alpha=0.8)
        axes[row].plot(results_imagined['envelope_pred'][:500], label='Reconstructed', linewidth=2, alpha=0.8, linestyle='--')
        axes[row].set_title(f'Test (Imagined): Spectrogram Envelope (first 500 windows)\nCorr={results_imagined["envelope_correlation"]:.3f}, MSE={results_imagined["envelope_mse"]:.4f}')
        axes[row].set_xlabel('Window Index')
        axes[row].set_ylabel('Average Amplitude')
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cross_modality_envelopes_all.png"), dpi=150)
    print(f"[INFO] Saved envelope visualization to {OUT_DIR}/cross_modality_envelopes_all.png")
    
    # ---- 10) Save results ----
    out_file = os.path.join(OUT_DIR, "cross_modality_results.npz")
    save_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'train_mse': train_mse,
        'train_correlations': train_correlations,
        'train_mean_correlation': train_mean_corr,
        'train_envelope_true': train_envelope_true,
        'train_envelope_pred': train_envelope_pred,
        'train_envelope_correlation': train_envelope_corr,
        'train_envelope_mse': train_envelope_mse,
        'window_duration_ms': WINDOW_DURATION_MS,
        'hop_size_ms': HOP_SIZE_MS,
        'ridge_alpha': ridge_alpha,
        'train_modality': "Vocalized"
    }
    
    save_dict["train_file_info"] = np.array(file_info_train, dtype=object)
    train_sent, train_rep = split_file_info(file_info_train)
    save_dict["train_sentence"] = train_sent
    save_dict["train_rep"] = train_rep

    # Add mimed results if available
    if results_mimed is not None:
        save_dict["mimed_file_info"] = np.array(file_info_mimed, dtype=object)
        mimed_sent, mimed_rep = split_file_info(file_info_mimed)
        save_dict["mimed_sentence"] = mimed_sent
        save_dict["mimed_rep"] = mimed_rep
        for key, value in results_mimed.items():
            save_dict[f'mimed_{key}'] = value
    
    # Add imagined results if available
    if results_imagined is not None:
        save_dict["imagined_file_info"] = np.array(file_info_imagined, dtype=object)
        imag_sent, imag_rep = split_file_info(file_info_imagined)
        save_dict["imagined_sentence"] = imag_sent
        save_dict["imagined_rep"] = imag_rep
        for key, value in results_imagined.items():
            save_dict[f'imagined_{key}'] = value
    
    
    # ---- 11) Comprehensive Evaluation ----
    train_results_dict = {
        'y_true': y_train,
        'y_pred': y_train_pred,
        'mean_correlation': train_mean_corr,
        'envelope_true': train_envelope_true,
        'envelope_pred': train_envelope_pred,
        'envelope_correlation': train_envelope_corr
    }
    
    eval_results = {}
    
    if results_mimed is not None:
        test_results_dict = {
            'y_true': results_mimed['y_test'],
            'y_pred': results_mimed['y_test_pred'],
            'mean_correlation': results_mimed['mean_correlation'],
            'envelope_true': results_mimed['envelope_true'],
            'envelope_pred': results_mimed['envelope_pred'],
            'envelope_correlation': results_mimed['envelope_correlation']
        }
        eval_results['mimed'] = evaluate_cross_modality_performance(
            train_results_dict, test_results_dict, test_modality_name="Mimed", n_permutations=1000
        )
    
    if results_imagined is not None:
        test_results_dict = {
            'y_true': results_imagined['y_test'],
            'y_pred': results_imagined['y_test_pred'],
            'mean_correlation': results_imagined['mean_correlation'],
            'envelope_true': results_imagined['envelope_true'],
            'envelope_pred': results_imagined['envelope_pred'],
            'envelope_correlation': results_imagined['envelope_correlation']
        }
        eval_results['imagined'] = evaluate_cross_modality_performance(
            train_results_dict, test_results_dict, test_modality_name="Imagined", n_permutations=N_PERM
        )
    
    # ---- 12) Summary ----
    print("\n" + "="*60)
    print("CROSS-MODALITY EVALUATION SUMMARY")
    print("="*60)
    print(f"Train modality: Vocalized")
    print(f"Ridge alpha: {ridge_alpha:.2f}")
    print(f"Training samples (Vocalized): {len(X_train)}")
    print(f"\nTraining (Vocalized) Performance:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  Correlation: {train_mean_corr:.4f}")
    print(f"  Envelope correlation: {train_envelope_corr:.4f}")
    
    if results_mimed is not None:
        print(f"\nTest (Mimed - Cross-Modality) Performance:")
        print(f"  Test samples (Mimed): {len(results_mimed['X_test'])}")
        print(f"  MSE: {results_mimed['mse']:.6f}")
        print(f"  Correlation: {results_mimed['mean_correlation']:.4f}")
        print(f"  Envelope correlation: {results_mimed['envelope_correlation']:.4f}")
        print(f"  Performance drop vs training:")
        print(f"    MSE increase: {results_mimed['mse'] - train_mse:.6f}")
        print(f"    Correlation drop: {train_mean_corr - results_mimed['mean_correlation']:.4f}")
        print(f"    Envelope correlation drop: {train_envelope_corr - results_mimed['envelope_correlation']:.4f}")
        # if 'mimed' in eval_results:
        #     print(f"  Statistical significance: P = {eval_results['mimed']['significance']['spectrogram']['p_value']:.4f}")
        if 'mimed' in eval_results:
            print(f"  Rating: {eval_results['mimed']['benchmarks']['spectrogram_rating']}")
    
    if results_imagined is not None:
        print(f"\nTest (Imagined - Cross-Modality) Performance:")
        print(f"  Test samples (Imagined): {len(results_imagined['X_test'])}")
        print(f"  MSE: {results_imagined['mse']:.6f}")
        print(f"  Correlation: {results_imagined['mean_correlation']:.4f}")
        print(f"  Envelope correlation: {results_imagined['envelope_correlation']:.4f}")
        print(f"  Performance drop vs training:")
        print(f"    MSE increase: {results_imagined['mse'] - train_mse:.6f}")
        print(f"    Correlation drop: {train_mean_corr - results_imagined['mean_correlation']:.4f}")
        print(f"    Envelope correlation drop: {train_envelope_corr - results_imagined['envelope_correlation']:.4f}")
        # if 'imagined' in eval_results:
        #     print(f"  Statistical significance: P = {eval_results['imagined']['significance']['spectrogram']['p_value']:.4f}")
        if 'imagined' in eval_results:
            print(f"  Rating: {eval_results['imagined']['benchmarks']['spectrogram_rating']}")
    
        # ---- finalize save_dict ----
    
    save_dict['evaluation_results'] = np.array(eval_results, dtype=object)

    save_dict["X_shape"] = np.array(X_train.shape, dtype=int)
    save_dict["Y_shape"] = np.array(y_train.shape, dtype=int)
    save_dict["N_FREQ_KEEP"] = N_FREQ_KEEP
    save_dict["N_CH_KEEP"]   = N_CH_KEEP
    save_dict["FRL_MS"]      = FRL_MS
    save_dict["TAU_MS"]      = TAU_MS
    save_dict["decoder_coef"] = decoder.coef_          # shape (F, D)
    save_dict["decoder_coef_shape"] = np.array(decoder.coef_.shape, dtype=int)
    save_dict["x_scaler_class"] = np.array([scaler.__class__.__name__], dtype=object)
    if hasattr(scaler, "mean_"):
        save_dict["x_scaler_mean"] = scaler.mean_
    if hasattr(scaler, "scale_"):
        save_dict["x_scaler_scale"] = scaler.scale_
    if hasattr(scaler, "var_"):
        save_dict["x_scaler_var"] = scaler.var_
    save_dict["decoder_intercept"] = decoder.intercept_
    save_dict["y_scaler_mean"] = spect_scaler.mean_
    save_dict["y_scaler_scale"] = spect_scaler.scale_
    save_dict["y_scaler_var"]   = spect_scaler.var_
    np.savez(out_file, **save_dict)
    print(f"[INFO] Saved results to {out_file}")    

def main_cross_modality_with_vcl_test():
    """
    Train on VOCALIZED-TRAIN (subset) and test on:
      - VOCALIZED-TEST (held-out sentences)  [NEW]
      - MIMED (all available)
      - IMAGINED (all available)

    This produces a cross_modality_results.npz that includes vocalized_* test keys
    so analyze_cross_modality_per_sentence.py can plot Vocalized performance fairly.
    """
    cochba = load_cochba(COCHBA_MAT_PATH)
    print(f"[INFO] cochba shape: {cochba.shape}")

    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")

    print("\n" + "="*60)
    print("CROSS-MODALITY EVALUATION (WITH VOCALIZED HELD-OUT TEST)")
    print("="*60)
    print(f"[INFO] Found {len(wav_files)} audio files")

    ridge_alpha = 100.0

    # ------------------------------------------------------------
    # 1) Collect VOCALIZED data (full pool), then split into train/test
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print("COLLECTING VOCALIZED DATA (POOL)")
    print("="*60)
    X_voc_all, y_voc_all, file_info_voc = collect_all_data(
        wav_files, modality="Vocalized", max_files=None
    )
    if len(X_voc_all) == 0:
        raise ValueError("No vocalized data collected!")

    print(f"[INFO] Collected {len(X_voc_all)} windows from vocalized data")

    # Split vocalized by sentence for a real held-out vocalized test
    print("\n" + "="*60)
    print("SPLITTING VOCALIZED BY SENTENCE (TRAIN vs TEST)")
    print("="*60)
    voc_train_mask, voc_test_mask, voc_train_sents, voc_test_sents = split_by_sentence(
        file_info_voc, test_size=0.2, random_state=42
    )

    X_train = X_voc_all[voc_train_mask]
    y_train = y_voc_all[voc_train_mask]
    X_test_vocalized = X_voc_all[voc_test_mask]
    y_test_vocalized = y_voc_all[voc_test_mask]

    file_info_train = [fi for fi, m in zip(file_info_voc, voc_train_mask) if m]
    file_info_vocalized_test = [fi for fi, m in zip(file_info_voc, voc_test_mask) if m]

    print(f"[INFO] Vocalized train windows: {len(X_train)}")
    print(f"[INFO] Vocalized test  windows: {len(X_test_vocalized)}")

    # ------------------------------------------------------------
    # 2) Collect MIMED + IMAGINED data (tests)
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print("COLLECTING MIMED DATA (TEST SET)")
    print("="*60)
    X_test_mimed, y_test_mimed, file_info_mimed = collect_all_data(
        wav_files, modality="Mimed", max_files=None
    )
    if len(X_test_mimed) == 0:
        print("[WARN] No mimed data collected!")
        X_test_mimed, y_test_mimed, file_info_mimed = None, None, None
    else:
        print(f"[INFO] Collected {len(X_test_mimed)} windows from mimed data")

    print("\n" + "="*60)
    print("COLLECTING IMAGINED DATA (TEST SET)")
    print("="*60)
    X_test_imagined, y_test_imagined, file_info_imagined = collect_all_data(
        wav_files, modality="Imagined", max_files=None
    )
    if len(X_test_imagined) == 0:
        print("[WARN] No imagined data collected!")
        X_test_imagined, y_test_imagined, file_info_imagined = None, None, None
    else:
        print(f"[INFO] Collected {len(X_test_imagined)} windows from imagined data")

    # ------------------------------------------------------------
    # 3) Normalize spectrograms (fit on VOCALIZED TRAIN only)
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print("NORMALIZING SPECTROGRAMS (FIT ON VOCALIZED TRAIN)")
    print("="*60)

    y_train_norm, spect_scaler = normalize_spectrograms(y_train, None)
    print(f"[INFO] y_train_norm: mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")

    # Normalize tests using training scaler
    y_test_vocalized_norm = spect_scaler.transform(y_test_vocalized)

    if X_test_mimed is not None:
        y_test_mimed_norm = spect_scaler.transform(y_test_mimed)
    if X_test_imagined is not None:
        y_test_imagined_norm = spect_scaler.transform(y_test_imagined)

    # ------------------------------------------------------------
    # 4) Train linear decoder on VOCALIZED TRAIN only
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print(f"TRAINING LINEAR DECODER ON VOCALIZED TRAIN (alpha={ridge_alpha:.2f})")
    print("="*60)
    decoder, scaler = train_linear_decoder(X_train, y_train_norm, alpha=ridge_alpha)

    # ------------------------------------------------------------
    # 5) Evaluate helper (reused for vocalized/mimed/imagined)
    # ------------------------------------------------------------
    def eval_one(X, y_true, label):
        y_pred_norm = reconstruct_spectrogram(X, decoder, scaler)
        y_pred = denormalize_spectrograms(y_pred_norm, spect_scaler)

        mse = mean_squared_error(y_true, y_pred)

        corrs = []
        for f in range(y_true.shape[1]):
            c = np.corrcoef(y_true[:, f], y_pred[:, f])[0, 1]
            corrs.append(c)
        mean_corr = float(np.mean(corrs))

        env_true = compute_spectrogram_envelope(y_true)
        env_pred = compute_spectrogram_envelope(y_pred)
        env_corr = float(np.corrcoef(env_true, env_pred)[0, 1])
        env_mse = mean_squared_error(env_true, env_pred)

        print(f"\n[{label}] MSE={mse:.6f}  mean_corr={mean_corr:.4f}  env_corr={env_corr:.4f}")

        return {
            "X_test": X,
            "y_test": y_true,
            "y_test_pred": y_pred,
            "mse": mse,
            "correlations": corrs,
            "mean_correlation": mean_corr,
            "envelope_true": env_true,
            "envelope_pred": env_pred,
            "envelope_correlation": env_corr,
            "envelope_mse": env_mse,
        }

    # ------------------------------------------------------------
    # 6) Evaluate: train (for reference), vocalized_test, mimed, imagined
    # ------------------------------------------------------------
    print("\n" + "="*60)
    print("EVALUATING ON VOCALIZED TRAIN (REFERENCE)")
    print("="*60)
    train_results = eval_one(X_train, y_train, "Vocalized-Train")

    print("\n" + "="*60)
    print("EVALUATING ON VOCALIZED TEST (HELD-OUT SENTENCES)  [NEW]")
    print("="*60)
    results_vocalized = eval_one(X_test_vocalized, y_test_vocalized, "Vocalized-Test")

    results_mimed = None
    if X_test_mimed is not None:
        print("\n" + "="*60)
        print("EVALUATING ON MIMED (CROSS-MODALITY)")
        print("="*60)
        results_mimed = eval_one(X_test_mimed, y_test_mimed, "Mimed-Test")

    results_imagined = None
    if X_test_imagined is not None:
        print("\n" + "="*60)
        print("EVALUATING ON IMAGINED (CROSS-MODALITY)")
        print("="*60)
        results_imagined = eval_one(X_test_imagined, y_test_imagined, "Imagined-Test")

    # ------------------------------------------------------------
    # 7) Save NPZ with vocalized_* test keys
    # ------------------------------------------------------------
    out_file = os.path.join(OUT_DIR, "cross_modality_results_with_vcl_test.npz")

    save_dict = {
        # training subset (vocalized train)
        "X_train": X_train,
        "y_train": y_train,
        "y_train_pred": train_results["y_test_pred"],
        "train_mse": train_results["mse"],
        "train_correlations": train_results["correlations"],
        "train_mean_correlation": train_results["mean_correlation"],
        "train_envelope_true": train_results["envelope_true"],
        "train_envelope_pred": train_results["envelope_pred"],
        "train_envelope_correlation": train_results["envelope_correlation"],
        "train_envelope_mse": train_results["envelope_mse"],
        "train_modality": "Vocalized",
        "ridge_alpha": ridge_alpha,
        "window_duration_ms": WINDOW_DURATION_MS,
        "hop_size_ms": HOP_SIZE_MS,
    }

    # train file info
    save_dict["train_file_info"] = np.array(file_info_train, dtype=object)
    train_sent, train_rep = split_file_info(file_info_train)
    save_dict["train_sentence"] = train_sent
    save_dict["train_rep"] = train_rep

    # vocalized test (NEW PREFIX)
    save_dict["vocalized_file_info"] = np.array(file_info_vocalized_test, dtype=object)
    voc_sent, voc_rep = split_file_info(file_info_vocalized_test)
    save_dict["vocalized_sentence"] = voc_sent
    save_dict["vocalized_rep"] = voc_rep

    # store vocalized test arrays under vocalized_*
    for key, value in results_vocalized.items():
        save_dict[f"vocalized_{key}"] = value

    # mimed
    if results_mimed is not None:
        save_dict["mimed_file_info"] = np.array(file_info_mimed, dtype=object)
        mimed_sent, mimed_rep = split_file_info(file_info_mimed)
        save_dict["mimed_sentence"] = mimed_sent
        save_dict["mimed_rep"] = mimed_rep
        for key, value in results_mimed.items():
            save_dict[f"mimed_{key}"] = value

    # imagined
    if results_imagined is not None:
        save_dict["imagined_file_info"] = np.array(file_info_imagined, dtype=object)
        imag_sent, imag_rep = split_file_info(file_info_imagined)
        save_dict["imagined_sentence"] = imag_sent
        save_dict["imagined_rep"] = imag_rep
        for key, value in results_imagined.items():
            save_dict[f"imagined_{key}"] = value

    # decoder/scalers (for importance + ablation in analyzer)
    save_dict["decoder_coef"] = decoder.coef_
    save_dict["decoder_coef_shape"] = np.array(decoder.coef_.shape, dtype=int)
    save_dict["decoder_intercept"] = decoder.intercept_
    save_dict["x_scaler_class"] = np.array([scaler.__class__.__name__], dtype=object)
    if hasattr(scaler, "mean_"):
        save_dict["x_scaler_mean"] = scaler.mean_
    if hasattr(scaler, "scale_"):
        save_dict["x_scaler_scale"] = scaler.scale_
    if hasattr(scaler, "var_"):
        save_dict["x_scaler_var"] = scaler.var_

    save_dict["y_scaler_mean"] = spect_scaler.mean_
    save_dict["y_scaler_scale"] = spect_scaler.scale_
    save_dict["y_scaler_var"] = spect_scaler.var_

    # extra metadata
    save_dict["X_shape"] = np.array(X_train.shape, dtype=int)
    save_dict["Y_shape"] = np.array(y_train.shape, dtype=int)
    save_dict["N_FREQ_KEEP"] = N_FREQ_KEEP
    save_dict["N_CH_KEEP"] = N_CH_KEEP
    save_dict["FRL_MS"] = FRL_MS
    save_dict["TAU_MS"] = TAU_MS

    np.savez(out_file, **save_dict)
    print(f"[INFO] Saved results to {out_file}")

    print("\n" + "="*60)
    print("DONE: cross_modality_results.npz now includes vocalized_* test keys")
    print("="*60)


if __name__ == "__main__":
    # Uncomment the function you want to run:
    # main()  # Original generalization within vocalized modality
    # main_cross_modality()  # Cross-modality: train on vocalized, test on mimed and imagined
    main_cross_modality_with_vcl_test()  # Cross-modality: train on vocalized, test on vocalized, test on mimed and img

