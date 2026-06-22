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

OUT_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results_envelope"
os.makedirs(OUT_DIR, exist_ok=True)

N_PERM = 20
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
RIDGE_ALPHA = 100.0  # chosen alpha after CV on spectrograms

if SPECT_TOOLS_PATH not in sys.path:
    sys.path.insert(0, SPECT_TOOLS_PATH)
from spect_tools import cortical_model_python  # noqa: E402


# -----------------------------
# Envelope helpers
# -----------------------------
def compute_spectrogram_envelope_from_frames(spect_frames_2d):
    """
    spect_frames_2d: (N, F)
    returns: (N,)
    """
    return np.mean(spect_frames_2d, axis=1)


def create_sliding_windows_from_aligned_envelope(seeg_aligned, spectrogram, fs_spect):
    """
    Same as your windowing, but target is ENVELOPE at the predicted frame.

    seeg_aligned: (T_spec, C)
    spectrogram:  (T_spec, F)
    returns:
      windows: (N_windows, window_frames, C)
      env_targets: (N_windows, 1)   # column for sklearn/scalers consistency
    """
    window_frames = int(np.round(WINDOW_DURATION_MS / 1000.0 * fs_spect))
    hop_frames = int(np.round(HOP_SIZE_MS / 1000.0 * fs_spect))

    T_spec, C = seeg_aligned.shape
    F = spectrogram.shape[1]

    # precompute envelope per frame: (T_spec,)
    env_full = np.mean(spectrogram, axis=1)

    windows = []
    env_indices = []

    for t_start in range(0, T_spec - window_frames + 1, hop_frames):
        t_end = t_start + window_frames
        windows.append(seeg_aligned[t_start:t_end, :])
        env_indices.append(t_end - 1)  # predict envelope at end of window

    if len(windows) == 0:
        return np.array([]), np.array([])

    windows = np.array(windows)
    env_indices = np.array(env_indices)

    valid_mask = env_indices < T_spec
    windows = windows[valid_mask]
    env_indices = env_indices[valid_mask]

    env_targets = env_full[env_indices]          # (N,)
    env_targets = env_targets.reshape(-1, 1)     # (N,1)

    return windows, env_targets


def normalize_targets(y_train, y_test=None):
    """
    y_train: (N,1) or (N,F)
    """
    scaler = StandardScaler()
    y_train_norm = scaler.fit_transform(y_train)
    if y_test is None:
        return y_train_norm, scaler
    return y_train_norm, scaler.transform(y_test), scaler


def denormalize_targets(y_norm, scaler):
    return scaler.inverse_transform(y_norm)


# -----------------------------
# Data collection (same logic, envelope targets)
# -----------------------------
def collect_all_data_envelope(wav_files, modality="Vocalized", max_files=None):
    cochba = load_cochba(COCHBA_MAT_PATH)
    fs_spect = 1 / FRL_MS * 1000.0  # e.g., 100 Hz

    all_windows = []
    all_env = []
    file_info = []

    if max_files is not None:
        wav_files = wav_files[:max_files]

    for i, wav_path in enumerate(wav_files, 1):
        sentence, rep = parse_audio_fname(wav_path)
        v_seeg_path = corresponding_seeg_path(sentence, rep, modality)

        if not os.path.exists(v_seeg_path):
            print(f"[WARN] Missing sEEG for {sentence} rep{rep}: {v_seeg_path} (skipping)")
            continue

        print(f"[{i}/{len(wav_files)}] Collecting (ENVELOPE) from {sentence} rep{rep} [{modality}]")

        # audio -> spectrogram
        wav, fs_audio = load_wav_mono_norm(wav_path)
        wav = resample_poly(wav, up=8000, down=fs_audio)

        v_spect = cortical_model_python(
            wav, 8000, FRL_MS, TAU_MS,
            pplot=False, which="wav", cochba=cochba
        )
        v_spect = np.sqrt(v_spect)
        v_spect = v_spect[:, :N_FREQ_KEEP]  # (T_spec, F)

        # load sEEG
        v_X = np.load(v_seeg_path)          # (T_seeg, C_all)
        v_X = v_X[:, -N_CH_KEEP:]           # (T_seeg, C)

        # align sEEG to spectrogram time
        v_X_aligned = align_seeg_to_spect_time(v_X, T_spec=v_spect.shape[0])  # (T_spec, C)

        # windows + envelope targets
        windows, env_targets = create_sliding_windows_from_aligned_envelope(v_X_aligned, v_spect, fs_spect)

        if len(windows) == 0:
            print(f"[WARN] No windows for {sentence} rep{rep} (skipping)")
            continue

        all_windows.append(windows)
        all_env.append(env_targets)

        n_windows = len(windows)
        file_info.extend([(sentence, rep)] * n_windows)

        print(f"  -> Collected {n_windows} windows")

    if len(all_windows) == 0:
        return np.array([]), np.array([]), []

    all_windows = np.concatenate(all_windows, axis=0)   # (N, W, C)
    all_env = np.concatenate(all_env, axis=0)           # (N, 1)

    print(f"\n[INFO] Total collected: {len(all_windows)} windows")
    print(f"[INFO] Window shape: {all_windows.shape}")
    print(f"[INFO] Envelope shape: {all_env.shape}")

    return all_windows, all_env, file_info


def split_by_sentence(file_info, test_size=0.2, random_state=42):
    unique_sentences = sorted(set([s for s, _ in file_info]))
    train_sentences, test_sentences = train_test_split(
        unique_sentences, test_size=test_size, random_state=random_state
    )
    train_sentences = set(train_sentences)
    test_sentences = set(test_sentences)

    train_mask = np.array([s in train_sentences for s, _ in file_info])
    test_mask = np.array([s in test_sentences for s, _ in file_info])

    print(f"\n[INFO] Split by sentence:")
    print(f"  Training sentences ({len(train_sentences)}): {sorted(train_sentences)}")
    print(f"  Test sentences ({len(test_sentences)}): {sorted(test_sentences)}")
    print(f"  Training samples: {train_mask.sum()}")
    print(f"  Test samples: {test_mask.sum()}")

    return train_mask, test_mask, train_sentences, test_sentences


def split_file_info(file_info):
    sents = [s for s, r in file_info]
    reps  = [r for s, r in file_info]
    return np.array(sents, dtype=object), np.array(reps, dtype=int)


# -----------------------------
# Evaluation helpers
# -----------------------------
def mean_corr_1d(y_true_1d, y_pred_1d):
    """
    y_true_1d, y_pred_1d: (N,) or (N,1)
    returns scalar correlation
    """
    yt = y_true_1d.reshape(-1)
    yp = y_pred_1d.reshape(-1)
    return float(np.corrcoef(yt, yp)[0, 1])


# -----------------------------
# Main: cross-modality WITH held-out vocalized test
# -----------------------------
def main_cross_modality_envelope_with_vcl_test():
    """
    Train Ridge decoder on VOCALIZED-TRAIN to predict ENVELOPE.
    Test on:
      - VOCALIZED-TEST (held-out sentences)
      - MIMED (all)
      - IMAGINED (all)

    Saves: cross_modality_envelope_results_with_vcl_test.npz
    with the SAME-style keys as your spectrogram version.
    """
    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")

    print("\n" + "="*60)
    print("ENVELOPE DECODING: CROSS-MODALITY (WITH VOCALIZED HELD-OUT TEST)")
    print("="*60)
    print(f"[INFO] Found {len(wav_files)} audio files")

    ridge_alpha = RIDGE_ALPHA

    # 1) Collect VOCALIZED pool
    print("\n" + "="*60)
    print("COLLECTING VOCALIZED DATA (POOL)")
    print("="*60)
    X_voc_all, y_env_voc_all, file_info_voc = collect_all_data_envelope(
        wav_files, modality="Vocalized", max_files=None
    )
    if len(X_voc_all) == 0:
        raise ValueError("No vocalized data collected!")
    
    # split by sentence for held-out vocalized test
    print("\n" + "="*60)
    print("SPLITTING VOCALIZED BY SENTENCE (TRAIN vs TEST)")
    print("="*60)
    voc_train_mask, voc_test_mask, voc_train_sents, voc_test_sents = split_by_sentence(
        file_info_voc, test_size=0.2, random_state=42
    )

    X_train = X_voc_all[voc_train_mask]
    y_train = y_env_voc_all[voc_train_mask]         # (N,1)
    X_test_vocalized = X_voc_all[voc_test_mask]
    y_test_vocalized = y_env_voc_all[voc_test_mask]

    file_info_train = [fi for fi, m in zip(file_info_voc, voc_train_mask) if m]
    file_info_vocalized_test = [fi for fi, m in zip(file_info_voc, voc_test_mask) if m]

    print(f"[INFO] Vocalized train windows: {len(X_train)}")
    print(f"[INFO] Vocalized test  windows: {len(X_test_vocalized)}")

    # 2) Collect mimed + imagined (tests)
    print("\n" + "="*60)
    print("COLLECTING MIMED DATA (TEST SET)")
    print("="*60)
    X_test_mimed, y_test_mimed, file_info_mimed = collect_all_data_envelope(
        wav_files, modality="Mimed", max_files=None
    )
    if len(X_test_mimed) == 0:
        print("[WARN] No mimed data collected!")
        X_test_mimed, y_test_mimed, file_info_mimed = None, None, None

    print("\n" + "="*60)
    print("COLLECTING IMAGINED DATA (TEST SET)")
    print("="*60)
    X_test_imagined, y_test_imagined, file_info_imagined = collect_all_data_envelope(
        wav_files, modality="Imagined", max_files=None
    )
    if len(X_test_imagined) == 0:
        print("[WARN] No imagined data collected!")
        X_test_imagined, y_test_imagined, file_info_imagined = None, None, None

    # 3) Normalize targets (fit on vocalized train only)
    print("\n" + "="*60)
    print("NORMALIZING ENVELOPE (FIT ON VOCALIZED TRAIN)")
    print("="*60)
    y_train_norm, y_scaler = normalize_targets(y_train, None)
    y_test_vocalized_norm = y_scaler.transform(y_test_vocalized)

    if X_test_mimed is not None:
        y_test_mimed_norm = y_scaler.transform(y_test_mimed)
    if X_test_imagined is not None:
        y_test_imagined_norm = y_scaler.transform(y_test_imagined)

    # 4) Train linear decoder (same function; y is (N,1))
    print("\n" + "="*60)
    print(f"TRAINING RIDGE DECODER: sEEG → ENVELOPE (alpha={ridge_alpha:.2f})")
    print("="*60)
    decoder, x_scaler = train_linear_decoder(X_train, y_train_norm, alpha=ridge_alpha)

    # 5) Eval helper
    def eval_one(X, y_true, label):
        y_pred_norm = reconstruct_spectrogram(X, decoder, x_scaler)  # (N,1)
        y_pred = denormalize_targets(y_pred_norm, y_scaler)          # (N,1)

        mse = float(mean_squared_error(y_true, y_pred))
        corr = mean_corr_1d(y_true, y_pred)

        print(f"[{label}] MSE={mse:.6f}  corr={corr:.4f}")
        return {
            "X_test": X,
            "y_test": y_true,
            "y_test_pred": y_pred,
            "mse": mse,
            "correlations": np.array([corr], dtype=float),  # keep a 'correlations' field
            "mean_correlation": corr,
            # keep envelope fields (now identical to y)
            "envelope_true": y_true.reshape(-1),
            "envelope_pred": y_pred.reshape(-1),
            "envelope_correlation": corr,
            "envelope_mse": mse,
        }

    # 6) Evaluate: vocalized train/ref, vocalized test, mimed, imagined
    print("\n" + "="*60)
    print("EVALUATING ON VOCALIZED TRAIN (REFERENCE)")
    print("="*60)
    train_results = eval_one(X_train, y_train, "Vocalized-Train")

    print("\n" + "="*60)
    print("EVALUATING ON VOCALIZED TEST (HELD-OUT SENTENCES)")
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

    # 7) Plots (same spirit; no spectrogram imshow, just envelope traces)
    print("\n[INFO] Creating visualizations...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    n_show = 1000

    ax.plot(train_results["envelope_true"][:n_show], label="True (Vocalized-Train)", linewidth=2, alpha=0.8)
    ax.plot(train_results["envelope_pred"][:n_show], label="Pred (Vocalized-Train)", linewidth=2, alpha=0.8, linestyle="--")

    ax.set_title(f"sEEG→Envelope (Train)  corr={train_results['mean_correlation']:.3f}  mse={train_results['mse']:.4f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Envelope")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "envelope_train_trace.png"), dpi=150)

    # multi-panel test traces
    rows = 1 + int(results_mimed is not None) + int(results_imagined is not None) + 1  # +1 for vocalized test
    fig, axes = plt.subplots(rows, 1, figsize=(16, 3 * rows), sharex=False)
    r = 0

    def plot_block(ax, res, title_prefix):
        ax.plot(res["envelope_true"][:n_show], label="True", linewidth=2, alpha=0.8)
        ax.plot(res["envelope_pred"][:n_show], label="Pred", linewidth=2, alpha=0.8, linestyle="--")
        ax.set_title(f"{title_prefix}  corr={res['mean_correlation']:.3f}  mse={res['mse']:.4f}")
        ax.set_ylabel("Envelope")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plot_block(axes[r], train_results, "Vocalized-Train"); r += 1
    plot_block(axes[r], results_vocalized, "Vocalized-Test (held-out)"); r += 1

    if results_mimed is not None:
        plot_block(axes[r], results_mimed, "Mimed-Test"); r += 1
    if results_imagined is not None:
        plot_block(axes[r], results_imagined, "Imagined-Test"); r += 1

    axes[-1].set_xlabel("Window index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "envelope_cross_modality_traces.png"), dpi=150)

    # 8) Save NPZ with same-style keys
    out_file = os.path.join(OUT_DIR, "cross_modality_envelope_results_with_vcl_test.npz")

    save_dict = {
        "X_train": X_train,
        "y_train": y_train,  # envelope targets (N,1)
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

    # file info
    save_dict["train_file_info"] = np.array(file_info_train, dtype=object)
    train_sent, train_rep = split_file_info(file_info_train)
    save_dict["train_sentence"] = train_sent
    save_dict["train_rep"] = train_rep

    # vocalized test (prefix vocalized_*)
    save_dict["vocalized_file_info"] = np.array(file_info_vocalized_test, dtype=object)
    voc_sent, voc_rep = split_file_info(file_info_vocalized_test)
    save_dict["vocalized_sentence"] = voc_sent
    save_dict["vocalized_rep"] = voc_rep
    for key, value in results_vocalized.items():
        save_dict[f"vocalized_{key}"] = value

    # mimed (prefix mimed_*)
    if results_mimed is not None:
        save_dict["mimed_file_info"] = np.array(file_info_mimed, dtype=object)
        mimed_sent, mimed_rep = split_file_info(file_info_mimed)
        save_dict["mimed_sentence"] = mimed_sent
        save_dict["mimed_rep"] = mimed_rep
        for key, value in results_mimed.items():
            save_dict[f"mimed_{key}"] = value

    # imagined (prefix imagined_*)
    if results_imagined is not None:
        save_dict["imagined_file_info"] = np.array(file_info_imagined, dtype=object)
        imag_sent, imag_rep = split_file_info(file_info_imagined)
        save_dict["imagined_sentence"] = imag_sent
        save_dict["imagined_rep"] = imag_rep
        for key, value in results_imagined.items():
            save_dict[f"imagined_{key}"] = value

    # decoder/scalers (same spirit)
    save_dict["decoder_coef"] = decoder.coef_                 # shape (1, D) typically
    save_dict["decoder_coef_shape"] = np.array(decoder.coef_.shape, dtype=int)
    save_dict["decoder_intercept"] = decoder.intercept_
    save_dict["x_scaler_class"] = np.array([x_scaler.__class__.__name__], dtype=object)
    if hasattr(x_scaler, "mean_"):
        save_dict["x_scaler_mean"] = x_scaler.mean_
    if hasattr(x_scaler, "scale_"):
        save_dict["x_scaler_scale"] = x_scaler.scale_
    if hasattr(x_scaler, "var_"):
        save_dict["x_scaler_var"] = x_scaler.var_

    # y scaler (now envelope scaler)
    save_dict["y_scaler_mean"] = y_scaler.mean_
    save_dict["y_scaler_scale"] = y_scaler.scale_
    save_dict["y_scaler_var"] = y_scaler.var_

    # metadata
    save_dict["X_shape"] = np.array(X_train.shape, dtype=int)
    save_dict["Y_shape"] = np.array(y_train.shape, dtype=int)
    save_dict["N_FREQ_KEEP"] = N_FREQ_KEEP
    save_dict["N_CH_KEEP"] = N_CH_KEEP
    save_dict["FRL_MS"] = FRL_MS
    save_dict["TAU_MS"] = TAU_MS

    np.savez(out_file, **save_dict)
    print(f"[INFO] Saved results to {out_file}")

    print("\n" + "="*60)
    print("DONE: Envelope decoding results saved.")
    print("="*60)


if __name__ == "__main__":
    main_cross_modality_envelope_with_vcl_test()
