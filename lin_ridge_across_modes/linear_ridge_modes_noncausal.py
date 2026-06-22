import os
import sys
import glob
import numpy as np
from scipy.signal import resample_poly
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import from the per-sentence file
sys.path.insert(0, "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2")
from stimrcnt_per_sentence import (
    SPECT_TOOLS_PATH, COCHBA_MAT_PATH, AUDIO_DIR,
    FRL_MS, TAU_MS, N_FREQ_KEEP, N_CH_KEEP,
    WINDOW_DURATION_MS, HOP_SIZE_MS,
    load_cochba, load_wav_mono_norm, parse_audio_fname,
    corresponding_seeg_path, align_seeg_to_spect_time,
    train_linear_decoder, reconstruct_spectrogram
)

OUT_DIR = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
os.makedirs(OUT_DIR, exist_ok=True)

RIDGE_ALPHA = 100.0
PAST_MS   = 300.0
FUTURE_MS = 300.0


if SPECT_TOOLS_PATH not in sys.path:
    sys.path.insert(0, SPECT_TOOLS_PATH)
from spect_tools import cortical_model_python


# ----------------------------
# Helpers (same as your script)
# ----------------------------
def create_centered_windows_from_aligned(seeg_aligned, spectrogram, fs_spect,
                                         past_ms=PAST_MS, future_ms=FUTURE_MS):
    """
    Build centered windows: [t-past, ..., t+future] -> predict y[t].

    seeg_aligned : (T, C) aligned to spectrogram frames (same T)
    spectrogram  : (T, F)
    fs_spect     : frames/sec (e.g., 100 Hz)

    Returns
    -------
    windows : (N, L, C) where L = past_frames + future_frames + 1
    spect_frames : (N, F) where each row is spectrogram[t]
    frame_idx : (N,) indices t in original timebase (useful for debugging)
    """
    past_frames   = int(np.round(past_ms   / 1000.0 * fs_spect))
    future_frames = int(np.round(future_ms / 1000.0 * fs_spect))

    T, C = seeg_aligned.shape
    assert spectrogram.shape[0] == T, "seeg_aligned and spectrogram must have same T"
    F = spectrogram.shape[1]

    L = past_frames + future_frames + 1
    if T < L:
        return np.array([]), np.array([]), np.array([], dtype=int)

    windows = []
    frame_idx = []

    # valid t such that [t-past, ..., t+future] is in-bounds
    t0 = past_frames
    t1 = T - future_frames  # exclusive upper bound for t+future < T
    for t in range(t0, t1):
        win = seeg_aligned[t - past_frames : t + future_frames + 1, :]  # (L, C)
        windows.append(win)
        frame_idx.append(t)

    windows = np.asarray(windows)               # (N, L, C)
    frame_idx = np.asarray(frame_idx, dtype=int)
    spect_frames = spectrogram[frame_idx, :]    # (N, F)
    return windows, spect_frames, frame_idx

def create_sliding_windows_from_aligned(seeg_aligned, spectrogram, fs_spect):
    window_frames = int(np.round(WINDOW_DURATION_MS / 1000.0 * fs_spect))
    hop_frames    = int(np.round(HOP_SIZE_MS / 1000.0 * fs_spect))

    T_spec, C = seeg_aligned.shape
    F = spectrogram.shape[1]

    windows = []
    frame_idx = []
    for t_start in range(0, T_spec - window_frames + 1, hop_frames):
        t_end = t_start + window_frames
        windows.append(seeg_aligned[t_start:t_end, :])
        frame_idx.append(t_end - 1)

    if len(windows) == 0:
        return np.array([]), np.array([])

    windows = np.array(windows)
    frame_idx = np.array(frame_idx)

    valid = frame_idx < T_spec
    windows = windows[valid]
    frame_idx = frame_idx[valid]

    spect_frames = spectrogram[frame_idx, :]
    return windows, spect_frames


def normalize_spectrograms(y_train, y_test=None):
    spect_scaler = StandardScaler()
    y_train_norm = spect_scaler.fit_transform(y_train)
    if y_test is None:
        return y_train_norm, spect_scaler
    y_test_norm = spect_scaler.transform(y_test)
    return y_train_norm, y_test_norm, spect_scaler


def denormalize_spectrograms(y_norm, spect_scaler):
    return spect_scaler.inverse_transform(y_norm)


def compute_spectrogram_envelope(spectrogram):
    return np.mean(spectrogram, axis=1)


def split_file_info(file_info):
    sents = [s for s, r in file_info]
    reps  = [r for s, r in file_info]
    return np.array(sents, dtype=object), np.array(reps, dtype=int)


def split_by_sentence(file_info, test_size=0.2, random_state=42):
    unique_sentences = sorted(set([s for s, _ in file_info]))
    train_sents, test_sents = train_test_split(
        unique_sentences, test_size=test_size, random_state=random_state
    )
    train_sents = set(train_sents)
    test_sents  = set(test_sents)

    train_mask = np.array([s in train_sents for s, _ in file_info])
    test_mask  = np.array([s in test_sents  for s, _ in file_info])
    return train_mask, test_mask, train_sents, test_sents


def collect_all_data(wav_files, modality="Vocalized", max_files=None):
    cochba = load_cochba(COCHBA_MAT_PATH)
    fs_spect = 1 / FRL_MS * 1000.0  # 100 Hz
    fs_new = 8000

    all_windows = []
    all_spect_frames = []
    file_info = []

    if max_files is not None:
        wav_files = wav_files[:max_files]
    frame_indices = []
    for i, wav_path in enumerate(wav_files, 1):
        sentence, rep = parse_audio_fname(wav_path)
        seeg_path = corresponding_seeg_path(sentence, rep, modality)

        if not os.path.exists(seeg_path):
            print(f"[WARN] Missing sEEG for {modality} {sentence} rep{rep} (skipping)")
            continue

        print(f"[{i}/{len(wav_files)}] {modality}: {sentence} rep{rep}")

        wav, fs_audio = load_wav_mono_norm(wav_path)
        wav = resample_poly(wav, up=fs_new, down=fs_audio)

        spect = cortical_model_python(
            wav, fs_new, FRL_MS, TAU_MS, pplot=False, which="wav", cochba=cochba
        )
        spect = np.sqrt(spect)
        spect = spect[:, :N_FREQ_KEEP]  # (T_spec, F)

        X = np.load(seeg_path)          # (T_seeg, C_all)
        X = X[:, -N_CH_KEEP:]           # (T_seeg, C_keep)

        X_aligned = align_seeg_to_spect_time(X, T_spec=spect.shape[0])

        windows, spect_frames, t_idx = create_centered_windows_from_aligned(
            X_aligned, spect, fs_spect, past_ms=PAST_MS, future_ms=FUTURE_MS
        )
        frame_indices.append(t_idx)
        # windows, spect_frames = create_sliding_windows_from_aligned(
        #     X_aligned, spect, fs_spect
        # )
        if len(windows) == 0:
            print(f"[WARN] No windows created for {modality} {sentence} rep{rep} (skipping)")
            continue

        all_windows.append(windows)
        all_spect_frames.append(spect_frames)
        file_info.extend([(sentence, rep)] * len(windows))

    if len(all_windows) == 0:
        return np.array([]), np.array([]), []
    
    frame_indices = np.concatenate(frame_indices, axis=0)
    all_windows = np.concatenate(all_windows, axis=0)
    all_spect_frames = np.concatenate(all_spect_frames, axis=0)

    print(f"[INFO] {modality}: total windows {all_windows.shape}")
    print(f"[INFO] {modality}: total targets {all_spect_frames.shape}")
    # return all_windows, all_spect_frames, file_info
    return all_windows, all_spect_frames, file_info, frame_indices

# ----------------------------
# Core: run one experiment
# ----------------------------
def run_cross_modality_experiment(
    train_mode: str,
    all_modes=("Vocalized", "Mimed", "Imagined"),
    test_size=0.2,
    random_state=42,
    ridge_alpha=RIDGE_ALPHA,
    make_plots=True,
):
    """
    Three supported cases (just pick train_mode):
      1) train_mode="Vocalized" -> test on Vocalized-heldout + Mimed + Imagined
      2) train_mode="Mimed"     -> test on Mimed-heldout + Vocalized + Imagined
      3) train_mode="Imagined"  -> test on Imagined-heldout + Vocalized + Mimed

    Saves:
      - train_* : training subset (train_mode, held-in sentences)
      - {mode_lower}_* : evaluation set for each mode, including held-out test for train_mode
      - decoder/scaler params, sentence/rep arrays, etc.
    """
    assert train_mode in all_modes, f"train_mode must be one of {all_modes}"

    wav_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "Audio_*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No Audio_*.wav found in {AUDIO_DIR}")
    print(f"[INFO] Found {len(wav_files)} audio files")

    # 1) Collect train_mode pool and split by sentence -> train vs held-out test
    print("\n" + "="*70)
    print(f"COLLECTING TRAIN POOL: {train_mode}")
    print("="*70)
    X_pool, y_pool, fi_pool, frame_indices_pool = collect_all_data(wav_files, modality=train_mode, max_files=None)
    if len(X_pool) == 0:
        raise ValueError(f"No data collected for train_mode={train_mode}!")

    print("\n" + "="*70)
    print(f"SPLITTING {train_mode} BY SENTENCE (TRAIN vs HELD-OUT TEST)")
    print("="*70)
    train_mask, test_mask, train_sents, test_sents = split_by_sentence(
        fi_pool, test_size=test_size, random_state=random_state
    )

    X_train = X_pool[train_mask]
    y_train = y_pool[train_mask]
    X_test_trainmode = X_pool[test_mask]
    y_test_trainmode = y_pool[test_mask]

    fi_train = [fi for fi, m in zip(fi_pool, train_mask) if m]
    fi_test_trainmode = [fi for fi, m in zip(fi_pool, test_mask) if m]

    print(f"[INFO] {train_mode} train windows: {len(X_train)}")
    print(f"[INFO] {train_mode} test  windows: {len(X_test_trainmode)}")

    # 2) Collect other modes fully (no split)
    test_sets = {}   # mode -> (X, y, file_info)
    for mode in all_modes:
        if mode == train_mode:
            continue
        print("\n" + "="*70)
        print(f"COLLECTING FULL TEST SET: {mode}")
        print("="*70)
        X_m, y_m, fi_m, frame_indices = collect_all_data(wav_files, modality=mode, max_files=None)
        if len(X_m) == 0:
            print(f"[WARN] No data for mode={mode} (skipping this test)")
            continue
        test_sets[mode] = (X_m, y_m, fi_m)

    # 3) Normalize targets using TRAIN ONLY
    print("\n" + "="*70)
    print("NORMALIZING TARGET SPECTROGRAMS (FIT ON TRAIN ONLY)")
    print("="*70)
    y_train_norm, y_scaler = normalize_spectrograms(y_train, None)
    print(f"[INFO] y_train_norm mean={y_train_norm.mean():.4f}, std={y_train_norm.std():.4f}")

    # 4) Train ridge on TRAIN ONLY
    print("\n" + "="*70)
    print(f"TRAINING RIDGE DECODER (train_mode={train_mode}, alpha={ridge_alpha})")
    print("="*70)
    decoder, x_scaler = train_linear_decoder(X_train, y_train_norm, alpha=ridge_alpha)

    # 5) Eval helper
    def eval_one(X, y_true, label):
        y_pred_norm = reconstruct_spectrogram(X, decoder, x_scaler)
        y_pred = denormalize_spectrograms(y_pred_norm, y_scaler)

        mse = mean_squared_error(y_true, y_pred)
        corrs = []
        for f in range(y_true.shape[1]):
            c = np.corrcoef(y_true[:, f], y_pred[:, f])[0, 1]
            corrs.append(c)
        mean_corr = float(np.nanmean(corrs))

        env_true = compute_spectrogram_envelope(y_true)
        env_pred = compute_spectrogram_envelope(y_pred)
        env_corr = float(np.corrcoef(env_true, env_pred)[0, 1])
        env_mse = mean_squared_error(env_true, env_pred)

        print(f"[{label}] mse={mse:.6f}  mean_corr={mean_corr:.4f}  env_corr={env_corr:.4f}")

        return {
            "X_test": X,
            "y_test": y_true,
            "y_test_pred": y_pred,
            "mse": mse,
            "correlations": np.array(corrs, dtype=float),
            "mean_correlation": mean_corr,
            "envelope_true": env_true,
            "envelope_pred": env_pred,
            "envelope_correlation": env_corr,
            "envelope_mse": env_mse,
        }

    # 6) Evaluate train subset (reference)
    print("\n" + "="*70)
    print(f"EVALUATING: TRAIN SUBSET ({train_mode}-Train)")
    print("="*70)
    train_results = eval_one(X_train, y_train, f"{train_mode}-Train")

    # 7) Evaluate held-out test of train_mode (this will be saved under {train_mode_lower}_*)
    print("\n" + "="*70)
    print(f"EVALUATING: HELD-OUT TEST ({train_mode}-Test)")
    print("="*70)
    trainmode_test_results = eval_one(X_test_trainmode, y_test_trainmode, f"{train_mode}-Test")

    # 8) Evaluate other full modes
    other_results = {}
    for mode, (X_m, y_m, fi_m) in test_sets.items():
        print("\n" + "="*70)
        print(f"EVALUATING: FULL TEST ({mode})")
        print("="*70)
        other_results[mode] = eval_one(X_m, y_m, f"{mode}-Full")

    # 9) Save NPZ (same style as your current saving)
    train_mode_l = train_mode.lower()
    tag = f"centered_p{int(PAST_MS)}_f{int(FUTURE_MS)}"
    out_file = os.path.join(
        OUT_DIR, f"cross_modality_{tag}_train_{train_mode_l}_with_heldout_test.npz"
    )

    # out_file = os.path.join(OUT_DIR, f"cross_modality_train_{train_mode_l}_with_heldout_test.npz")

    save_dict = {
        # Always keep train_* for the training subset
        "train_modality": train_mode,
        "ridge_alpha": float(ridge_alpha),
        "window_duration_ms": float(WINDOW_DURATION_MS),
        "hop_size_ms": float(HOP_SIZE_MS),

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

        # metadata for train subset
        "train_file_info": np.array(fi_train, dtype=object),
        "train_sentence": split_file_info(fi_train)[0],
        "train_rep": split_file_info(fi_train)[1],

        # decoder/scalers for downstream importance/ablation
        "decoder_coef": decoder.coef_,
        "decoder_coef_shape": np.array(decoder.coef_.shape, dtype=int),
        "decoder_intercept": decoder.intercept_,

        "x_scaler_class": np.array([x_scaler.__class__.__name__], dtype=object),
        "y_scaler_mean": y_scaler.mean_,
        "y_scaler_scale": y_scaler.scale_,
        "y_scaler_var": y_scaler.var_,

        "N_FREQ_KEEP": int(N_FREQ_KEEP),
        "N_CH_KEEP": int(N_CH_KEEP),
        "FRL_MS": float(FRL_MS),
        "TAU_MS": float(TAU_MS),

        "X_shape": np.array(X_train.shape, dtype=int),
        "Y_shape": np.array(y_train.shape, dtype=int),

        "past_ms": float(PAST_MS),
        "future_ms": float(FUTURE_MS),
        "window_frames": int(np.round(PAST_MS/1000* (1/FRL_MS*1000.0))) + int(np.round(FUTURE_MS/1000*(1/FRL_MS*1000.0))) + 1,
    }

    # store x_scaler stats if present
    if hasattr(x_scaler, "mean_"):
        save_dict["x_scaler_mean"] = x_scaler.mean_
    if hasattr(x_scaler, "scale_"):
        save_dict["x_scaler_scale"] = x_scaler.scale_
    if hasattr(x_scaler, "var_"):
        save_dict["x_scaler_var"] = x_scaler.var_

    # Save held-out test for the training mode under its own prefix (vocalized_* / mimed_* / imagined_*)
    prefix = train_mode.lower()
    save_dict[f"{prefix}_file_info"] = np.array(fi_test_trainmode, dtype=object)
    sent_arr, rep_arr = split_file_info(fi_test_trainmode)
    save_dict[f"{prefix}_sentence"] = sent_arr
    save_dict[f"{prefix}_rep"] = rep_arr
    for k, v in trainmode_test_results.items():
        save_dict[f"{prefix}_{k}"] = v

    # Save other modes under their own prefixes too
    for mode, (X_m, y_m, fi_m) in test_sets.items():
        prefix = mode.lower()
        save_dict[f"{prefix}_file_info"] = np.array(fi_m, dtype=object)
        sent_arr, rep_arr = split_file_info(fi_m)
        save_dict[f"{prefix}_sentence"] = sent_arr
        save_dict[f"{prefix}_rep"] = rep_arr
        for k, v in other_results[mode].items():
            save_dict[f"{prefix}_{k}"] = v

    np.savez(out_file, **save_dict)
    print(f"\n[INFO] Saved results to: {out_file}")

    # 10) Optional quick envelope plot (train + each test)
    if make_plots:
        fig_rows = 1 + 1 + len(other_results)  # train + heldout + other modes
        fig, axes = plt.subplots(fig_rows, 1, figsize=(16, 3.0 * fig_rows), sharex=False)
        if fig_rows == 1:
            axes = [axes]

        r = 0
        axes[r].plot(train_results["envelope_true"][:500], label=f"True ({train_mode}-Train)", linewidth=2)
        axes[r].plot(train_results["envelope_pred"][:500], label="Recon", linewidth=2, linestyle="--")
        axes[r].set_title(f"{train_mode}-Train  env_corr={train_results['envelope_correlation']:.3f}")
        axes[r].legend(); axes[r].grid(True, alpha=0.3)

        r += 1
        axes[r].plot(trainmode_test_results["envelope_true"][:500], label=f"True ({train_mode}-Test)", linewidth=2)
        axes[r].plot(trainmode_test_results["envelope_pred"][:500], label="Recon", linewidth=2, linestyle="--")
        axes[r].set_title(f"{train_mode}-Test  env_corr={trainmode_test_results['envelope_correlation']:.3f}")
        axes[r].legend(); axes[r].grid(True, alpha=0.3)

        for mode, res in other_results.items():
            r += 1
            axes[r].plot(res["envelope_true"][:500], label=f"True ({mode})", linewidth=2)
            axes[r].plot(res["envelope_pred"][:500], label="Recon", linewidth=2, linestyle="--")
            axes[r].set_title(f"{mode}  env_corr={res['envelope_correlation']:.3f}")
            axes[r].legend(); axes[r].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, f"env_overview_train_{train_mode.lower()}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved envelope overview to: {fig_path}")


# ----------------------------
# Choose one of the 3 cases here
# ----------------------------
if __name__ == "__main__":
    # Case 1:
    run_cross_modality_experiment(train_mode="Vocalized", make_plots=True)

    # Case 2:
    # run_cross_modality_experiment(train_mode="Mimed", make_plots=True)

    # # Case 3:
    # run_cross_modality_experiment(train_mode="Imagined", make_plots=True)
