import os
import numpy as np
from scipy.io import savemat

BASE = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
OUT_MAT = os.path.join(BASE, "hist_data_real_vs_null_sentence_env_corr.mat")

TRAIN_MODES = ["vocalized", "mimed", "imagined"]
TEST_MODES  = ["vocalized", "mimed", "imagined"]

def safe_corr(a, b, eps=1e-12):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    if denom <= eps:
        return np.nan
    return float(np.dot(a, b) / denom)

def sentence_level_env_corr(y_true, y_pred, sentence_ids):
    """
    Returns:
      corr_dict: {sentence(str): corr(float)}
      (one correlation per sentence; envelope is mean across freq per window)
    """
    sentence_ids = np.asarray(sentence_ids, dtype=object)

    corr_dict = {}
    for sent in np.unique(sentence_ids):
        sent_str = str(sent)
        idx = (sentence_ids == sent)
        if idx.sum() < 2:
            continue
        yt = y_true[idx].mean(axis=1)  # envelope over windows
        yp = y_pred[idx].mean(axis=1)
        corr_dict[sent_str] = safe_corr(yt, yp)
    # drop NaNs
    corr_dict = {k: v for k, v in corr_dict.items() if np.isfinite(v)}
    return corr_dict

def load_npz_pair(train_mode):
    real_path = os.path.join(BASE, f"cross_modality_train_{train_mode}_with_heldout_test.npz")
    null_path = os.path.join(BASE, f"cross_modality_train_{train_mode}_with_heldout_test_null.npz")
    real = np.load(real_path, allow_pickle=True)
    null = np.load(null_path, allow_pickle=True)
    return real_path, null_path, real, null

# Nested dicts -> matlab-friendly later
DATA = {
    "train_modes": np.array(TRAIN_MODES, dtype=object),
    "test_modes":  np.array(TEST_MODES, dtype=object),
    "base_dir":    BASE,
}

# Store per train_mode/test_mode:
#   sentences: list of sentence strings (paired set)
#   r_real:    np.array (n_sent,)
#   r_null:    np.array (n_sent,)
#   delta:     np.array (n_sent,)
# plus some summary stats
for tr in TRAIN_MODES:
    real_path, null_path, npz_real, npz_null = load_npz_pair(tr)

    DATA[f"{tr}_real_npz_path"] = real_path
    DATA[f"{tr}_null_npz_path"] = null_path

    for te in TEST_MODES:
        # --- REAL ---
        yT_r = np.asarray(npz_real[f"{te}_y_test"])
        yP_r = np.asarray(npz_real[f"{te}_y_test_pred"])
        s_r  = np.asarray(npz_real[f"{te}_sentence"], dtype=object)
        corr_real = sentence_level_env_corr(yT_r, yP_r, s_r)

        # --- NULL ---
        yT_n = np.asarray(npz_null[f"{te}_y_test"])
        yP_n = np.asarray(npz_null[f"{te}_y_test_pred"])
        s_n  = np.asarray(npz_null[f"{te}_sentence"], dtype=object)
        corr_null = sentence_level_env_corr(yT_n, yP_n, s_n)

        # Pair by sentence intersection (so Δ is well-defined)
        common = sorted(set(corr_real.keys()).intersection(set(corr_null.keys())))
        r_real = np.array([corr_real[s] for s in common], dtype=float)
        r_null = np.array([corr_null[s] for s in common], dtype=float)
        delta  = r_real - r_null

        key = f"train_{tr}__test_{te}"
        DATA[f"{key}__sentences"] = np.array(common, dtype=object)
        DATA[f"{key}__r_real"]    = r_real
        DATA[f"{key}__r_null"]    = r_null
        DATA[f"{key}__delta"]     = delta

        # Helpful scalars for annotations
        DATA[f"{key}__n"]          = int(len(common))
        DATA[f"{key}__mean_real"]  = float(np.nanmean(r_real)) if r_real.size else np.nan
        DATA[f"{key}__mean_null"]  = float(np.nanmean(r_null)) if r_null.size else np.nan
        DATA[f"{key}__mean_delta"] = float(np.nanmean(delta))  if delta.size else np.nan
        DATA[f"{key}__pct_delta_ge0"] = float(100.0 * np.mean(delta >= 0.0)) if delta.size else np.nan

print("[INFO] Saving:", OUT_MAT)
savemat(OUT_MAT, DATA, do_compression=True)
print("[OK] Wrote .mat with histogram inputs:", OUT_MAT)