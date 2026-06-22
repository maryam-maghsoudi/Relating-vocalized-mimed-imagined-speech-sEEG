#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
NPZ_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_results.npz"
OUT_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"

N_BINS = 25
DPI = 300
OVERLAY_GAUSSIAN = True
TITLE = "Per-sentence envelope correlation distributions (all windows)"
# --------------------------------------------

COLORS = {
    "Voc→Voc": "#1f77b4",      # blue
    "Voc→Mimed": "#2ca02c",    # green
    "Voc→Imagined": "#d62728", # red
}

def safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)

def envelope_raw(Y: np.ndarray) -> np.ndarray:
    """Y: (T, F) -> env: (T,)"""
    return np.mean(Y, axis=1)

def per_sentence_env_corr_all_windows(y_true: np.ndarray, y_pred: np.ndarray, sent: np.ndarray):
    """
    One r per unique sentence, using ALL windows for that sentence.
    """
    out = {}
    for s in np.unique(sent):
        idx = np.where(sent == s)[0]
        if idx.size < 2:
            continue
        Yt = y_true[idx]
        Yp = y_pred[idx]
        et = envelope_raw(Yt)
        ep = envelope_raw(Yp)
        r = safe_corr(et, ep)
        if not np.isnan(r):
            out[str(s)] = r
    return out

def plot_overlay_hist(rs_dict, out_base):
    plt.figure(figsize=(7.2, 4.2))

    all_vals = np.concatenate([v for v in rs_dict.values() if len(v) > 0])
    if all_vals.size == 0:
        raise RuntimeError("No correlation values found to plot.")

    rmin = max(-1.0, float(np.min(all_vals)) - 0.05)
    rmax = min( 1.0, float(np.max(all_vals)) + 0.05)
    bins = np.linspace(rmin, rmax, N_BINS + 1)
    xs = np.linspace(rmin, rmax, 400)

    for name, vals in rs_dict.items():
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            continue

        c = COLORS.get(name, None)

        # histogram
        plt.hist(vals, bins=bins, density=True, alpha=0.30, color=c,
                 label=f"{name} (n={vals.size})")

        # gaussian + mean line (same color)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

        if OVERLAY_GAUSSIAN and sd > 0:
            pdf = (1.0 / (sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
            plt.plot(xs, pdf, linewidth=2.0, color=c)

        plt.axvline(mu, linestyle="--", linewidth=1.8, color=c)

    plt.xlabel("Envelope correlation r (per sentence)")
    plt.ylabel("Density")
    plt.title(TITLE)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    plt.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight")
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(NPZ_PATH, allow_pickle=True)

    # Vocalized
    y_true_train = np.array(d["y_train"])
    y_pred_train = np.array(d["y_train_pred"])
    sent_train   = np.array(d["train_sentence"], dtype=object)

    # Mimed
    y_true_mimed = np.array(d["mimed_y_test"])
    y_pred_mimed = np.array(d["mimed_y_test_pred"])
    sent_mimed   = np.array(d["mimed_sentence"], dtype=object)

    # Imagined
    y_true_imag = np.array(d["imagined_y_test"])
    y_pred_imag = np.array(d["imagined_y_test_pred"])
    sent_imag   = np.array(d["imagined_sentence"], dtype=object)

    print("[INFO] Unique sentences:")
    print("  vocalized:", np.unique(sent_train).size)
    print("  mimed:    ", np.unique(sent_mimed).size)
    print("  imagined: ", np.unique(sent_imag).size)

    r_voc = per_sentence_env_corr_all_windows(y_true_train, y_pred_train, sent_train)
    r_mim = per_sentence_env_corr_all_windows(y_true_mimed, y_pred_mimed, sent_mimed)
    r_img = per_sentence_env_corr_all_windows(y_true_imag,  y_pred_imag,  sent_imag)

    r_voc_vals = np.array(list(r_voc.values()), dtype=float)
    r_mim_vals = np.array(list(r_mim.values()), dtype=float)
    r_img_vals = np.array(list(r_img.values()), dtype=float)

    print("[INFO] Per-sentence summary (all windows):")
    for name, arr in [("Voc→Voc", r_voc_vals), ("Voc→Mimed", r_mim_vals), ("Voc→Imagined", r_img_vals)]:
        if arr.size:
            print(f"  {name:12s} n={arr.size:3d}  mean={arr.mean():.3f}  std={arr.std(ddof=1):.3f}")
        else:
            print(f"  {name:12s} n=0")

    rs_dict = {
        "Voc→Voc": r_voc_vals,
        "Voc→Mimed": r_mim_vals,
        "Voc→Imagined": r_img_vals,
    }

    out_base = os.path.join(OUT_DIR, f"fig_envcorr_overlay_allwindows_bins{N_BINS}")
    plot_overlay_hist(rs_dict, out_base)
    print("[OK] Saved:", out_base + ".png/.pdf")

if __name__ == "__main__":
    main()
