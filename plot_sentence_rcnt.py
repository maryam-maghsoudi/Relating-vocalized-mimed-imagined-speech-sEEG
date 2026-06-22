#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pdb

# ---------------- USER CONFIG ----------------
NPZ_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_results.npz"
OUT_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"

N_SENTENCES = 5
N_WINDOWS_SHOW = 250
START_AT = 0

CMAP = "magma"
DPI = 300

APPLY_SQRT_DISPLAY = False   # purely for visualization (does NOT change metrics)
EPS_SQRT = 1e-12            # keeps sqrt stable if tiny negatives exist
# --------------------------------------------
import matplotlib.colors as colors

SCALE_MODE = "column"   # "panel" (like your first fig) OR "column" OR "global"
CLIP_PCT = (2, 98)     # percentile clip helps True not dominate
USE_POWER_NORM = True  # better than sqrt(Y) for display; set False if you prefer raw

def get_vmin_vmax(A, pct=(2,98)):
    lo, hi = np.percentile(A, pct)
    return float(lo), float(hi)

def get_norm(vmin, vmax):
    if not USE_POWER_NORM:
        return None
    # gamma<1 boosts low values (like sqrt), but without modifying data
    return colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

def safe_fname(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:120]

def get_sentence_block(y: np.ndarray, sent_labels: np.ndarray, sentence: str,
                       n_show: int, start_at: int = 0):
    idx = np.where(sent_labels == sentence)[0]
    if len(idx) == 0:
        return None
    start_at = int(max(0, start_at))
    if start_at >= len(idx):
        start_at = 0
    idx = idx[start_at:start_at + n_show]
    if len(idx) == 0:
        return None
    return y[idx]

def safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)

def mean_freq_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    F = y_true.shape[1]
    rs = []
    for f in range(F):
        r = safe_corr(y_true[:, f], y_pred[:, f])
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if len(rs) else float("nan")

def sqrt_display(Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Apply a safe sqrt for visualization.
    If values are nonnegative (expected), this is just sqrt(Y).
    If tiny negatives exist due to numerics, clip them to 0.
    """
    Y = np.asarray(Y)
    return np.sqrt(np.clip(Y, 0.0, None) + eps)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(NPZ_PATH, allow_pickle=True)

    # Arrays
    y_true_train = np.array(d["y_train"])
    y_pred_train = np.array(d["y_train_pred"])
    sent_train   = np.array(d["train_sentence"], dtype=object)

    y_mimed_true = np.array(d["mimed_y_test"])
    y_mimed_pred = np.array(d["mimed_y_test_pred"])
    sent_mimed   = np.array(d["mimed_sentence"], dtype=object)

    y_imag_true  = np.array(d["imagined_y_test"])
    y_imag_pred  = np.array(d["imagined_y_test_pred"])
    sent_imag    = np.array(d["imagined_sentence"], dtype=object)

    # Choose sentences
    unique_sentences = np.unique(sent_train)
    example_sentences = unique_sentences[:N_SENTENCES]
    print(f"[INFO] Plotting {len(example_sentences)} sentences as columns")

    # Prepare grid: 4 rows x N_SENTENCES columns
    fig, axes = plt.subplots(
        4, len(example_sentences),
        figsize=(3.2 * len(example_sentences), 10),
        sharex=True, sharey=True
    )
    if len(example_sentences) == 1:
        axes = axes.reshape(4, 1)

    row_labels = ["True", "Voc→Voc", "Voc→Mimed", "Voc→Imagined"]

    last_im = None

    for col, sentence in enumerate(example_sentences):
        # blocks
        Y_true = get_sentence_block(y_true_train, sent_train, sentence, N_WINDOWS_SHOW, START_AT)
        Y_voc  = get_sentence_block(y_pred_train, sent_train, sentence, N_WINDOWS_SHOW, START_AT)

        Y_mim_true = get_sentence_block(y_mimed_true, sent_mimed, sentence, N_WINDOWS_SHOW, START_AT)
        Y_mim_pred = get_sentence_block(y_mimed_pred, sent_mimed, sentence, N_WINDOWS_SHOW, START_AT)

        Y_img_true = get_sentence_block(y_imag_true, sent_imag, sentence, N_WINDOWS_SHOW, START_AT)
        Y_img_pred = get_sentence_block(y_imag_pred, sent_imag, sentence, N_WINDOWS_SHOW, START_AT)

        if (Y_true is None) or (Y_voc is None) or (Y_mim_pred is None) or (Y_img_pred is None):
            print(f"[WARN] Missing data for sentence={sentence}. Skipping column.")
            # blank out column
            for r in range(4):
                axes[r, col].axis("off")
            continue

        # correlations (computed on original scale, not sqrt-display)
        r_voc = mean_freq_corr(Y_true, Y_voc)
        r_mim = mean_freq_corr(Y_mim_true, Y_mim_pred) if Y_mim_true is not None else np.nan
        r_img = mean_freq_corr(Y_img_true, Y_img_pred) if Y_img_true is not None else np.nan

        panels = [Y_true, Y_voc, Y_mim_pred, Y_img_pred]

        # optional sqrt for visualization only
        if APPLY_SQRT_DISPLAY:
            panels_plot = [sqrt_display(P, EPS_SQRT) for P in panels]
        else:
            panels_plot = panels

        # flatten helpers for clipping
        flat_all = np.concatenate([P.ravel() for P in panels_plot])

        if SCALE_MODE == "panel":
            # each subplot gets its own vmin/vmax (matches your first figure vibe)
            vm = [get_vmin_vmax(P, CLIP_PCT) for P in panels_plot]

        elif SCALE_MODE == "column":
            # one vmin/vmax for all 4 rows of this sentence
            vmin, vmax = get_vmin_vmax(flat_all, CLIP_PCT)
            vm = [(vmin, vmax)] * 4

        # elif SCALE_MODE == "global":
        #     # you need to precompute this outside the sentence loop
        #     # (see below)
        #     vm = [(GV_MIN, GV_MAX)] * 4

        else:
            raise ValueError("Bad SCALE_MODE")

        titles = [
            "True",
            f"Voc→Voc\nr={r_voc:.2f}",
            f"Voc→Mimed\nr={r_mim:.2f}",
            f"Voc→Imag\nr={r_img:.2f}",
        ]

        for row in range(4):
            ax = axes[row, col]
            vmin, vmax = vm[row]
            norm = get_norm(vmin, vmax)

            last_im = ax.imshow(
                panels_plot[row].T,
                aspect="auto", origin="lower", interpolation="none",
                cmap=CMAP,
                # vmin=vmin, vmax=vmax,
                norm=norm,
            )

            # column title only on top row: sentence name
            if row == 0:
                ax.set_title(f"{sentence}", fontsize=11)

            # row labels on first column
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nFreq bin", fontsize=10)

            # small label per panel (top-left)
            ax.text(
                0.01, 0.98, titles[row],
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2)
            )

            if row == 3:
                ax.set_xlabel("Time (windows)")

    # One colorbar for the whole figure (uses last_im)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("sqrt(amplitude)" if APPLY_SQRT_DISPLAY else "amplitude",
                       rotation=270, labelpad=15)

    fig.suptitle(
        f"Spectrogram reconstructions per sentence (windows {START_AT}:{START_AT+N_WINDOWS_SHOW})",
        y=1.01, fontsize=14
    )
    plt.tight_layout()

    out_base = os.path.join(OUT_DIR, f"fig_columns_{N_SENTENCES}sent_win{N_WINDOWS_SHOW}_sqrt{int(APPLY_SQRT_DISPLAY)}")
    fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved:", out_base + ".png/.pdf")

if __name__ == "__main__":
    main()
