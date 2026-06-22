#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
OUT  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/lin_ridge_across_modes/outputs_with_null"
os.makedirs(OUT, exist_ok=True)

TRAIN_MODES = ["vocalized", "mimed", "imagined"]
TEST_MODES  = ["vocalized", "mimed", "imagined"]

# File template (real + null)
REAL_FMT = os.path.join(BASE, "cross_modality_train_{train}_with_heldout_test.npz")
NULL_FMT = os.path.join(BASE, "cross_modality_train_{train}_with_heldout_test_null.npz")

# Plot style
COLORS = {
    "vocalized": "#1f77b4",  # blue
    "mimed":     "#2ca02c",  # green
    "imagined":  "#ff7f0e",  # orange
}
BINS = 20
XLIM = (-0.6, 0.95)   # adjust if needed

def sentence_level_corr_dict(y_true, y_pred, sentence_ids):
    """
    Returns dict: sentence_id -> corr(envelope_true, envelope_pred) over windows for that sentence.
    Envelope = mean over frequency bins per window.
    """
    out = {}
    sentence_ids = np.asarray(sentence_ids)

    for sent in np.unique(sentence_ids):
        idx = sentence_ids == sent
        if idx.sum() < 2:
            continue

        yt = y_true[idx].mean(axis=1)
        yp = y_pred[idx].mean(axis=1)

        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            continue

        out[str(sent)] = float(np.corrcoef(yt, yp)[0, 1])

    return out

def paired_sentence_arrays(npz_real, npz_null, test_mode):
    """
    Build paired arrays real/null by sentence intersection.
    Returns: common_sents, real_vals, null_vals, delta
    """
    yT_r = npz_real[f"{test_mode}_y_test"]
    yP_r = npz_real[f"{test_mode}_y_test_pred"]
    s_r  = npz_real[f"{test_mode}_sentence"]

    yT_n = npz_null[f"{test_mode}_y_test"]
    yP_n = npz_null[f"{test_mode}_y_test_pred"]
    s_n  = npz_null[f"{test_mode}_sentence"]

    dr = sentence_level_corr_dict(yT_r, yP_r, s_r)
    dn = sentence_level_corr_dict(yT_n, yP_n, s_n)

    common = sorted(set(dr.keys()) & set(dn.keys()))
    real_vals = np.array([dr[s] for s in common], dtype=float)
    null_vals = np.array([dn[s] for s in common], dtype=float)
    delta = real_vals - null_vals
    return common, real_vals, null_vals, delta

def vline(ax, x, color, ls="--", lw=2):
    ax.axvline(x, color=color, linestyle=ls, linewidth=lw, alpha=0.95)

def setup_pub(ax):
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig, outbase, dpi=300):
    fig.savefig(outbase + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", dpi=dpi, bbox_inches="tight")

def main():
    # Collect summary rows
    summary = []
    summary.append("train_mode\ttest_mode\tn_sent\tmean_real\tmean_null\tmean_delta\tfrac_delta_pos\n")

    # =========================
    # FIG 1: Real vs Null overlay
    # =========================
    fig1, axes1 = plt.subplots(
        nrows=len(TEST_MODES),
        ncols=len(TRAIN_MODES),
        figsize=(16, 10),
        sharex=True,
        sharey=False,
        constrained_layout=True
    )

    # =========================
    # FIG 2: Delta histograms
    # =========================
    fig2, axes2 = plt.subplots(
        nrows=len(TEST_MODES),
        ncols=len(TRAIN_MODES),
        figsize=(16, 10),
        sharex=True,
        sharey=False,
        constrained_layout=True
    )

    for c, train_mode in enumerate(TRAIN_MODES):
        real_path = REAL_FMT.format(train=train_mode)
        null_path = NULL_FMT.format(train=train_mode)

        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Missing real file: {real_path}")
        if not os.path.exists(null_path):
            raise FileNotFoundError(f"Missing null file: {null_path}")

        npz_real = np.load(real_path, allow_pickle=True)
        npz_null = np.load(null_path, allow_pickle=True)

        for r, test_mode in enumerate(TEST_MODES):
            ax1 = axes1[r, c]
            ax2 = axes2[r, c]

            common, real_vals, null_vals, delta = paired_sentence_arrays(npz_real, npz_null, test_mode)
            n_sent = len(common)

            # Summary stats
            m_real = float(np.nanmean(real_vals)) if n_sent else np.nan
            m_null = float(np.nanmean(null_vals)) if n_sent else np.nan
            m_del  = float(np.nanmean(delta)) if n_sent else np.nan
            frac_pos = float(np.mean(delta > 0)) if n_sent else np.nan

            summary.append(
                f"{train_mode}\t{test_mode}\t{n_sent}\t{m_real:.6f}\t{m_null:.6f}\t{m_del:.6f}\t{frac_pos:.6f}\n"
            )

            # -------- FIG1: overlay real vs null
            color = COLORS[test_mode]
            ax1.hist(null_vals, bins=BINS, density=True, alpha=0.30, color=color, label="Null")
            ax1.hist(real_vals, bins=BINS, density=True, alpha=0.55, color=color, label="Real")

            if n_sent:
                vline(ax1, m_null, color=color, ls=":", lw=2.2)
                vline(ax1, m_real, color=color, ls="--", lw=2.2)

            setup_pub(ax1)
            ax1.set_xlim(XLIM)

            # Titles / labels
            if r == 0:
                ax1.set_title(f"Train on {train_mode.capitalize()}", fontsize=14, pad=10)
            if c == 0:
                ax1.set_ylabel(f"Test: {test_mode.capitalize()}\nDensity", fontsize=12)

            # Annotation (top-right)
            ax1.text(
                0.98, 0.95,
                f"n={n_sent}\nΔmean={m_del:.3f}\n%Δ>0={100*frac_pos:.1f}%",
                transform=ax1.transAxes,
                ha="right", va="top",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none")
            )

            if r == len(TEST_MODES) - 1:
                ax1.set_xlabel("Sentence-level envelope corr (r)", fontsize=12)

            # -------- FIG2: delta histogram
            ax2.hist(delta, bins=BINS, density=True, alpha=0.70, color=color)
            if n_sent:
                vline(ax2, m_del, color=color, ls="--", lw=2.2)
                vline(ax2, 0.0, color="k", ls=":", lw=2.0)

            setup_pub(ax2)
            ax2.set_xlim(-0.8, 0.8)

            if r == 0:
                ax2.set_title(f"Train on {train_mode.capitalize()}", fontsize=14, pad=10)
            if c == 0:
                ax2.set_ylabel(f"Test: {test_mode.capitalize()}\nDensity", fontsize=12)

            ax2.text(
                0.98, 0.95,
                f"n={n_sent}\nmeanΔ={m_del:.3f}\n%Δ>0={100*frac_pos:.1f}%",
                transform=ax2.transAxes,
                ha="right", va="top",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none")
            )

            if r == len(TEST_MODES) - 1:
                ax2.set_xlabel("Δr = Real − Null", fontsize=12)

    # Legends (one shared)
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    if handles:
        fig1.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, fontsize=12)

    fig1.suptitle("Real vs Null: Sentence-level envelope correlation (paired by sentence)", fontsize=16, y=1.02)
    fig2.suptitle("Real − Null: Sentence-level envelope correlation improvement", fontsize=16, y=1.02)

    save_fig(fig1, os.path.join(OUT, "fig_real_vs_null_sentence_envcorr_3x3_overlay"))
    save_fig(fig2, os.path.join(OUT, "fig_real_minus_null_sentence_envcorr_3x3_delta"))

    # Save summary TSV
    tsv_path = os.path.join(OUT, "summary_real_vs_null_sentence_envcorr.tsv")
    with open(tsv_path, "w") as f:
        f.writelines(summary)

    print("Saved:")
    print(" ", os.path.join(OUT, "fig_real_vs_null_sentence_envcorr_3x3_overlay.[png/pdf]"))
    print(" ", os.path.join(OUT, "fig_real_minus_null_sentence_envcorr_3x3_delta.[png/pdf]"))
    print(" ", tsv_path)

if __name__ == "__main__":
    main()