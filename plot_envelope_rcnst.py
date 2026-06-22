#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
NPZ_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_results.npz"
OUT_DIR  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"

N_SENTENCES = 5
N_WINDOWS_SHOW = 250
START_AT = 0

DPI = 300
# --------------------------------------------

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
    return y[idx]  # (T, F)

def envelope_raw(Y: np.ndarray) -> np.ndarray:
    """Raw envelope: mean across frequency bins per window."""
    return np.mean(Y, axis=1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(NPZ_PATH, allow_pickle=True)

    # Arrays
    y_true_train = np.array(d["y_train"])
    y_pred_train = np.array(d["y_train_pred"])
    sent_train   = np.array(d["train_sentence"], dtype=object)

    y_mimed_pred = np.array(d["mimed_y_test_pred"])
    sent_mimed   = np.array(d["mimed_sentence"], dtype=object)

    y_imag_pred  = np.array(d["imagined_y_test_pred"])
    sent_imag    = np.array(d["imagined_sentence"], dtype=object)

    # Choose sentences
    unique_sentences = np.unique(sent_train)
    example_sentences = unique_sentences[:N_SENTENCES]
    print(f"[INFO] Plotting {len(example_sentences)} sentences as rows (raw envelopes)")

    # Figure: N rows × 1 column
    fig, axes = plt.subplots(
        len(example_sentences), 1,
        figsize=(7.5, 2.2 * len(example_sentences)),
        sharex=True
    )
    if len(example_sentences) == 1:
        axes = [axes]

    x = np.arange(N_WINDOWS_SHOW)

    for i, sentence in enumerate(example_sentences):
        ax = axes[i]

        Y_true = get_sentence_block(y_true_train, sent_train, sentence, N_WINDOWS_SHOW, START_AT)
        Y_voc  = get_sentence_block(y_pred_train, sent_train, sentence, N_WINDOWS_SHOW, START_AT)
        Y_mim  = get_sentence_block(y_mimed_pred, sent_mimed, sentence, N_WINDOWS_SHOW, START_AT)
        Y_img  = get_sentence_block(y_imag_pred,  sent_imag,  sentence, N_WINDOWS_SHOW, START_AT)

        if (Y_true is None) or (Y_voc is None) or (Y_mim is None) or (Y_img is None):
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        e_true = envelope_raw(Y_true)
        e_voc  = envelope_raw(Y_voc)
        e_mim  = envelope_raw(Y_mim)
        e_img  = envelope_raw(Y_img)

        # Plot: true thicker
        e_true = e_true - np.mean(e_true)
        e_voc  = e_voc  - np.mean(e_voc)
        e_mim  = e_mim  - np.mean(e_mim)
        e_img  = e_img  - np.mean(e_img)    
        e_true = e_true / (np.std(e_true) + 1e-12)
        e_voc  = e_voc  / (np.std(e_voc) + 1e-12)
        e_mim  = e_mim  / (np.std(e_mim) + 1e-12)    
        e_img  = e_img  / (np.std(e_img) + 1e-12)
        ax.plot(x, e_true, label="True", linewidth=2.5)
        ax.plot(x, e_voc,  label="Voc→Voc", linewidth=1.4)
        ax.plot(x, e_mim,  label="Voc→Mimed", linewidth=1.4)
        ax.plot(x, e_img,  label="Voc→Imagined", linewidth=1.4)

        ax.set_title(str(sentence), fontsize=10)
        ax.set_ylabel("Envelope")

        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (windows)")

    # Single legend for entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle(
        f"Raw spectrogram envelopes (windows {START_AT}:{START_AT + N_WINDOWS_SHOW})",
        y=1.06, fontsize=12
    )

    plt.tight_layout()

    out_base = os.path.join(
        OUT_DIR,
        f"fig_envelopes_raw_rows_zscore_{N_SENTENCES}sent_win{N_WINDOWS_SHOW}"
    )
    fig.savefig(out_base + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Saved:", out_base + ".png/.pdf")

if __name__ == "__main__":
    main()
