import numpy as np
import matplotlib.pyplot as plt
import os

BASE = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
OUT = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/lin_ridge_across_modes/outputs_noncausal"
# FILES = {
#     "Vocalized": os.path.join(BASE, "cross_modality_train_vocalized_with_heldout_test.npz"),
#     "Mimed":     os.path.join(BASE, "cross_modality_train_mimed_with_heldout_test.npz"),
#     "Imagined":  os.path.join(BASE, "cross_modality_train_imagined_with_heldout_test.npz"),
# }
FILES = {
    "Vocalized": os.path.join(BASE, "cross_modality_centered_p300_f300_train_vocalized_with_heldout_test.npz"),
    "Mimed":     os.path.join(BASE, "cross_modality_centered_p300_f300_train_mimed_with_heldout_test.npz"),
    "Imagined":  os.path.join(BASE, "cross_modality_centered_p300_f300_train_imagined_with_heldout_test.npz"),
}

MODES = ["vocalized", "mimed", "imagined"]

COLORS = {
    "vocalized": "#1f77b4",
    "mimed":     "#ff7f0e",
    "imagined":  "#2ca02c",
}

ALPHA = 0.45
BINS = 40
XLIM = (-0.2, 0.8)   # adjust after first plot if needed


def load_corr(npz, mode):
    """Return per-frequency correlation array, dropping NaNs."""
    x = npz[f"{mode}_correlations"]
    return x[np.isfinite(x)]


#### per sentence ####
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"

TRAIN_MODES = ["vocalized", "mimed", "imagined"]
TEST_MODES  = ["vocalized", "mimed", "imagined"]

COLORS = {
    "vocalized": "#1f77b4",
    "mimed":     "#ff7f0e",
    "imagined":  "#2ca02c",
}

def sentence_level_corr(y_true, y_pred, sentence_ids):
    """
    One correlation per sentence:
      - average over frequency → 1D time-series (over windows)
      - corr(true, pred) per sentence
    """
    corr_dict = {}
    sentence_ids = np.asarray(sentence_ids)

    for sent in np.unique(sentence_ids):
        idx = sentence_ids == sent
        if idx.sum() < 2:
            continue

        # average over frequency -> (n_windows_for_sentence,)
        yt = y_true[idx].mean(axis=1)
        yp = y_pred[idx].mean(axis=1)

        # guard against zero-variance
        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            corr = np.nan
        else:
            corr = np.corrcoef(yt, yp)[0, 1]

        corr_dict[sent] = corr

    # return just values (drop nans)
    vals = np.array(list(corr_dict.values()), dtype=float)
    vals = vals[np.isfinite(vals)]
    return vals


##### per sentence plot #####
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

for ax, train_mode in zip(axes, TRAIN_MODES):
    npz_path = os.path.join(BASE, f"cross_modality_train_{train_mode}_with_heldout_test.npz")
    npz = np.load(npz_path, allow_pickle=True)

    for test_mode in TEST_MODES:
        y_true   = npz[f"{test_mode}_y_test"]
        y_pred   = npz[f"{test_mode}_y_test_pred"]
        sent_ids = npz[f"{test_mode}_sentence"]

        vals = sentence_level_corr(y_true, y_pred, sent_ids)

        ax.hist(
            vals, bins=20, density=True, alpha=0.5,
            label=test_mode.capitalize(),
            color=COLORS[test_mode]
        )
        ax.axvline(vals.mean(), linestyle="--", color=COLORS[test_mode], linewidth=2)

    ax.set_title(f"Train on {train_mode.capitalize()}")
    ax.set_xlabel("Sentence-level envelope correlation")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, fontsize=11)
plt.suptitle("Sentence-level Reconstruction: Within vs Cross Modality", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = os.path.join(OUT, "hist_cross_modality_per_sentence_correlations_3panel.png")
plt.savefig(out_path, dpi=200)

print("Saved:", out_path)


##### per frequency #####
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, (train_mode, fname) in zip(axes, FILES.items()):
    npz = np.load(fname, allow_pickle=True)

    for test_mode in MODES:
        corrs = load_corr(npz, test_mode)

        ax.hist(
            corrs,
            bins=BINS,
            density=True,
            alpha=ALPHA,
            color=COLORS[test_mode],
            label=test_mode.capitalize(),
        )

        ax.axvline(
            np.mean(corrs),
            color=COLORS[test_mode],
            linestyle="--",
            linewidth=2,
        )

    ax.set_title(f"Train on {train_mode}", fontsize=13)
    ax.set_xlabel("Reconstruction correlation (r)")
    ax.set_xlim(XLIM)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Density")
axes[0].legend(frameon=False, fontsize=11)

plt.suptitle(
    "Distribution of Spectrogram Reconstruction Correlations\n"
    "Within- vs Cross-Modality Generalization",
    fontsize=15,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(OUT, "hist_cross_modality_correlations.png"), dpi=200)
