#!/usr/bin/env python3
"""
Analyze cross_modality_results.npz:
- Per-sentence spectrogram correlation (mean over frequencies)
- Per-sentence envelope correlation
- Bootstrap CIs
- Publication-ready figures (PNG + PDF)
- Channel importance from decoder weights (fast)
- Optional top-K channel ablation (slower, more direct)

Usage:
  python analyze_cross_modality_per_sentence.py \
    --npz /fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_results.npz \
    --outdir /fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/figs_sentence_corr \
    --bootstrap 500 \
    --topk 20 \
    --do_ablation 0

If you want ablation:
  python analyze_cross_modality_per_sentence.py ... --do_ablation 1 --ablation_topk 10 --ablation_max_windows 8000
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb
import numpy as np
import textwrap

p="/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_results_with_vcl_test.npz"
d=np.load(p, allow_pickle=True)
print("Loaded:", p)
print("Has X_train?", "X_train" in d.files)
print("Has decoder_coef?", "decoder_coef" in d.files)
print("First 40 keys:", d.files[:40])

# -----------------------------
# Metrics
# -----------------------------
def safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson corr for 1D arrays; returns nan if degenerate."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size or x.size < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)


def mean_freq_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    y_true/y_pred: (N, F)
    Return: mean over frequency bins of corr across windows.
    """
    F = y_true.shape[1]
    corrs = []
    for f in range(F):
        r = safe_corr(y_true[:, f], y_pred[:, f])
        if not np.isnan(r):
            corrs.append(r)
    return float(np.mean(corrs)) if len(corrs) else np.nan


def envelope_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Envelope = mean across frequency bins per window, then correlate over windows.
    """
    env_t = np.mean(y_true, axis=1)
    env_p = np.mean(y_pred, axis=1)
    return safe_corr(env_t, env_p)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 500, seed: int = 0):
    """
    Bootstrap CI over windows (rows).
    Returns: (point_est, lo, hi)
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    if N < 5:
        val = metric_fn(y_true, y_pred)
        return val, np.nan, np.nan

    point = metric_fn(y_true, y_pred)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = metric_fn(y_true[idx], y_pred[idx])

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# -----------------------------
# Data loading helpers
# -----------------------------
def load_modality(npz, prefix: str):
    """
    prefix in {"mimed", "imagined"}.
    Expects keys:
      f"{prefix}_y_test", f"{prefix}_y_test_pred", f"{prefix}_sentence"
    Returns dict or None.
    """
    k_y = f"{prefix}_y_test"
    k_p = f"{prefix}_y_test_pred"
    k_s = f"{prefix}_sentence"

    if k_y not in npz.files or k_p not in npz.files or k_s not in npz.files:
        return None

    y_true = np.array(npz[k_y])
    y_pred = np.array(npz[k_p])
    sent   = np.array(npz[k_s], dtype=object)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(f"{prefix}: y arrays must be (N,F). Got {y_true.shape} and {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{prefix}: y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if sent.shape[0] != y_true.shape[0]:
        raise ValueError(f"{prefix}: sentence length mismatch: {sent.shape[0]} vs {y_true.shape[0]}")

    return {"y_true": y_true, "y_pred": y_pred, "sentence": sent}


def group_indices_by_sentence(sentence_arr: np.ndarray):
    """
    sentence_arr: (N,) object strings
    Returns: dict sentence -> np.ndarray indices
    """
    groups = {}
    for i, s in enumerate(sentence_arr):
        groups.setdefault(str(s), []).append(i)
    for s in list(groups.keys()):
        groups[s] = np.array(groups[s], dtype=int)
    return groups


# -----------------------------
# Channel importance
# -----------------------------
def channel_importance_from_weights(decoder_coef: np.ndarray, window_frames: int, n_channels: int, mode="abs_mean"):
    """
    decoder_coef: (F, D) where D = window_frames * n_channels
    reshape -> (F, window_frames, n_channels)
    importance:
      - abs_mean: mean(|W|) over (F, T)
      - l2: sqrt(mean(W^2)) over (F, T)
    """
    F, D = decoder_coef.shape
    expected = window_frames * n_channels
    if D != expected:
        raise ValueError(f"decoder_coef has D={D}, expected {expected} (= {window_frames}*{n_channels})")

    W = decoder_coef.reshape(F, window_frames, n_channels)

    if mode == "abs_mean":
        imp = np.mean(np.abs(W), axis=(0, 1))
    elif mode == "l2":
        imp = np.sqrt(np.mean(W * W, axis=(0, 1)))
    else:
        raise ValueError(f"Unknown importance mode: {mode}")

    return imp  # (n_channels,)


# -----------------------------
# Optional ablation (top-K only)
# -----------------------------
def ablation_delta_r_topk(X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                          sentence_arr: np.ndarray, channels: np.ndarray,
                          max_windows: int = 8000, seed: int = 0):
    """
    Very lightweight “proxy” ablation:
      We cannot re-run the model without re-applying the linear decoder here.
      But since you saved decoder weights, we *can* re-run predictions with channels zeroed.

    Requires:
      - decoder_coef/intercept + x_scaler_mean/scale + y_scaler_mean/scale
    So we do ablation properly in main() when do_ablation=1.

    This function only computes bookkeeping / grouping.
    """
    raise NotImplementedError("Ablation is implemented in main() because it needs the decoder/scalers.")


def apply_linear_decoder(X_windows: np.ndarray,
                         decoder_coef: np.ndarray, decoder_intercept: np.ndarray,
                         x_mean: np.ndarray, x_scale: np.ndarray,
                         y_mean: np.ndarray, y_scale: np.ndarray):
    """
    Recompute y_pred given X_windows and saved scalers/decoder params.
    X_windows: (N, T, C)
    Steps:
      - Flatten X -> (N, D)
      - Standardize: (X - x_mean) / x_scale
      - Ridge: y_norm = Xz @ coef.T + intercept  (coef shape (F, D))
      - Denorm: y = y_norm * y_scale + y_mean
    Returns:
      y_pred: (N, F)
    """
    N, T, C = X_windows.shape
    D = T * C
    Xf = X_windows.reshape(N, D)

    # standardize
    Xz = (Xf - x_mean[None, :]) / x_scale[None, :]

    # predict normalized spectrogram
    y_norm = Xz @ decoder_coef.T + decoder_intercept[None, :]

    # denormalize
    y = y_norm * y_scale[None, :] + y_mean[None, :]
    return y


# -----------------------------
# Plot helpers (pub-ready)
# -----------------------------
def setup_pub_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, outbase: str, dpi: int = 300):
    fig.savefig(outbase + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(outbase + ".pdf", dpi=dpi, bbox_inches="tight")

def setup_pub_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def wrap_labels(labels, width=18):
    """Wrap long sentence labels so they don't collide."""
    out = []
    for s in labels:
        s = str(s)
        out.append("\n".join(textwrap.wrap(s, width=width, break_long_words=False)))
    return out


def plot_per_sentence_with_ci(
    outbase: str,
    title: str,
    ylabel: str,
    sent_order,
    series_dict,
    rotate=70,
    wrap_width=18,
    dodge=0.25,
):
    """
    series_dict: dict name -> dict with keys:
      "r", "lo", "hi"  (each np.ndarray of shape (n_sent,))
    Produces a publication-ready figure with dodged errorbars.
    """
    n = len(sent_order)
    x = np.arange(n)

    # Wider figure to give labels room; also leave more bottom margin.
    fig, ax = plt.subplots(figsize=(max(12, 0.55 * n), 5.0), constrained_layout=False)

    # deterministic offsets (so modalities are consistently placed)
    names = list(series_dict.keys())
    k = len(names)
    if k == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-dodge, dodge, k)

    for off, name in zip(offsets, names):
        r  = series_dict[name]["r"]
        lo = series_dict[name]["lo"]
        hi = series_dict[name]["hi"]

        yerr = np.vstack([r - lo, hi - r])
        ax.errorbar(
            x + off,
            r,
            yerr=yerr,
            fmt="o",
            markersize=4.5,
            capsize=2.5,
            elinewidth=1.2,
            linewidth=0,
            label=name,
            alpha=0.95,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad = 18)
    ax.grid(True, axis="y", alpha=0.25)

    # Wrapped labels + extra spacing
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(sent_order, width=wrap_width), rotation=rotate, ha="center")
    ax.margins(x=0.01)

    # Give labels more breathing room
    fig.subplots_adjust(bottom=0.30, top=0.72)


    ax.legend(
    frameon=False,
    ncol=min(3, k),
    loc="upper center",
    bbox_to_anchor=(0.5, 1.35)
    )


    save_fig(fig, outbase)
    plt.close(fig)

def filter_rows_by_sentence_set(rows, keep_sents_set):
    """Keep only per-sentence rows whose sentence is in keep_sents_set."""
    return [r for r in rows if r[0] in keep_sents_set]

from scipy.io import savemat

def save_example_spectrogram(
    out_path,
    sentence,
    y_true,
    y_pred_v,
    y_pred_m,
    y_pred_i,
):
    """
    Save true + predicted spectrograms for one sentence.
    Arrays expected shape: (N_windows, F)
    We keep full resolution (no averaging).
    """
    savemat(out_path, {
        "sentence": sentence,
        "S_true": y_true,
        "S_pred_vocalized": y_pred_v,
        "S_pred_mimed": y_pred_m,
        "S_pred_imagined": y_pred_i,
    })

# -----------------------------
# Main analysis
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to cross_modality_results.npz")
    ap.add_argument("--outdir", required=True, help="Output directory for figures")
    ap.add_argument("--bootstrap", type=int, default=20, help="Bootstrap samples for sentence CIs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20, help="Top-K channels to show in importance barplot")
    ap.add_argument("--importance_mode", type=str, default="abs_mean", choices=["abs_mean", "l2"])
    ap.add_argument("--do_ablation", type=int, default=0, help="1 to run top-K channel ablation (slower)")
    ap.add_argument("--ablation_topk", type=int, default=10, help="How many top channels to ablate if do_ablation=1")
    ap.add_argument("--ablation_max_windows", type=int, default=8000, help="Max windows to use for ablation per modality (subsample)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_pub_style()

    npz = np.load(args.npz, allow_pickle=True)
    # Load modalities
    mimed = load_modality(npz, "mimed")
    imagined = load_modality(npz, "imagined")
    vocalized = load_modality(npz, "vocalized")

    # Global vocalized test performance (single-number reference)
    voc_spec_mean = np.nan
    voc_env_mean  = np.nan
    if vocalized is not None:
        voc_spec_mean = mean_freq_corr(vocalized["y_true"], vocalized["y_pred"])
        voc_env_mean  = envelope_corr(vocalized["y_true"], vocalized["y_pred"])

    if mimed is None and imagined is None and vocalized is None:
        raise RuntimeError("No mimed/imagined/vocalized data found in npz. Check keys in the file.")

    # Infer shapes from training X to compute channel importance
    if "X_train" not in npz.files or "decoder_coef" not in npz.files:
        raise RuntimeError("Expected keys X_train and decoder_coef in npz (you said you saved them).")

    X_train = np.array(npz["X_train"]) #(94200, 30, 110)
    window_frames = int(X_train.shape[1])
    n_channels = int(X_train.shape[2])

    decoder_coef = np.array(npz["decoder_coef"])  # (F, D)
    decoder_intercept = np.array(npz["decoder_intercept"])  # (F,)
    x_mean = np.array(npz["x_scaler_mean"])
    x_scale = np.array(npz["x_scaler_scale"])
    y_mean = np.array(npz["y_scaler_mean"])
    y_scale = np.array(npz["y_scaler_scale"])

    # Channel importance
    chan_imp = channel_importance_from_weights(decoder_coef, window_frames, n_channels, mode=args.importance_mode)
    chan_rank = np.argsort(-chan_imp)
    topk = min(args.topk, n_channels)
    top_ch = chan_rank[:topk]

    # -------- Figure 1: Channel importance barplot --------
    fig = plt.figure(figsize=(10, 4.5))
    ax = plt.gca()
    ax.bar(np.arange(topk), chan_imp[top_ch])
    ax.set_xticks(np.arange(topk))
    ax.set_xticklabels([f"ch{c}" for c in top_ch], rotation=60, ha="right")
    ax.set_ylabel("Importance (from weights)")
    ax.set_title(f"Top-{topk} channel importance (mode={args.importance_mode})")
    ax.grid(True, axis="y", alpha=0.25)
    save_fig(fig, os.path.join(args.outdir, "fig1_channel_importance_topk"))
    plt.close(fig)

    # -------- Per-sentence metrics --------
    def per_sentence_table(mod, name: str):
        print(f"[INFO] Computing per-sentence metrics for {name}...")
        groups = group_indices_by_sentence(mod["sentence"])
        rows = []
        for sent, idx in groups.items():
            yt = mod["y_true"][idx]
            yp = mod["y_pred"][idx]
            r_spec, lo_spec, hi_spec = bootstrap_ci(yt, yp, mean_freq_corr, n_boot=args.bootstrap, seed=args.seed)
            r_env, lo_env, hi_env = bootstrap_ci(yt, yp, envelope_corr, n_boot=args.bootstrap, seed=args.seed + 13)
            rows.append((sent, len(idx), r_spec, lo_spec, hi_spec, r_env, lo_env, hi_env))
        # sort by spectrogram corr desc
        rows.sort(key=lambda x: (-np.nan_to_num(x[2], nan=-1.0), x[0]))
        return rows
    # pdb.set_trace()
    rows_mimed = per_sentence_table(mimed, "mimed") if mimed is not None else None
    rows_imag = per_sentence_table(imagined, "imagined") if imagined is not None else None
    rows_voc = per_sentence_table(vocalized, "vocalized") if vocalized is not None else None

    # =========================
    # Save example spectrograms
    # =========================
    # Pick a sentence that exists in all modalities
    common_sents = set(rows_voc[i][0] for i in range(len(rows_voc)))
    if rows_mimed is not None:
        common_sents &= set(r[0] for r in rows_mimed)
    if rows_imag is not None:
        common_sents &= set(r[0] for r in rows_imag)

    example_sentence = sorted(list(common_sents))[3]  # deterministic choice
    print(f"[INFO] Saving spectrogram example for sentence: {example_sentence}")
    # Get indices
    idx_v = np.where(vocalized["sentence"] == example_sentence)[0]
    idx_m = np.where(mimed["sentence"] == example_sentence)[0]
    idx_i = np.where(imagined["sentence"] == example_sentence)[0]
    # Extract spectrograms (keep all windows)
    S_true = vocalized["y_true"][idx_v]
    S_voc  = vocalized["y_pred"][idx_v]
    S_mim  = mimed["y_pred"][idx_m]
    S_img  = imagined["y_pred"][idx_i]
    # Save
    out_mat = os.path.join(args.outdir, f"spectrogram_example_{example_sentence}.mat")
    save_example_spectrogram(
        out_mat,
        example_sentence,
        S_true,
        S_voc,
        S_mim,
        S_img,
    )
    print(f"[OK] Saved example spectrograms to {out_mat}")

    pdb.set_trace()
    if rows_voc is None:
        raise RuntimeError("Need vocalized test rows for test-only sentence restriction.")

    voc_test_sents = set([r[0] for r in rows_voc])   # sentences present in vocalized test subset
    sent_order_testonly = [r[0] for r in rows_voc]   # keep vocalized-test ordering (nice + consistent)
    rows_mimed_testonly = filter_rows_by_sentence_set(rows_mimed, voc_test_sents) if rows_mimed is not None else None
    rows_imag_testonly  = filter_rows_by_sentence_set(rows_imag,  voc_test_sents) if rows_imag  is not None else None
    rows_voc_testonly   = rows_voc  # already the test subset


    # Prefer vocalized order if exists; else mimed; else img.
    sent_order = []
    if rows_imag is not None:
        sent_order.extend([r[0] for r in rows_imag])
    if rows_mimed is not None:
        for s in [r[0] for r in rows_mimed]:
            if s not in sent_order:
                sent_order.append(s)

    def rows_to_arrays(rows, order):
        m = {r[0]: r for r in rows}
        n = len(order)
        Nw = np.zeros(n, dtype=int)
        r_spec = np.full(n, np.nan)
        lo_spec = np.full(n, np.nan)
        hi_spec = np.full(n, np.nan)
        r_env = np.full(n, np.nan)
        lo_env = np.full(n, np.nan)
        hi_env = np.full(n, np.nan)
        for i, s in enumerate(order):
            if s in m:
                _, nwin, rs, ls, hs, re, le, he = m[s]
                Nw[i] = int(nwin)
                r_spec[i] = rs
                lo_spec[i] = ls
                hi_spec[i] = hs
                r_env[i] = re
                lo_env[i] = le
                hi_env[i] = he
        return Nw, r_spec, lo_spec, hi_spec, r_env, lo_env, hi_env

    if rows_mimed is not None:
        _, rS_m, loS_m, hiS_m, rE_m, loE_m, hiE_m = rows_to_arrays(rows_mimed, sent_order)
    if rows_imag is not None:
        _, rS_i, loS_i, hiS_i, rE_i, loE_i, hiE_i = rows_to_arrays(rows_imag, sent_order)
    if rows_voc is not None:
        _, rS_v, loS_v, hiS_v, rE_v, loE_v, hiE_v = rows_to_arrays(rows_voc, sent_order)

    if rows_mimed_testonly is not None:
        _, rS_m_t, loS_m_t, hiS_m_t, rE_m_t, loE_m_t, hiE_m_t = rows_to_arrays(rows_mimed_testonly, sent_order_testonly)
    if rows_imag_testonly is not None:
        _, rS_i_t, loS_i_t, hiS_i_t, rE_i_t, loE_i_t, hiE_i_t = rows_to_arrays(rows_imag_testonly, sent_order_testonly)
    if rows_imag_testonly is not None:
        _, rS_v_t, loS_v_t, hiS_v_t, rE_v_t, loE_v_t, hiE_v_t = rows_to_arrays(rows_voc_testonly, sent_order_testonly)
    
    
    # pdb.set_trace()
    x = np.arange(len(sent_order))

   # -------- Figure 2: Per-sentence spectrogram correlation (mimed vs imagined vs vocalized) --------
    series = {}
    if rows_mimed is not None:
        series["Mimed"] = {"r": rS_m, "lo": loS_m, "hi": hiS_m}
    if rows_imag is not None:
        series["Imagined"] = {"r": rS_i, "lo": loS_i, "hi": hiS_i}
    if rows_voc is not None:
        series["Vocalized"] = {"r": rS_v, "lo": loS_v, "hi": hiS_v}

    plot_per_sentence_with_ci(
        outbase=os.path.join(args.outdir, "fig2_per_sentence_spectrogram_corr"),
        title="Per-sentence reconstruction performance (spectrogram)",
        ylabel="Mean spectrogram corr (mean over freq)",
        sent_order=sent_order,
        series_dict=series,
        rotate=70,          # keep upright since we wrap
        wrap_width=18,     # adjust if needed
        dodge=0.28,
    )


    # -------- Figure 3: Per-sentence envelope correlation (mimed vs imagined vs vocalized) --------
    series = {}
    if rows_mimed is not None:
        series["Mimed"] = {"r": rE_m, "lo": loE_m, "hi": hiE_m}
    if rows_imag is not None:
        series["Imagined"] = {"r": rE_i, "lo": loE_i, "hi": hiE_i}
    if rows_voc is not None:
        series["Vocalized"] = {"r": rE_v, "lo": loE_v, "hi": hiE_v}

    plot_per_sentence_with_ci(
        outbase=os.path.join(args.outdir, "fig3_per_sentence_envelope_corr"),
        title="Per-sentence reconstruction performance (envelope)",
        ylabel="Envelope corr (mean over freq, then corr)",
        sent_order=sent_order,
        series_dict=series,
        rotate=70,
        wrap_width=18,
        dodge=0.28,
    )

    # -------- Figure A1: Per-sentence spectrogram correlation (test-only) --------
    series = {}
    if rows_mimed_testonly is not None:
        series["Mimed (test)"] = {"r": rS_m_t, "lo": loS_m_t, "hi": hiS_m_t}
    if rows_imag_testonly is not None:
        series["Imagined (test)"] = {"r": rS_i_t, "lo": loS_i_t, "hi": hiS_i_t}
    series["Vocalized (test)"] = {"r": rS_v_t, "lo": loS_v_t, "hi": hiS_v_t}

    plot_per_sentence_with_ci(
        outbase=os.path.join(args.outdir, "figA1_testSentencesOnly_per_sentence_spectrogram"),
        title="Per-sentence performance (vocalized test sentences only)",
        ylabel="Mean spectrogram corr (mean over freq)",
        sent_order=sent_order_testonly,
        series_dict=series,
        rotate=70,
        wrap_width=18,
        dodge=0.28,
    )

    # -------- Figure A2: Per-sentence envelope correlation (test-only) --------
    series = {}
    if rows_mimed_testonly is not None:
        series["Mimed (test)"] = {"r": rE_m_t, "lo": loE_m_t, "hi": hiE_m_t}
    if rows_imag_testonly is not None:
        series["Imagined (test)"] = {"r": rE_i_t, "lo": loE_i_t, "hi": hiE_i_t}
    series["Vocalized (test)"] = {"r": rE_v_t, "lo": loE_v_t, "hi": hiE_v_t}
    plot_per_sentence_with_ci(
        outbase=os.path.join(args.outdir, "figA2_testonly_per_sentence_envelope"),
        title="Per-sentence envelope performance (test-only)",
        ylabel="Envelope corr (mean over freq, then corr)",
        sent_order=sent_order_testonly,
        series_dict=series,
        rotate=70,
        wrap_width=18,
        dodge=0.28,
    )

    def plot_mi_plus_vocalized_meanline(outbase, title, ylabel, sent_order,
                                   r_m, lo_m, hi_m, r_i, lo_i, hi_i,
                                   voc_mean, wrap_width=18):
        x = np.arange(len(sent_order))
        fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(sent_order)), 5.0))

        # dodge mimed/imagined so they don't overlap
        off_m, off_i = -0.14, +0.14

        ax.errorbar(
            x + off_m, r_m,
            yerr=np.vstack([r_m - lo_m, hi_m - r_m]),
            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.2,
            linewidth=0, label="Mimed (test)", alpha=0.95
        )
        ax.errorbar(
            x + off_i, r_i,
            yerr=np.vstack([r_i - lo_i, hi_i - r_i]),
            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.2,
            linewidth=0, label="Imagined (test)", alpha=0.95
        )

        ax.axhline(0.0, linestyle="--", linewidth=1, alpha=0.7)

        # vocalized mean (over its own test windows)
        if np.isfinite(voc_mean):
            ax.axhline(voc_mean, linestyle="-", linewidth=2, alpha=0.9, label="Vocalized mean (test subset)")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

        ax.set_xticks(x)
        ax.set_xticklabels(wrap_labels(sent_order, width=wrap_width), rotation=0, ha="center")
        fig.subplots_adjust(bottom=0.30)
        ax.legend(frameon=False, ncol=3, loc="upper left")

        save_fig(fig, outbase)
        plt.close(fig)


    plot_mi_plus_vocalized_meanline(
        outbase=os.path.join(args.outdir, "figB1_mi_allSent_plus_vocMean_spectrogram"),
        title="Mimed & Imagined per-sentence (all sentences) + Vocalized mean reference",
        ylabel="Mean spectrogram corr (mean over freq)",
        sent_order=sent_order,
        r_m=rS_m, lo_m=loS_m, hi_m=hiS_m,
        r_i=rS_i, lo_i=loS_i, hi_i=hiS_i,
        voc_mean=voc_spec_mean,
        wrap_width=18,
    )
    plot_mi_plus_vocalized_meanline(
    outbase=os.path.join(args.outdir, "figB2_mi_allSent_plus_vocMean_envelope"),
    title="Mimed & Imagined envelope per-sentence (all sentences) + Vocalized mean reference",
    ylabel="Envelope corr (mean over freq, then corr)",
    sent_order=sent_order,
    r_m=rE_m, lo_m=loE_m, hi_m=hiE_m,
    r_i=rE_i, lo_i=loE_i, hi_i=hiE_i,
    voc_mean=voc_env_mean,
    wrap_width=18,
    )



    # -------- Save a CSV-like table (TSV) --------
    tsv_path = os.path.join(args.outdir, "per_sentence_metrics.tsv")
    with open(tsv_path, "w") as f:
        f.write("sentence\tmodality\tn_windows\tspec_r\tspec_ci_lo\tspec_ci_hi\tenv_r\tenv_ci_lo\tenv_ci_hi\n")
        if rows_mimed is not None:
            for r in rows_mimed:
                f.write(f"{r[0]}\tmimed\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]:.6f}\n")
        if rows_imag is not None:
            for r in rows_imag:
                f.write(f"{r[0]}\timagined\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]:.6f}\n")
        if rows_voc is not None:
            for r in rows_voc:
                f.write(f"{r[0]}\tvocalized\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]:.6f}\n")
   

    print(f"[OK] Wrote per-sentence metrics: {tsv_path}")

    # -------- Optional: Ablation for top channels --------
    if args.do_ablation == 1:
        print("[INFO] Running top-channel ablation (this can be slower)...")

        ab_topk = min(args.ablation_topk, n_channels)
        ab_ch = chan_rank[:ab_topk]

        def run_ablation_for_mod(mod, prefix: str):
            # Need X_test too (saved in npz as prefix_X_test)
            kX = f"{prefix}_X_test"
            if kX not in npz.files:
                print(f"[WARN] No {kX} in npz; skipping ablation for {prefix}.")
                return None

            X = np.array(npz[kX])  # (N,T,C)
            yT = mod["y_true"]
            sent = mod["sentence"]

            # subsample windows for speed
            rng = np.random.default_rng(args.seed + 123)
            N = X.shape[0]
            useN = min(args.ablation_max_windows, N)
            idx = rng.choice(N, size=useN, replace=False)

            Xs = X[idx].copy()
            yTs = yT[idx].copy()
            sents = sent[idx].copy()

            # baseline recompute (should match stored y_pred approximately)
            yP0 = apply_linear_decoder(Xs, decoder_coef, decoder_intercept, x_mean, x_scale, y_mean, y_scale)
            base_r = mean_freq_corr(yTs, yP0)

            delta = []
            for c in ab_ch:
                Xa = Xs.copy()
                Xa[:, :, c] = 0.0
                yPa = apply_linear_decoder(Xa, decoder_coef, decoder_intercept, x_mean, x_scale, y_mean, y_scale)
                r = mean_freq_corr(yTs, yPa)
                delta.append(base_r - r)

            return base_r, np.array(delta), ab_ch

        # mimed
        if mimed is not None:
            out = run_ablation_for_mod(mimed, "mimed")
            if out is not None:
                base_r, delta, ab_ch_local = out
                fig = plt.figure(figsize=(8.5, 4.0))
                ax = plt.gca()
                ax.bar(np.arange(len(ab_ch_local)), delta)
                ax.set_xticks(np.arange(len(ab_ch_local)))
                ax.set_xticklabels([f"ch{c}" for c in ab_ch_local], rotation=60, ha="right")
                ax.set_ylabel("Δr (baseline - ablated)")
                ax.set_title(f"Mimed ablation impact (baseline r={base_r:.3f}, N={min(args.ablation_max_windows, np.array(npz['mimed_X_test']).shape[0])})")
                ax.grid(True, axis="y", alpha=0.25)
                save_fig(fig, os.path.join(args.outdir, "fig4_mimed_ablation_top_channels"))
                plt.close(fig)

        # imagined
        if imagined is not None:
            out = run_ablation_for_mod(imagined, "imagined")
            if out is not None:
                base_r, delta, ab_ch_local = out
                fig = plt.figure(figsize=(8.5, 4.0))
                ax = plt.gca()
                ax.bar(np.arange(len(ab_ch_local)), delta)
                ax.set_xticks(np.arange(len(ab_ch_local)))
                ax.set_xticklabels([f"ch{c}" for c in ab_ch_local], rotation=60, ha="right")
                ax.set_ylabel("Δr (baseline - ablated)")
                ax.set_title(f"Imagined ablation impact (baseline r={base_r:.3f}, N={min(args.ablation_max_windows, np.array(npz['imagined_X_test']).shape[0])})")
                ax.grid(True, axis="y", alpha=0.25)
                save_fig(fig, os.path.join(args.outdir, "fig5_imagined_ablation_top_channels"))
                plt.close(fig)

    print(f"[DONE] Figures saved in: {args.outdir}")


if __name__ == "__main__":
    main()
