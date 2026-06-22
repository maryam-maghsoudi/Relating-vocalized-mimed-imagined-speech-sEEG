#!/usr/bin/env python3
"""
decoder_filter_analysis.py

Analysis of Ridge decoder coefficients ("filters") saved in:
  cross_modality_train_{vocalized|mimed|imagined}_with_heldout_test.npz

Runs 4 analyses for each trained modality:
  A) Channel importance (energy per channel)
  B) Temporal (lag) profile (energy per lag)
  C) Frequency profile (energy per spectrogram frequency bin)
  D) Cross-model similarity between decoders (cosine similarity)

Outputs:
  - PNG figures saved under OUT_DIR
  - .npz with computed metrics for reuse
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------- CONFIG ---------------------------
BASE = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results"
OUT_DIR = os.path.join(BASE, "decoder_filter_analysis_noncausal")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_MODES = ["vocalized", "mimed", "imagined"]

# File naming convention you used:
NPZ_TEMPLATE = "cross_modality_centered_p300_f300_train_{mode}_with_heldout_test.npz"

# If you ever change window/hop, we infer n_lags from the model coef shape.
# We also infer n_channels from N_CH_KEEP saved inside each NPZ.
# --------------------------------------------------------------


def load_npz_for_mode(mode: str):
    path = os.path.join(BASE, NPZ_TEMPLATE.format(mode=mode))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing NPZ for mode={mode}: {path}")
    return np.load(path, allow_pickle=True), path


def infer_shapes(npz):
    """
    Infer (F, L, C) from decoder_coef and saved metadata.

    decoder_coef: shape (F, D) where D = L*C
    N_CH_KEEP: number of channels
    L inferred as D // C

    Returns
    -------
    F : int  (# spectrogram frequency bins)
    L : int  (# lags/frames in window)
    C : int  (# channels kept)
    """
    W_flat = npz["decoder_coef"]
    if W_flat.ndim != 2:
        raise ValueError(f"decoder_coef expected 2D, got shape {W_flat.shape}")

    F, D = W_flat.shape
    C = int(npz["N_CH_KEEP"]) if "N_CH_KEEP" in npz else None
    if C is None:
        raise KeyError("NPZ missing N_CH_KEEP; cannot infer #channels.")

    if D % C != 0:
        raise ValueError(f"Cannot reshape decoder_coef: D={D} not divisible by C={C}")

    L = D // C
    return F, L, C


def reshape_weights(npz):
    """
    Reshape decoder_coef into (F, L, C).
    Assumes flattening order was [lag-major, then channel] or [channel-major, then lag].
    We try one order and validate by basic sanity; if needed, swap axes later.

    train_linear_decoder likely used windows shaped (N, L, C) then flattened to (N, L*C)
    which corresponds to lag-major then channel (C changes fastest).
    """
    W_flat = npz["decoder_coef"]  # (F, D)
    F, L, C = infer_shapes(npz)
    W = W_flat.reshape(F, L, C)   # (F, L, C)
    return W, F, L, C


def analysis_A_channel_energy(W):
    """
    Channel importance as energy pooled over frequency and lag.
    W: (F, L, C)
    Returns: (C,)
    """
    # Frobenius norm across (F, L) for each channel
    return np.linalg.norm(W, axis=(0, 1))


def analysis_B_lag_energy(W):
    """
    Temporal/lag profile as energy pooled over frequency and channels.
    W: (F, L, C)
    Returns: (L,)
    """
    return np.linalg.norm(W, axis=(0, 2))


def analysis_C_freq_energy(W):
    """
    Frequency profile as energy pooled over lag and channels.
    W: (F, L, C)
    Returns: (F,)
    """
    return np.linalg.norm(W, axis=(1, 2))


def plot_overlay(series_dict, xlabel, ylabel, title, out_path, x=None, legend_loc="best"):
    """
    series_dict: {name: 1D array}
    """
    plt.figure(figsize=(7, 3.5))
    for name, y in series_dict.items():
        if x is None:
            plt.plot(y, linewidth=2, label=name)
        else:
            plt.plot(x, y, linewidth=2, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(frameon=False, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def compute_G_channel_lag_zscore(
    W: np.ndarray,
    use_abs: bool = False,
    eps: float = 1e-12,
):
    """
    Parameters
    ----------
    W : np.ndarray, shape (F, L, C)
        Decoder weights
    use_abs : bool
        Whether to use |W| before averaging over frequency
    eps : float
        Numerical stability for std

    Returns
    -------
    G : np.ndarray, shape (C, L)
        Frequency-averaged channel-lag map
    Gz : np.ndarray, shape (C, L)
        Channel-wise z-scored version across lags
    """
    assert W.ndim == 3, "W must have shape (F, L, C)"

    # 1) collapse frequency
    if use_abs:
        G = np.mean(np.abs(W), axis=0).T   # (C, L)
    else:
        G = np.mean(W, axis=0).T           # (C, L)

    # 2) z-score within each channel across lags
    mu = G.mean(axis=1, keepdims=True)
    std = G.std(axis=1, keepdims=True) + eps
    Gz = (G - mu) / std

    return G, Gz

def plot_similarity_matrix(sim_mat, labels, out_path):
    plt.figure(figsize=(4.5, 4))
    im = plt.imshow(sim_mat, aspect="equal", interpolation="none")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("Decoder cosine similarity")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def analysis_E_timefreq_map(W):
    """
    Time-frequency "filter map" by averaging decoder coefficients across channels.

    Parameters
    ----------
    W : np.ndarray, shape (F, L, C)

    Returns
    -------
    G_signed : (F, L)  signed mean across channels
    G_absmean: (F, L)  mean absolute value across channels
    """
    G_signed = W.mean(axis=2)          # (F, L)
    G_absmean = np.abs(W).mean(axis=2) # (F, L)
    return G_signed, G_absmean

def analysis_F_channel_lag_map(W):
    """
    Channel × lag maps by averaging over frequency.

    Parameters
    ----------
    W : np.ndarray, shape (F, L, C)

    Returns
    -------
    G_cl_signed : np.ndarray, shape (C, L)
        Signed mean over frequency
    G_cl_absmean : np.ndarray, shape (C, L)
        Mean absolute over frequency
    """
    G_cl_signed, G_cl_zscored = compute_G_channel_lag_zscore(W, use_abs=False)
    # G_cl_signed = W.mean(axis=0).T          # (L, C) -> transpose -> (C, L)
    G_cl_absmean = np.abs(W).mean(axis=0).T # (L, C) -> transpose -> (C, L)
    return G_cl_zscored, G_cl_absmean

# def plot_heatmap(G, title, out_path, hop_ms=None, lag_offset_ms=0.0):
def plot_heatmap(G, title, out_path, hop_ms=None, lag_offset_ms=0.0, xlabel="Lag (ms)", ylabel="Spectrogram frequency bin"):
    """
    G: (F, L)
    x-axis: lag (ms) if hop_ms provided else lag index
    y-axis: frequency bin index
    """
    F, L = G.shape
    plt.figure(figsize=(6.5, 4.0))

    if hop_ms is not None and hop_ms > 0:
        # extent = [0, (L - 1) * hop_ms, 0, F - 1]
        extent = [lag_offset_ms,
                  lag_offset_ms + (G.shape[1] - 1) * hop_ms, 
                  0, F - 1]
        plt.imshow(G, aspect="auto", origin="lower", interpolation="none", extent=extent)
        plt.xlabel(xlabel)
    else:
        plt.imshow(G, aspect="auto", origin="lower", interpolation="none")
        plt.xlabel("Lag index")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_heatmap_3panel(G_dict, title, out_path, hop_ms=None):
    """
    G_dict: {"Vocalized": (F,L), "Mimed": (F,L), "Imagined": (F,L)}
    Creates a 1x3 comparison figure.
    """
    modes = list(G_dict.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), sharey=True)

    # choose common color scale across panels for fair comparison
    all_vals = np.concatenate([G_dict[m].ravel() for m in modes])
    vmin, vmax = np.percentile(all_vals, [2, 98])  # robust limits (avoids outlier-driven saturation)

    for ax, m in zip(axes, modes):
        G = G_dict[m]
        F, L = G.shape

        if hop_ms is not None and hop_ms > 0:
            extent = [0, (L - 1) * hop_ms, 0, F - 1]
            im = ax.imshow(G, aspect="auto", origin="lower", interpolation="none",
                           extent=extent, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Lag (ms)")
        else:
            im = ax.imshow(G, aspect="auto", origin="lower", interpolation="none",
                           vmin=vmin, vmax=vmax)
            ax.set_xlabel("Lag index")

        ax.set_title(f"Train on {m}")

    axes[0].set_ylabel("Spectrogram frequency bin")
    fig.suptitle(title, fontsize=14)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=200)
    plt.close()

def crop_lag_edges(G, hop_ms, crop_ms=10.0):
    """
    Crop first and last `crop_ms` milliseconds along lag axis.

    Parameters
    ----------
    G : np.ndarray, shape (F, L)
        Time–frequency map
    hop_ms : float
        Hop size in milliseconds
    crop_ms : float
        Amount to crop at start and end (ms)

    Returns
    -------
    G_crop : np.ndarray
    lag_start : int
        Starting lag index after crop
    lag_end : int
        Ending lag index after crop (exclusive)
    """
    if hop_ms is None or hop_ms <= 0:
        return G, 0, G.shape[1]

    n_crop = int(np.round(crop_ms / hop_ms))
    if 2 * n_crop >= G.shape[1]:
        raise ValueError("Crop too large relative to number of lags.")

    return G[:, n_crop:-n_crop], n_crop, G.shape[1] - n_crop

def temporal_profile_from_heatmap(G, mode="mean"):
    """
    Collapse time–frequency map across frequency.

    Parameters
    ----------
    G : np.ndarray, shape (F, L)
    mode : str, "mean" or "sum"

    Returns
    -------
    g : np.ndarray, shape (L,)
    """
    if mode == "mean":
        return G.mean(axis=0)
    elif mode == "sum":
        return G.sum(axis=0)
    else:
        raise ValueError("mode must be 'mean' or 'sum'")

def plot_temporal_profiles_from_heatmaps(
    G_dict,
    hop_ms,
    out_dir,
    prefix,
    mode_names=None,
    ylabel="Avg filter weight",
    title_suffix="",
    plot_overlay=True,
):
    """
    Plot temporal profiles obtained by averaging heatmaps across frequency.

    Parameters
    ----------
    G_dict : dict
        {mode_name: G}, where G has shape (F, L) and is already cropped
    hop_ms : float
        Hop size in milliseconds
    out_dir : str
        Directory to save figures
    prefix : str
        Filename prefix (e.g., "signed_mean", "abs_mean")
    mode_names : list or None
        Order of modes to plot (default: keys of G_dict)
    ylabel : str
        Y-axis label
    title_suffix : str
        Extra text for titles
    plot_overlay : bool
        Whether to also create a single overlay plot across modes
    """

    os.makedirs(out_dir, exist_ok=True)

    if mode_names is None:
        mode_names = list(G_dict.keys())

    temporal_profiles = {}

    # -------- per-mode plots --------
    for mode in mode_names:
        G = G_dict[mode]              # (F, L)
        g_time = G.mean(axis=0)       # (L,)
        temporal_profiles[mode] = g_time

        L = g_time.shape[0]
        x = np.arange(L) * hop_ms

        plt.figure(figsize=(5.5, 3))
        plt.plot(x, g_time, linewidth=2)
        plt.xlabel("Lag (ms)")
        plt.ylabel(ylabel)
        plt.title(f"Temporal decoder profile\nTrain on {mode}{title_suffix}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{prefix}_temporal_profile_train_{mode}.png"),
            dpi=200,
        )
        plt.close()

    # -------- overlay plot --------
    if plot_overlay:
        plt.figure(figsize=(6.5, 3.5))
        for mode in mode_names:
            g_time = temporal_profiles[mode]
            L = g_time.shape[0]
            x = np.arange(L) * hop_ms
            plt.plot(x, g_time, linewidth=2, label=mode)

        plt.xlabel("Lag (ms)")
        plt.ylabel(ylabel)
        plt.title(f"Temporal decoder profile (avg over frequency){title_suffix}")
        plt.legend(frameon=False)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{prefix}_temporal_profile_overlay.png"),
            dpi=200,
        )
        plt.close()

    return temporal_profiles

def plot_channel_lag_heatmap(Gcl, title, out_path, hop_ms=None, lag_offset_ms=0.0):
    """
    Gcl: (C, L) channel × lag
    """
    C, L = Gcl.shape
    plt.figure(figsize=(7.2, 5.0))

    if hop_ms is not None and hop_ms > 0:
        extent = [lag_offset_ms, lag_offset_ms + (L - 1) * hop_ms, 0, C - 1]
        plt.imshow(Gcl, aspect="auto", origin="lower", interpolation="none", extent=extent)
        plt.xlabel("Lag (ms)")
    else:
        plt.imshow(Gcl, aspect="auto", origin="lower", interpolation="none")
        plt.xlabel("Lag index")

    plt.ylabel("Channel index")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    # ------------------ LOAD + ANALYZE EACH TRAIN MODE ------------------
    Ws = {}              # flattened for similarity
    W3 = {}              # reshaped (F, L, C)
    meta = {}            # per mode metadata
    A = {}               # channel energy
    B = {}               # lag energy
    C = {}               # freq energy

    for mode in TRAIN_MODES:
        npz, path = load_npz_for_mode(mode)
        W, F, L, Cn = reshape_weights(npz)

        Ws[mode] = npz["decoder_coef"].ravel().astype(np.float64)
        W3[mode] = W.astype(np.float64)

        HOP_MS = float(npz["hop_size_ms"]) if "hop_size_ms" in npz else None
        meta[mode] = dict(
            path=path,
            F=F,
            L=L,
            C=Cn,
            hop_ms=HOP_MS,
        )

        A[mode] = analysis_A_channel_energy(W3[mode])
        B[mode] = analysis_B_lag_energy(W3[mode])
        C[mode] = analysis_C_freq_energy(W3[mode])

        print(f"[INFO] Loaded {mode}: W (F,L,C)=({F},{L},{Cn})  hop_ms={HOP_MS}")

    # ------------------ ANALYSIS D: CROSS-MODE SIMILARITY ------------------
    labels = [m.capitalize() for m in TRAIN_MODES]
    sim_mat = np.zeros((len(TRAIN_MODES), len(TRAIN_MODES)), dtype=float)

    for i, mi in enumerate(TRAIN_MODES):
        for j, mj in enumerate(TRAIN_MODES):
            sim_mat[i, j] = cosine_similarity([Ws[mi]], [Ws[mj]])[0, 0]

    # ------------------ PLOTS (OVERLAYS) ------------------
    # A) Channel energy overlay
    plot_overlay(
        {m.capitalize(): A[m] for m in TRAIN_MODES},
        xlabel="Channel index",
        ylabel="Filter energy (||W||)",
        title="A) Channel importance (energy pooled over freq + lag)",
        out_path=os.path.join(OUT_DIR, "A_channel_energy_overlay.png"),
        legend_loc="best",
    )

    # B) Lag energy overlay (x-axis in ms if hop_size_ms available and consistent)
    # If hop differs across modes (unlikely), we just plot index.
    hop_vals = [meta[m]["hop_ms"] for m in TRAIN_MODES]
    if all(h is not None for h in hop_vals) and len(set(hop_vals)) == 1:
        hop_ms = hop_vals[0]
        # assume same L across modes too; if not, x won't match, so fall back
        L_vals = [meta[m]["L"] for m in TRAIN_MODES]
        if len(set(L_vals)) == 1:
            x_lag = np.arange(L_vals[0]) * hop_ms
            plot_overlay(
                {m.capitalize(): B[m] for m in TRAIN_MODES},
                xlabel="Lag (ms)",
                ylabel="Filter energy (||W||)",
                title="B) Temporal (lag) profile (energy pooled over freq + channels)",
                out_path=os.path.join(OUT_DIR, "B_lag_energy_overlay.png"),
                x=x_lag,
                legend_loc="best",
            )
        else:
            plot_overlay(
                {m.capitalize(): B[m] for m in TRAIN_MODES},
                xlabel="Lag index",
                ylabel="Filter energy (||W||)",
                title="B) Temporal (lag) profile (energy pooled over freq + channels)",
                out_path=os.path.join(OUT_DIR, "B_lag_energy_overlay.png"),
                legend_loc="best",
            )
    else:
        plot_overlay(
            {m.capitalize(): B[m] for m in TRAIN_MODES},
            xlabel="Lag index",
            ylabel="Filter energy (||W||)",
            title="B) Temporal (lag) profile (energy pooled over freq + channels)",
            out_path=os.path.join(OUT_DIR, "B_lag_energy_overlay.png"),
            legend_loc="best",
        )

    # C) Frequency profile overlay (index as x)
    plot_overlay(
        {m.capitalize(): C[m] for m in TRAIN_MODES},
        xlabel="Spectrogram frequency bin",
        ylabel="Filter energy (||W||)",
        title="C) Frequency dependence (energy pooled over lag + channels)",
        out_path=os.path.join(OUT_DIR, "C_freq_energy_overlay.png"),
        legend_loc="best",
    )

    # D) Similarity matrix
    plot_similarity_matrix(
        sim_mat, labels,
        out_path=os.path.join(OUT_DIR, "D_decoder_cosine_similarity_matrix.png")
    )

        # ------------------ ANALYSIS E: TIME-FREQUENCY FILTER MAP ------------------
    G_signed = {}
    G_absmean = {}
    G_time_signed = {}
    G_time_abs = {}
    for mode in TRAIN_MODES:
        hop_ms = meta[mode]["hop_ms"]
        Gs, Ga = analysis_E_timefreq_map(W3[mode])
        hop_ms = meta[mode]["hop_ms"]
        Gs_crop, i0, i1 = crop_lag_edges(Gs, hop_ms, crop_ms=20.0)
        Ga_crop, _, _   = crop_lag_edges(Ga, hop_ms, crop_ms=20.0)
        G_signed[mode] = Gs_crop
        G_absmean[mode] = Ga_crop
        g_time_signed = temporal_profile_from_heatmap(Gs_crop, mode="mean")
        g_time_abs    = temporal_profile_from_heatmap(Ga_crop, mode="mean")

        # store
        G_time_signed[mode] = g_time_signed
        G_time_abs[mode]    = g_time_abs

        # Per-mode heatmaps
        plot_heatmap(
            Gs_crop,
            title=f"E) Signed mean filter map (avg over channels)\nTrain on {mode.capitalize()}",
            out_path=os.path.join(OUT_DIR, f"E_heatmap_signed_mean_train_{mode}.png"),
            hop_ms=hop_ms,
            lag_offset_ms=i0 * hop_ms
        )
        plot_heatmap(
            Ga_crop,
            title=f"E) Abs-mean filter map (avg |W| over channels)\nTrain on {mode.capitalize()}",
            out_path=os.path.join(OUT_DIR, f"E_heatmap_abs_mean_train_{mode}.png"),
            hop_ms=hop_ms,
            lag_offset_ms=i0 * hop_ms
        )

    G_time_signed = plot_temporal_profiles_from_heatmaps(
        G_dict={m: G_signed[m] for m in TRAIN_MODES},
        hop_ms=hop_ms,
        out_dir=OUT_DIR,
        prefix="signed_mean",
        ylabel="Avg filter weight",
        title_suffix=" (signed mean)",
    )


    # 3-panel comparisons (signed + absmean)
    # plot_heatmap_3panel(
    #     {m.capitalize(): G_signed[m] for m in TRAIN_MODES},
    #     title="E) Signed mean decoder filter map (avg over channels)",
    #     out_path=os.path.join(OUT_DIR, "E_heatmap_signed_mean_3panel.png"),
    #     hop_ms=meta[TRAIN_MODES[0]]["hop_ms"]
    # )

    # plot_heatmap_3panel(
    #     {m.capitalize(): G_absmean[m] for m in TRAIN_MODES},
    #     title="E) Abs-mean decoder filter map (avg |W| over channels)",
    #     out_path=os.path.join(OUT_DIR, "E_heatmap_abs_mean_3panel.png"),
    #     hop_ms=meta[TRAIN_MODES[0]]["hop_ms"]
    # )

    # ------------------ ANALYSIS F: CHANNEL × LAG HEATMAPS ------------------
    F_Gcl_signed = {}
    F_Gcl_absmean = {}

    for mode in TRAIN_MODES:
        hop_ms = meta[mode]["hop_ms"]
        # import pdb
        # pdb.set_trace()
        Gcl_signed, Gcl_absmean = analysis_F_channel_lag_map(W3[mode])
        
        if hop_ms is not None and hop_ms > 0:
            # crop function expects (F, L) so we adapt:
            # treat channels as "frequency axis" for cropping purposes
            Gcl_signed_crop, i0, i1 = crop_lag_edges(Gcl_signed, hop_ms, crop_ms=20.0)
            Gcl_absmean_crop, _, _  = crop_lag_edges(Gcl_absmean, hop_ms, crop_ms=20.0)
            lag_offset_ms = i0 * hop_ms
        else:
            Gcl_signed_crop = Gcl_signed
            Gcl_absmean_crop = Gcl_absmean
            lag_offset_ms = 0.0

        F_Gcl_signed[mode] = Gcl_signed_crop
        F_Gcl_absmean[mode] = Gcl_absmean_crop

        # per-mode plots
        plot_channel_lag_heatmap(
            Gcl_signed_crop,
            title=f"F) Channel×Lag filter map (signed mean over freq)\nTrain on {mode.capitalize()}",
            out_path=os.path.join(OUT_DIR, f"F_channel_lag_signed_train_{mode}.png"),
            hop_ms=hop_ms,
            lag_offset_ms=lag_offset_ms,
        )


    # ------------------ SAVE METRICS ------------------
    out_npz = os.path.join(OUT_DIR, "decoder_filter_metrics.npz")
    save = {}

    # store arrays
    for m in TRAIN_MODES:
        save[f"{m.lower()}_temporal_profile_signed"] = G_time_signed[m]
        save[f"{m.lower()}_temporal_profile_abs"] = G_time_abs[m]
        save[f"{m}_G_signed_mean"] = G_signed[m]
        save[f"{m}_G_abs_mean"] = G_absmean[m]
        save[f"{m}_channel_energy"] = A[m]
        save[f"{m}_lag_energy"] = B[m]
        save[f"{m}_freq_energy"] = C[m]
        # store shapes & hop
        save[f"{m}_FLC"] = np.array([meta[m]["F"], meta[m]["L"], meta[m]["C"]], dtype=int)
        save[f"{m}_hop_ms"] = np.array([meta[m]["hop_ms"] if meta[m]["hop_ms"] is not None else -1.0], dtype=float)

    save["similarity_matrix"] = sim_mat
    save["train_modes"] = np.array(TRAIN_MODES, dtype=object)

    np.savez(out_npz, **save)
    print(f"[INFO] Saved metrics to {out_npz}")

    print("\n[INFO] Saved figures:")
    print(" -", os.path.join(OUT_DIR, "A_channel_energy_overlay.png"))
    print(" -", os.path.join(OUT_DIR, "B_lag_energy_overlay.png"))
    print(" -", os.path.join(OUT_DIR, "C_freq_energy_overlay.png"))
    print(" -", os.path.join(OUT_DIR, "D_decoder_cosine_similarity_matrix.png"))


if __name__ == "__main__":
    main()
