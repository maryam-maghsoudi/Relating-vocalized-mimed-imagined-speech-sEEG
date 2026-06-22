#!/usr/bin/env python3
"""
Build a 200x200 identification matrix (predicted vs true) where each sentence has 2 reps:
  index 2*k   = sentence_k rep1
  index 2*k+1 = sentence_k rep2

Rows: predicted (per item)
Cols: true      (per item)

We collapse frequency by averaging over F=128, so each item is a 1D time series (windows).

Outputs:
  - R_pred_vs_true_<modality>_200x200.npy
  - R_pred_vs_true_<modality>_200x200.png
  - R_pred_vs_true_<modality>_200x200_labels.npz (labels + ordering)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb

from pyparsing import Any


def resample_1d(x, T_new):
    """Linear resample 1D array x to length T_new."""
    if len(x) == T_new:
        return x
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    t_new = np.linspace(0.0, 1.0, num=T_new, endpoint=True)
    return np.interp(t_new, t_old, x)


def build_items(data, modality):
    """
    Returns:
      items_pred: list of 1D arrays (one per (sent,rep))
      items_true: list of 1D arrays
      labels:     list of strings "SentenceName_rep1"
      order:      list of (sentence, rep)
    """
    sent_key = f"{modality}_sentence"
    rep_key  = f"{modality}_rep"
    y_true_key = f"{modality}_y_test"
    y_pred_key = f"{modality}_y_test_pred"

    for k in [sent_key, rep_key, y_true_key, y_pred_key]:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in npz. Available keys include: {sorted(data.keys())[:30]} ...")

    sentences = data[sent_key]          # (N_windows,)
    reps      = data[rep_key].astype(int)
    y_true    = data[y_true_key]        # (N_windows, F)
    y_pred    = data[y_pred_key]        # (N_windows, F)

    # Determine sentence order (sorted unique strings, like your np.unique did)
    unique_sents = np.unique(sentences)

    items_pred = []
    items_true = []
    labels = []
    order = []

    # We expect reps {1,2}, but don’t hard-crash if something else appears.
    for s in unique_sents:
        for r in [1, 2]:
            m = (sentences == s) & (reps == r)
            if not np.any(m):
                raise ValueError(f"No data for sentence='{s}' rep={r} in modality='{modality}'")

            # Collapse frequency: (Nw, F) -> (Nw,)
            true_series = y_true[m].mean(axis=1)
            pred_series = y_pred[m].mean(axis=1)

            items_true.append(true_series.astype(np.float64))
            items_pred.append(pred_series.astype(np.float64))
            labels.append(f"{s}_rep{r}")
            order.append((str(s), int(r)))

    if len(items_true) != 2 * len(unique_sents):
        raise RuntimeError("Unexpected item count; expected 2 items per sentence.")

    return items_pred, items_true, labels, order, unique_sents


def zscore_rows(X, eps=1e-8):
    """Row-wise z-score. X shape: (N, T)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

def split_same_vs_other_sentence(R):
    """
    Split a 200x200 matrix into same-sentence and other-sentence values.

    Assumes:
      - 2 consecutive indices correspond to one sentence
      - index // 2 gives sentence index

    Parameters
    ----------
    R : np.ndarray, shape (200, 200)

    Returns
    -------
    same_vals : np.ndarray
        Values where predicted sentence == true sentence
        (includes 2x2 blocks)
    other_vals : np.ndarray
        All other entries
    """
    assert R.shape[0] == R.shape[1], "R must be square"
    N = R.shape[0]

    same_vals = []
    other_vals = []

    for i in range(N):
        si = i // 2
        for j in range(N):
            sj = j // 2
            if si == sj:
                same_vals.append(R[i, j])
            else:
                other_vals.append(R[i, j])

    return np.array(same_vals), np.array(other_vals)


def plot_same_vs_other_hist(R, out_dir, modality, bins=60):
    same_vals, other_vals = split_same_vs_other_sentence(R)

    plt.figure(figsize=(8, 5))

    plt.hist(
        other_vals,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Other sentence",
        color="C1",
    )

    plt.hist(
        same_vals,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Same sentence",
        color="C0",
    )

    plt.axvline(
        same_vals.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Same mean = {same_vals.mean():.3f}",
        color="C0",
    )
    plt.axvline(
        other_vals.mean(),
        linestyle="--",
        linewidth=2,
        label=f"Other mean = {other_vals.mean():.3f}",
        color="C1",
    )

    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f"Pred vs True correlation distribution — {modality}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"histogram_pred_vs_true_{modality}.png"), dpi=160)

    print("Same-sentence:")
    print(f"  mean = {same_vals.mean():.4f}")
    print(f"  std  = {same_vals.std():.4f}")
    print(f"  N    = {len(same_vals)}")
    print("Other-sentence:")
    print(f"  mean = {other_vals.mean():.4f}")
    print(f"  std  = {other_vals.std():.4f}")
    print(f"  N    = {len(other_vals)}")

def sentence_level_matrix_from_200(R, agg="mean"):
    """
    Aggregate an (N x N) matrix into a sentence-level matrix by averaging
    2x2 blocks, where N must be even.

    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Rep-level matrix (pred vs true)
    agg : str
        "mean" or "median"

    Returns
    -------
    R_sent : np.ndarray, shape (N//2, N//2)
        Sentence-level matrix
    """
    assert R.ndim == 2 and R.shape[0] == R.shape[1], "R must be square"
    N = R.shape[0]
    assert N % 2 == 0, "Matrix size must be even (2 reps per sentence)"

    n_sent = N // 2
    R_sent = np.zeros((n_sent, n_sent), dtype=np.float64)

    for si in range(n_sent):
        r0, r1 = 2 * si, 2 * si + 1
        for sj in range(n_sent):
            c0, c1 = 2 * sj, 2 * sj + 1
            block = R[r0:r1+1, c0:c1+1]  # 2x2
            if agg == "median":
                R_sent[si, sj] = np.median(block)
            else:
                R_sent[si, sj] = np.mean(block)

    return R_sent

def split_diag_vs_offdiag(R):
    """For a square matrix: diag values vs off-diagonal values."""
    diag = np.diag(R)
    off  = R[~np.eye(R.shape[0], dtype=bool)]
    return diag, off

def plot_sentence_level_hist(R100, modality, out_dir, bins=50):
    diag, off = split_diag_vs_offdiag(R100)

    plt.figure(figsize=(8, 5))
    plt.hist(off,  bins=bins, density=True, alpha=0.6, label="Other sentence (off-diag)", color="C1")
    plt.hist(diag, bins=int(bins/2), density=True, alpha=0.6, label="Same sentence (diag)")

    plt.axvline(diag.mean(), linestyle="--", linewidth=2, label=f"Diag mean = {diag.mean():.3f}", color="C0")
    plt.axvline(off.mean(),  linestyle="--", linewidth=2, label=f"Off mean = {off.mean():.3f}", color="C1"  )

    plt.xlabel("Correlation")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"Sentence-level (100x100) correlation distribution — {modality}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"histogram_sentence_level_{modality}.png"), dpi=160)

    print("Sentence-level (100x100) stats:")
    print(f"  diag: mean={diag.mean():.4f}, std={diag.std():.4f}, N={len(diag)}")
    print(f"  off : mean={off.mean():.4f}, std={off.std():.4f}, N={len(off)}")


def compute_sentence_ranks(R):
    """
    Compute rank of the correct sentence for each row of R.

    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Sentence-level correlation matrix

    Returns
    -------
    ranks : np.ndarray, shape (N,)
        Rank of correct sentence (1 = best)
    """
    assert R.shape[0] == R.shape[1], "R must be square"
    N = R.shape[0]

    ranks = np.zeros(N, dtype=int)

    for i in range(N):
        row = R[i]
        # argsort descending
        sorted_idx = np.argsort(row)[::-1]
        # find position of correct sentence
        rank = np.where(sorted_idx == i)[0][0] + 1  # +1 for 1-based rank
        ranks[i] = rank

    return ranks

def plot_rank_distribution(ranks, modality, out_dir):
    N = len(ranks)
    # pdb.set_trace()
    chance_rank = (N + 1) / 2

    plt.figure(figsize=(7, 5))
    plt.hist(ranks, bins=np.arange(1, N+2), density=True, alpha=0.7)
    plt.axvline(chance_rank, color="k", linestyle="--",
                label=f"Chance = {chance_rank:.1f}")
    plt.axvline(np.mean(ranks), color="r", linestyle="--",
                label=f"Mean rank = {np.mean(ranks):.1f}")

    plt.xlabel("Rank of correct sentence (1 = best)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title(f"Sentence identification rank distribution — {modality}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"rank_distribution_{modality}.png"), dpi=160)

    print("Rank stats:")
    print(f"  Mean rank:   {np.mean(ranks):.2f}")
    print(f"  Median rank:{np.median(ranks):.2f}")
    print(f"  Chance rank:{chance_rank:.2f}")

def topk_curve_from_ranks(ranks, max_k=None):
    """
    Compute top-k accuracy curve from 1-based ranks.

    ranks: array-like, shape (N,)
      ranks[i] in {1..N}, where 1=best

    Returns:
      ks: (K,) array of k values
      acc: (K,) array where acc[k-1] = P(rank <= k)
    """
    ranks = np.asarray(ranks).astype(int)
    N = len(ranks)
    if max_k is None:
        max_k = N
    max_k = int(max_k)

    ks = np.arange(1, max_k + 1)
    acc = np.array([(ranks <= k).mean() for k in ks], dtype=float)
    return ks, acc

def plot_topk_curves(ranks_dict, max_k=None, title=None, save_path=None):
    """
    Plot top-k accuracy curves for multiple modalities/models.

    ranks_dict: dict[str, np.ndarray]
      e.g. {"Mimed": ranks_mimed, "Imagined": ranks_imagined, "Vocalized": ranks_vocalized}

    max_k: int or None
      maximum k to plot (default: N)

    Produces:
      - curves for each entry in ranks_dict
      - chance line y = k / N for each N (assumes all N equal; if not, uses first)
    """
    # Determine N (assume consistent across curves; if not, will still plot per-curve correctly)
    first_key = next(iter(ranks_dict))
    N0 = len(np.asarray(ranks_dict[first_key]))
    if max_k is None:
        max_k = N0

    plt.figure(figsize=(7.5, 5.5))

    # Plot each curve
    for name, ranks in ranks_dict.items():
        ranks = np.asarray(ranks).astype(int)
        N = len(ranks)
        mk = min(int(max_k), N)
        ks, acc = topk_curve_from_ranks(ranks, max_k=mk)
        plt.plot(ks, acc, linewidth=2, label=f"{name} (N={N})")

    # Chance line (use N0 and max_k for display)
    ks = np.arange(1, int(max_k) + 1)
    chance = ks / float(N0)
    plt.plot(ks, chance, "k--", linewidth=2, label=f"Chance (k/N, N={N0})")

    plt.xlabel("k (Top-k)")
    plt.ylabel("Identification accuracy  P(rank ≤ k)")
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] Saved top-k plot to: {save_path}")

    plt.close()

def print_topk_table(ranks, ks=(1, 5, 10, 20)):
    """
    Print top-k accuracies and chance levels for a single ranks array.
    """
    ranks = np.asarray(ranks).astype(int)
    N = len(ranks)
    print(f"N={N}")
    for k in ks:
        k = int(k)
        acc = (ranks <= k).mean()
        chance = k / float(N)
        print(f"  Top-{k:>2}: acc={acc:.3f}   chance={chance:.3f}   lift={acc/chance:.2f}x")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results/cross_modality_train_imagined_with_heldout_test.npz", help="Path to cross_modality_results_with_vcl_test.npz")
    ap.add_argument("--modality", default="imagined", choices=["imagined", "mimed", "vocalized"],
                    help="Which modality to use for pred vs true matching.")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: same folder as npz)")
    ap.add_argument("--T", type=int, default=0,
                    help="Optional fixed resample length. If 0, use min length across items.")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    npz_path = args.npz
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)

    out_dir = args.out_dir or os.path.dirname(npz_path)
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    # pdb.set_trace()
    items_pred, items_true, labels, order, unique_sents = build_items(data, args.modality)

    # Check lengths and choose common T
    lengths = np.array([len(x) for x in items_true], dtype=int)
    lengths_pred = np.array([len(x) for x in items_pred], dtype=int)

    if not np.all(lengths == lengths_pred):
        # This should not happen if masks align, but just in case:
        raise ValueError("Mismatch between true and pred window counts for some (sentence,rep).")

    T_common = args.T if args.T > 0 else int(lengths.min())
    if np.any(lengths != T_common):
        print(f"[WARN] Not all items have same window length. Resampling all to T={T_common}.")
        print(f"       lengths: min={lengths.min()}, max={lengths.max()}, unique={np.unique(lengths)}")

    # Stack into matrices (N_items=200, T_common)
    P = np.stack([resample_1d(x, T_common) for x in items_pred], axis=0)
    G = np.stack([resample_1d(x, T_common) for x in items_true], axis=0)
    # pdb.set_trace()
    # Correlation matrix via dot product of z-scored rows
    Pz = zscore_rows(P)
    Gz = zscore_rows(G)
    R = (Pz @ Gz.T) / float(T_common)   # (200, 200)

    print("\n" + "=" * 70)
    print(f"Built R for modality='{args.modality}'  shape={R.shape}  T={T_common}")
    print(f"Sentences: {len(unique_sents)}  (items = {len(labels)})")
    print("Diagonal stats:")
    diag = np.diag(R)
    print(f"  mean={diag.mean():.4f}  std={diag.std():.4f}  min={diag.min():.4f}  max={diag.max():.4f}")
    off = R[~np.eye(R.shape[0], dtype=bool)]
    print("Off-diagonal stats:")
    print(f"  mean={off.mean():.4f}  std={off.std():.4f}  min={off.min():.4f}  max={off.max():.4f}")
    print("=" * 70)

    # Save matrix
    out_npy = os.path.join(out_dir, f"R_pred_vs_true_{args.modality}_200x200.npy")
    np.save(out_npy, R)
    print(f"[OK] Saved matrix: {out_npy}")

    # Save labels + ordering for later use
    out_labels = os.path.join(out_dir, f"R_pred_vs_true_{args.modality}_200x200_labels.npz")
    # np.savez(
    #     out_labels,
    #     labels=np.array(labels, dtype=object),
    #     order=np.array(order, dtype=object),
    #     unique_sentences=np.array(unique_sents, dtype=object),
    #     T_common=np.array([T_common], dtype=int),
    # )
    print(f"[OK] Saved labels: {out_labels}")

    # Plot heatmap (optional but useful)
    # fig = plt.figure(figsize=(10, 9))
    # ax = plt.gca()
    # im = ax.imshow(R, aspect="auto", origin="lower", interpolation="none")
    # ax.set_title(f"Pred vs True identification matrix (200×200) — {args.modality}")
    # ax.set_xlabel("True item (sentence, rep)")
    # ax.set_ylabel("Pred item (sentence, rep)")
    # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.tight_layout()

    # out_png = os.path.join(out_dir, f"R_pred_vs_true_{args.modality}_200x200.png")
    # plt.savefig(out_png, dpi=args.dpi)
    # print(f"[OK] Saved heatmap: {out_png}")

    # plot_same_vs_other_hist(R, out_dir, args.modality, bins=60)
    R100 = sentence_level_matrix_from_200(R, agg="mean")
    # plot_sentence_level_hist(R100, args.modality, out_dir, bins=50)
    ranks = compute_sentence_ranks(R100)
    
    from scipy.io import savemat
    train_modality = "imagined"
    test_modality  = args.modality
    fname = f"sentence_ranks_train-{train_modality}_test-{test_modality}.mat"
    out_path = os.path.join(out_dir, fname)
    savemat(out_path, {
    "ranks": ranks,
    "train_modality": train_modality,
    "test_modality": test_modality
    })


    pdb.set_trace()
    # plot_rank_distribution(ranks, args.modality, out_dir)
    out_topk = os.path.join(out_dir, f"topk_curve_{args.modality}.png")
    plot_topk_curves({args.modality: ranks}, max_k=10, title=f"Top-k curve — {args.modality}", save_path=out_topk)
    print_topk_table(ranks, ks=(1, 5, 10))  


if __name__ == "__main__":
    main()
