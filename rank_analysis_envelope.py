import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Core helpers
# -----------------------------
def corr_1d(a, b, eps=1e-12):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    L = min(len(a), len(b))
    if L < 2:
        return np.nan
    a = a[:L]; b = b[:L]
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def build_sentence_level_envs(npz, prefix="mimed"):
    """
    Groups windows by (sentence, rep) and returns one 1D envelope vector per (sentence, rep).
    Returns:
      keys: list of (sentence, rep) sorted
      pred_list: list of 1D arrays (one per key)
      true_list: list of 1D arrays (one per key)
    """
    sent = npz[f"{prefix}_sentence"]          # (N_windows,)
    rep  = npz[f"{prefix}_rep"]               # (N_windows,)
    y_true = npz[f"{prefix}_envelope_true"]   # (N_windows,)
    y_pred = npz[f"{prefix}_envelope_pred"]   # (N_windows,)

    groups = {}
    for s, r, yt, yp in zip(sent, rep, y_true, y_pred):
        key = (str(s), int(r))
        if key not in groups:
            groups[key] = {"true": [], "pred": []}
        groups[key]["true"].append(float(yt))
        groups[key]["pred"].append(float(yp))

    keys = sorted(groups.keys(), key=lambda x: (x[0], x[1]))

    true_list, pred_list = [], []
    for k in keys:
        t = np.array(groups[k]["true"], dtype=np.float64)
        p = np.array(groups[k]["pred"], dtype=np.float64)
        L = min(len(t), len(p))
        true_list.append(t[:L])
        pred_list.append(p[:L])

    return keys, pred_list, true_list


def assert_consecutive_pairs(keys):
    """
    Checks pattern: (s,1),(s,2),(s2,1),(s2,2),...
    """
    assert len(keys) % 2 == 0, "Expected even number of items (pairs)."
    for i in range(0, len(keys), 2):
        s1, r1 = keys[i]
        s2, r2 = keys[i + 1]
        assert s1 == s2, f"Non-matching sentence pair at rows {i},{i+1}: {keys[i]} vs {keys[i+1]}"
        assert {r1, r2} == {1, 2}, f"Expected reps {{1,2}} at rows {i},{i+1}, got reps {r1},{r2}"


def average_2x2_blocks(R200):
    """
    Compress 200x200 -> 100x100 by averaging each 2x2 block.
    """
    N = R200.shape[0]
    assert R200.shape[0] == R200.shape[1], "R must be square."
    assert N % 2 == 0, "R size must be even."
    M = N // 2
    R100 = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            block = R200[2*i:2*i+2, 2*j:2*j+2]
            R100[i, j] = np.nanmean(block)
    return R100


def compute_rank_matrix(pred_list, true_list):
    N = len(pred_list)
    assert N == len(true_list)
    R = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            R[i, j] = corr_1d(pred_list[i], true_list[j])
    return R


def diagonal_ranks(R, higher_is_better=True):
    """
    Rank of diagonal entry within each row. 1 = best.
    """
    N = R.shape[0]
    ranks = np.zeros(N, dtype=int)
    for i in range(N):
        row = R[i, :]
        order = np.argsort(-row) if higher_is_better else np.argsort(row)
        ranks[i] = int(np.where(order == i)[0][0] + 1)
    return ranks


def topk_curve(ranks, k_list):
    """
    Returns array of accuracies for each k (same order as k_list).
    """
    ranks = np.asarray(ranks)
    return np.array([np.mean(ranks <= k) for k in k_list], dtype=np.float64)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_rank_hist_overlay(ranks_dict, out_png, title, bins=None):
    """
    ranks_dict: {label: ranks_array}
    """
    plt.figure(figsize=(8, 4.5))
    if bins is None:
        max_rank = max(int(np.max(r)) for r in ranks_dict.values())
        bins = np.arange(1, max_rank + 2) - 0.5

    for label, ranks in ranks_dict.items():
        plt.hist(ranks, bins=bins, alpha=0.45, label=f"{label} (n={len(ranks)})")

    plt.xlabel("Rank (1 = best)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()


def plot_topk_curves(curves_dict, k_list, N_items, out_png, title):
    """
    curves_dict: {label: acc_array aligned with k_list}
    chance for top-k is k/N_items (NOT 1/k).
    """
    plt.figure(figsize=(7, 4.5))

    for label, acc in curves_dict.items():
        plt.plot(k_list, acc, marker="o", label=label)

    chance = np.array(k_list, dtype=np.float64) / float(N_items)
    plt.plot(k_list, chance, marker=".", linestyle="--", label=f"Chance (k/{N_items})")

    plt.xlabel("k")
    plt.ylabel("Top-k accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    plt.show()


# -----------------------------
# One split analysis
# -----------------------------
def run_split(npz, prefix, out_dir, k_list):
    """
    Runs rank analysis for one prefix: 'mimed', 'imagined', or 'vocalized'
    Returns a dict with R200/R100, ranks, and top-k.
    """
    # check required keys exist
    need = [f"{prefix}_sentence", f"{prefix}_rep", f"{prefix}_envelope_true", f"{prefix}_envelope_pred"]
    for k in need:
        if k not in npz:
            raise ValueError(f"Missing key in NPZ: {k}")

    keys, pred_list, true_list = build_sentence_level_envs(npz, prefix=prefix)
    N = len(keys)
    print(f"[INFO] {prefix}: Found {N} (sentence, rep) items")

    R200 = compute_rank_matrix(pred_list, true_list)

    # pair-averaging: 200 -> 100
    assert_consecutive_pairs(keys)
    R100 = average_2x2_blocks(R200)

    ranks200 = diagonal_ranks(R200, higher_is_better=True)
    ranks100 = diagonal_ranks(R100, higher_is_better=True)

    acc200 = topk_curve(ranks200, k_list)
    acc100 = topk_curve(ranks100, k_list)

    # save per-split artifacts
    if out_dir is not None:
        split_dir = os.path.join(out_dir, prefix)
        os.makedirs(split_dir, exist_ok=True)

        np.save(os.path.join(split_dir, f"{prefix}_R200.npy"), R200)
        np.save(os.path.join(split_dir, f"{prefix}_ranks200.npy"), ranks200)
        np.save(os.path.join(split_dir, f"{prefix}_R100.npy"), R100)
        np.save(os.path.join(split_dir, f"{prefix}_ranks100.npy"), ranks100)

        # quick standalone hist + topk per split (optional but useful)
        plot_rank_hist_overlay(
            {f"{prefix} (R100)": ranks100},
            out_png=os.path.join(split_dir, f"{prefix}_rank_hist_R100.png"),
            title=f"{prefix}: rank histogram after 2x2 avg (N={R100.shape[0]})",
            bins=np.arange(1, R100.shape[0] + 2) - 0.5
        )

        plt.figure(figsize=(7,4.5))
        plt.plot(k_list, acc100, marker="o", label=f"{prefix} (R100)")
        chance = np.array(k_list, dtype=np.float64) / float(R100.shape[0])
        plt.plot(k_list, chance, marker=".", linestyle="--", label=f"Chance (k/{R100.shape[0]})")
        plt.xlabel("k"); plt.ylabel("Top-k accuracy")
        plt.title(f"{prefix}: top-k (R100)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(split_dir, f"{prefix}_topk_R100.png"), dpi=150)
        plt.show()

    return {
        "keys": keys,
        "R200": R200, "ranks200": ranks200, "acc200": acc200,
        "R100": R100, "ranks100": ranks100, "acc100": acc100,
        "N200": R200.shape[0], "N100": R100.shape[0],
    }


# -----------------------------
# Main driver
# -----------------------------
def main(npz_path, out_dir):
    npz = np.load(npz_path, allow_pickle=True)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # choose ks (cap at 100 for R100)
    k_list = [1, 2, 5, 10, 20, 50, 100]

    # run the three splits you asked for
    results = {}
    for prefix in ["vocalized", "mimed", "imagined"]:
        if f"{prefix}_envelope_true" not in npz:
            print(f"[WARN] prefix '{prefix}' not found in NPZ (skipping).")
            continue
        results[prefix] = run_split(npz, prefix=prefix, out_dir=out_dir, k_list=k_list)

    # --- Combined plots (R100) across the 3 sets ---
    # rank histograms
    ranks_dict = {p: results[p]["ranks100"] for p in results.keys()}
    plot_rank_hist_overlay(
        ranks_dict,
        out_png=os.path.join(out_dir, "ALL_rank_hist_R100.png") if out_dir else None,
        title="Rank histograms after rep-averaging (R100)",
        bins=np.arange(1, 100 + 2) - 0.5
    )

    # top-k curves + chance
    curves_dict = {p: results[p]["acc100"] for p in results.keys()}
    plot_topk_curves(
        curves_dict,
        k_list=k_list,
        N_items=100,
        out_png=os.path.join(out_dir, "ALL_topk_R100.png") if out_dir else None,
        title="Top-k after rep-averaging (R100)"
    )

    # print a compact summary
    for p in results.keys():
        r = results[p]["ranks100"]
        print(f"[SUMMARY] {p:9s}  R100: mean rank={r.mean():.1f}, median rank={np.median(r):.1f}, top1={np.mean(r==1):.3f}")

    return results


if __name__ == "__main__":
    npz_path = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results_envelope/cross_modality_envelope_results_with_vcl_test.npz"
    out_dir  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/vocalmind/stim_rcnst_2/generalization_results_envelope/rank_analysis"
    main(npz_path, out_dir)
