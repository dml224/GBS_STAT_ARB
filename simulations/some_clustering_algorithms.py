import scipy.integrate
if not hasattr(scipy.integrate, "simps"):
   scipy.integrate.simps = scipy.integrate.simpson
import pandas as pd
from sklearn.cluster import SpectralClustering 
from signet.cluster import Cluster 
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np
import math
from collections import defaultdict

# Spectral #

def spectral_clustering(df, num, seed=42):  
    df = df.abs()
    G = nx.from_pandas_adjacency(df)
    sc = SpectralClustering(
        num,
        assign_labels="kmeans",
        affinity="precomputed",
        n_init=20,
        random_state=seed
    )
    a = sc.fit(nx.to_pandas_adjacency(G))
    return pd.DataFrame(list(zip(df.columns, a.labels_)), columns=["index","cluster"])


# SPONGE #

def SPONGE_sym_Clustering(df, num,method="sym"): 
    df_pos = df[df>=0].fillna(0)
    df_neg = -df[df<=0].fillna(0)
    c = Cluster((csr_matrix(df_pos.values), csr_matrix(df_neg.values)))
    if method == "regular": 
        predictions = c.SPONGE(k=num, tau_p=1, tau_n=1, eigens=None, mi=None)
    elif method =="sym":
        predictions = c.SPONGE_sym(k=num, tau_p=1, tau_n=1, eigens=None, mi=None)
    result = pd.DataFrame(df.columns)
    result.columns = ["index"]
    result["cluster"] = predictions
    return result


# GBS Roots #

def boost_clustering_agglomerative(adjacency_matrix,
                                   stock_names,
                                   loss_rate=0.0,
                                   squeezing=None,
                                   target_clusters=None,
                                   min_nodes=2,
                                   L=2,
                                   random_seed: int | None = None,
                                   value_metric: str = "sum_edges",
                                   residual_window_df: pd.DataFrame | None = None,
                                   max_seeds: int = 10,
                                   min_union_k: int = 2,
                                   group_corr_matrix: pd.DataFrame | np.ndarray | None = None):
    """
    GBS Roots / Boost_agglom:
      - Per iteration, draw one GBS sample batch on the residual graph.
      - Build one or more disjoint unions (>=2 samples each) with criteria weighted density (wd)
      - Remove all chosen unions' nodes and repeat.
    """
    A_signed_full = np.asarray(adjacency_matrix, float)
    if A_signed_full.shape[0] != len(stock_names):
        raise ValueError("Adjacency / stock_names length mismatch")
    total_n = A_signed_full.shape[0]
    full_adj = A_signed_full
    name_to_idx_full = {name: i for i, name in enumerate(stock_names)}

    def _cluster_value_full_from_names(name_list):
        idxs = []
        for nm in name_list:
            if nm not in name_to_idx_full:
                raise KeyError(f"Unknown stock name '{nm}' when scoring cluster.")
            idxs.append(name_to_idx_full[nm])
        return _cluster_value( # Get value, wd, intra and inter cluster weights
            idxs,
            full_adj,
            total_n,
            value_metric=value_metric,
            residual_window=residual_window_df,
            group_corr_matrix=group_corr_matrix
        )
    def _append_log(log_list: list, iteration: int, clusters: list[list[str]], chosen_names: list[str],
                    metrics: dict, mean_clicks_val: float, loss_rate: float, squeezing: float,
                    n_mean_eff: float | None = None, n_mean_base: float | None = None, tau: float | None = None,
                    gbs_stats: dict | None = None, disp_cluster_alloc: dict | None = None):
        log_list.append({
            "iteration": iteration,
            "cluster_id": len(clusters) - 1,
            "size": len(chosen_names),
            "stocks": chosen_names,
            "value": metrics.get("value", np.nan),
            "weighted_density": metrics.get("wd", np.nan),
            "raw_density": metrics.get("raw_density", np.nan),
            "neg_intra": metrics.get("neg_intra", np.nan),
            "pos_between": metrics.get("pos_between", np.nan),
            "mean_clicks": mean_clicks_val,
            "loss_rate": loss_rate,
            "squeezing": squeezing,
            "n_mean_eff": (float(n_mean_eff) if n_mean_eff is not None else np.nan),
            "n_mean_base": (float(n_mean_base) if n_mean_base is not None else np.nan),
            "tau": (float(tau) if tau is not None else np.nan),
            "gbs_kept_rate": (gbs_stats.get("kept_rate") if gbs_stats else np.nan),
            "gbs_kept_masks": (gbs_stats.get("kept_masks") if gbs_stats else np.nan),
            "gbs_raw_attempted": (gbs_stats.get("raw_attempted") if gbs_stats else np.nan),
            "gbs_n_mean_in": (gbs_stats.get("n_mean_in") if gbs_stats else np.nan),
            "gbs_batches": (gbs_stats.get("batches") if gbs_stats else np.nan),
            "detected_n_mean": ((gbs_stats.get("raw_mean_photons")
                                  if gbs_stats.get("raw_mean_photons", None) is not None
                                  else gbs_stats.get("raw_mean_clicks")) if gbs_stats else np.nan),
            "squeezing_r": (gbs_stats.get("takagi_r_mean") if gbs_stats else np.nan),
            "squeezing_r_vec": (gbs_stats.get("takagi_r") if gbs_stats else np.nan),
            "disp_vec_applied": (gbs_stats.get("disp_vec_applied") if gbs_stats else np.nan),
            "disp_vec_norm": (gbs_stats.get("disp_vec_norm") if gbs_stats else np.nan),
            "disp_distribution": (gbs_stats.get("disp_distribution") if gbs_stats else np.nan),
            "n_mean_detect": (gbs_stats.get("n_mean_detect") if gbs_stats else np.nan),
            "n_squeezed": (gbs_stats.get("n_squeezed") if gbs_stats else np.nan),
            "target_input_total": (gbs_stats.get("target_input_total") if gbs_stats else np.nan),
            "extra_disp_total": (gbs_stats.get("extra_disp_total") if gbs_stats else np.nan),
            "disp_cluster_alloc": (disp_cluster_alloc if disp_cluster_alloc is not None else {}),
        })

    clusters = []
    iteration_log = []
    sample_density_accum = []
    raw_clicks_sum_total = 0
    raw_clicks_count_total = 0
    sampler_success_flags = []
    iteration = 1

    A_res = A_signed_full.copy()
    current_names = list(stock_names)
    prev_residual_names = current_names.copy()

    # Target K feasibility (set to NONE for unconstraiend)
    targetK = None
    if target_clusters is not None:
        max_feasible = max(1, A_signed_full.shape[0] // max(2, min_nodes))
        targetK = min(target_clusters, max_feasible)

    while A_res.shape[0] > 0 and (targetK is None or len(clusters) < targetK):
        if targetK is not None:
            remaining_slots = max(0, targetK - len(clusters))
            if remaining_slots == 0:
                break
        else:
            remaining_slots = None
        n = A_res.shape[0]

        # Stopping rule: if residual is size 2, take it as final cluster; if size 1, take prior residual
        if n == 2:
            clusters.append(current_names.copy())
            break
        if n == 1:
            if prev_residual_names and len(prev_residual_names) >= min_nodes:
                clusters.append(prev_residual_names.copy())
            else:
                clusters.append(current_names.copy())
            break

        # Snapshot current residual for next-iteration fallback
        prev_residual_names = current_names.copy()

        if n <= 3:
            if n == 3:
                pair = _best_pair_from_B(np.maximum(A_res, 0.0)) # Pick strongest weight edge
                cluster_names = [current_names[i] for i in pair]
                clusters.append(cluster_names)
                leftover = [i for i in range(n) if i not in pair]
                if leftover:
                    attach_name = current_names[leftover[0]]

                    def _score_target(ci):
                        value, *_ = _cluster_value_full_from_names([*clusters[ci], attach_name])
                        return value

                    best_target = max(range(len(clusters)), key=_score_target)
                    clusters[best_target].append(attach_name)
            break

        min_size = 2
        guard_floor = max(min_nodes, 3)
        max_size = n - 1

        print(f"\n[GBS_Agglom] Iter {iteration}: residual nodes={n}, target_clusters={targetK}, "
              f"built={len(clusters)}, size bounds=[{min_size},{max_size}]")

        # Prepare B 
        B = build_B_matrix(A_res, max_spectral=0.95, clip_neg=False) # GBS input matrix
        smax = np.linalg.svd(B, compute_uv=False)[0] if B.size else 0.0

        if not np.allclose(B, B.T) or smax <= 1e-12:
            raise RuntimeError(f"[GBS_Agglom] Invalid B (symmetric={np.allclose(B, B.T)}, smax={smax:.3e})")

        # Photon budget 
        base_n_mean = float(math.sqrt(n))
        tau = max(0.0, 1.0 - float(loss_rate))
        n_mean_eff = float(base_n_mean)  # fixed across loss

        sample_batch = adjusted_sample_target(n, loss_rate) # Get number of samples per GBS call
        target_eff_samples = max(1, int(sample_batch))
        print(f"  [GBS_Agglom] Sampling once: batch={sample_batch}, target_eff={target_eff_samples}")
        gbs_stats = {}
        # Get GBS samples
        raw_samples = gbs_sampling( 
            B,
            num_samples=sample_batch,
            target_mean_photons=base_n_mean,
            retries=1,
            base_seed=(None if random_seed is None else random_seed + iteration * 997),
            verbose=False,
            loss_rate=float(loss_rate),
            squeezing=float(squeezing) if squeezing is not None else None,
            displacement=gbs_disp,
            compensate_via_displacement=True,
            disp_distribution="degree",
            target_eff_samples=target_eff_samples,
            stats_out=gbs_stats,
        )
        if DEBUG_DISPLACEMENT: # Boolean flag to control displacement printing
            err = gbs_stats.get("explicit_disp_error")
            msg = "  [GBS] displacement: applied={} norm={} extra_total={} kwarg={}".format(
                gbs_stats.get("disp_vec_applied"),
                gbs_stats.get("disp_vec_norm"),
                gbs_stats.get("extra_disp_total"),
                gbs_stats.get("displacement_kwarg"),
            )
            if err:
                msg += f" explicit_error={err}"
            print(msg)
        if gbs_stats and gbs_stats.get("raw_clicks_count", 0) > 0:
            raw_clicks_sum_total += int(gbs_stats.get("raw_clicks_sum", 0))
            raw_clicks_count_total += int(gbs_stats.get("raw_clicks_count", 0))
        sampler_success_flags.append(True)

        # Build unique sample pool 
        masks = [np.array(s, int) for s in raw_samples if len(s) == n]
        clicks = [int(np.count_nonzero(m)) for m in masks]
        sample_density_accum.extend(clicks)
        mean_clicks = float(np.mean(clicks)) if clicks else 0.0

        uniq = {}
        for m in masks:
            idxs = np.where(m > 0)[0].tolist()
            if len(idxs) < min_size or len(idxs) > max_size:
                continue
            uniq.setdefault(frozenset(idxs), idxs)
        sample_sets = list(uniq.values())

        # Prune sample sets: keep top-K by (frequency, quick intra-sum) per size
        MAX_SAMPLESETS_PER_SIZE = 80
        freq_map = defaultdict(int)
        for m in masks:
            freq_map[frozenset(np.where(m > 0)[0].tolist())] += 1

        def _quick_intra_sum(idxs: list[int]) -> float:
            if len(idxs) < 2:
                return 0.0
            sub = A_res[np.ix_(idxs, idxs)]
            return float(np.triu(sub, 1).sum())

        grouped = defaultdict(list)
        for s in sample_sets:
            grouped[len(s)].append(s)

        pruned_sets: list[list[int]] = []
        for sz, lst in grouped.items():
            scored = []
            for s in lst:
                key = frozenset(s)
                freq = freq_map.get(key, 1)
                proxy = _quick_intra_sum(s)
                scored.append((freq, proxy, s))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            pruned_sets.extend([s for _, _, s in scored[:MAX_SAMPLESETS_PER_SIZE]])
        sample_sets = pruned_sets

        # Prevent choices that strand a handful of nodes before the <=3 guard can run.
        def _residual_violates(candidate_indices, already_used):
            cand_set = set(candidate_indices)
            used = set(already_used).union(cand_set)
            remaining = n - len(used)
            return 0 < remaining < guard_floor

        # Scoring function
        def _score(idxs):
            v, wd, neg_intra, pos_between = _cluster_value(
                idxs, A_res, A_signed_full.shape[0],
                value_metric=value_metric,
                residual_window=residual_window_df,
                full_stock_names=current_names,
                group_corr_matrix=group_corr_matrix
            )
            return wd, v

        def _better_metrics(wd_a: float, v_a: float, wd_b: float, v_b: float, tol: float = 1e-9) -> bool:
            """
            Lexicographic compare: prioritise weighted density, then value.
            Returns True if (wd_a, v_a) is strictly better than (wd_b, v_b).
            """
            if wd_a > wd_b + tol:
                return True
            if wd_a + tol < wd_b:
                return False
            return v_a > v_b + tol

        # Seed: top-K unique samples by (wd, value)
        seeds = []
        for s in sample_sets:
            wd, v = _score(s)
            seeds.append((wd, v, s))
        # Prioritise weighted density, then value
        seeds.sort(key=lambda t: (t[0], t[1]), reverse=True)
        seeds = seeds[:max_seeds]

        # Greedy grow best union from seeds
        def _grow_best_union(sets_pool, used_global: set, slots_remaining_for_future: int | None):
            best_union_local = None
            best_metrics_local = None

            def _disjoint(a, b): return set(a).isdisjoint(b)

            for _, _, seed in seeds:
                if (not _disjoint(seed, used_global) or
                        _residual_violates(seed, used_global)):
                    continue
                U = list(seed)
                used = set(U)
                wdU, vU = _score(U)
                chosen_samples = 1
                improved = True
                while improved:
                    improved = False
                    best_gain = (-np.inf, -np.inf)
                    best_add = None
                    best_loc = None
                    for s in sets_pool:
                        if not _disjoint(s, used) or not _disjoint(s, used_global):
                            continue
                        # Constrained feasibility: leave enough for future clusters
                        if remaining_slots and remaining_slots > 1:
                            leftover = n - len(used_global.union(set(U)).union(s))
                            if leftover < (remaining_slots - 1) * min_size:
                                continue
                        cand = list(sorted(set(U).union(s)))
                        if len(cand) > max_size or _residual_violates(cand, used_global):
                            continue
                        wdC, vC = _score(cand)
                        gain = (wdC - wdU, vC - vU)
                        if _better_metrics(gain[0], gain[1], best_gain[0], best_gain[1]):
                            best_gain = gain
                            best_add = s
                            best_loc = (wdC, vC)
                    if best_add is not None and best_loc is not None and _better_metrics(best_loc[0], best_loc[1], wdU, vU):
                        U = list(sorted(set(U).union(best_add)))
                        used.update(best_add)
                        wdU, vU = best_loc
                        chosen_samples += 1
                        improved = True

                # Enforce minimum union_k by adding any disjoint set 
                if chosen_samples < min_union_k:
                    for s in sets_pool:
                        if _disjoint(s, used) and _disjoint(s, used_global):
                            cand = list(sorted(set(U).union(s)))
                            if len(cand) <= max_size and not _residual_violates(cand, used_global):
                                wdU, vU = _score(cand)
                                U = cand
                                chosen_samples += 1
                                break

                if best_union_local is None or _better_metrics(
                    wdU, vU,
                    (best_metrics_local or {}).get("wd", -np.inf),
                    (best_metrics_local or {}).get("value", -np.inf)
                ):
                    best_union_local = U
                    best_metrics_local = {"wd": wdU, "value": vU}
            return best_union_local, best_metrics_local

        # First (best) union
        used_all = set()
        best_union, best_union_metrics = _grow_best_union(sample_sets, used_all, remaining_slots)

        if best_union is None or best_union_metrics is None:
            raise RuntimeError("[GBS_Agglom][STRICT] No feasible union found from sampled sets.")

        # Guard: avoid taking full residual when more clusters needed (for unconstrained)
        if remaining_slots and remaining_slots > 1 and len(best_union) == n:
            target_sz = min(max_size, n - (remaining_slots - 1) * min_size)
            subA = A_res[np.ix_(best_union, best_union)]
            deg = subA.sum(axis=1)
            order = np.argsort(-deg)
            best_union = [best_union[i] for i in order[:target_sz]]
            wdU, vU = _score(best_union)
            best_union_metrics.update({"wd": wdU, "value": vU})
            print(f"  [GBS_Agglom] Reduced full-residual union to size {len(best_union)}")

        chosen_unions = [best_union]
        chosen_metrics = [best_union_metrics]
        used_all.update(best_union)

        # Append all chosen unions this iteration
        for U, M in zip(chosen_unions, chosen_metrics):
            namesU = [current_names[i] for i in U]
            clusters.append(namesU)
            disp_cluster_alloc = None
            disp_photons = gbs_stats.get("disp_photons_per_mode") if isinstance(gbs_stats, dict) else None
            if isinstance(disp_photons, (list, np.ndarray)) and len(disp_photons) == len(current_names):
                try:
                    disp_cluster_alloc = {current_names[i]: float(disp_photons[i]) for i in U}
                except Exception:
                    disp_cluster_alloc = None
            elif DEBUG_DISPLACEMENT:
                disp_cluster_alloc = {current_names[i]: 0.0 for i in U}
            _append_log(
                iteration_log,
                iteration,
                clusters,
                namesU,
                {"value": M["value"], "wd": M["wd"]},
                mean_clicks_val=mean_clicks,
                loss_rate=loss_rate,
                squeezing=(squeezing if squeezing is not None else np.nan),
                n_mean_eff=(tau * base_n_mean),
                n_mean_base=base_n_mean,
                tau=tau,
                gbs_stats=gbs_stats,
                disp_cluster_alloc=disp_cluster_alloc,
            )
            print(f"  [GBS_Agglom] Chosen union: size={len(U)}, wd={M['wd']:.4f}, value={M['value']:.4f}")

        # Remove all nodes used this iteration
        keep_idx = [i for i in range(n) if i not in used_all]
        A_res = A_res[np.ix_(keep_idx, keep_idx)]
        current_names = [current_names[i] for i in keep_idx]
        iteration += 1

    # Enforce disjointness (should already be disjoint; this only strips accidental repeats, not unique names)
    seen = set()
    cleaned = []
    for cl in clusters:
        uniq = [s for s in cl if s not in seen]
        if len(uniq) >= min_nodes:
            cleaned.append(uniq)
            seen.update(uniq)
        else:
            # If cluster fell below min_nodes after uniqueness filtering, redistribute its names
            for s in uniq:
                # attach to cluster with highest degree
                A_full_abs = np.abs(A_signed_full)
                name_to_idx_full = {n: i for i, n in enumerate(stock_names)}
                j = name_to_idx_full[s]
                best_c = None
                best_score = -1.0
                for ci, cl2 in enumerate(cleaned):
                    idxs = [name_to_idx_full[n] for n in cl2]
                    if not idxs: continue
                    sc = float(A_full_abs[j, idxs].mean())
                    if sc > best_score:
                        best_score = sc
                        best_c = ci
                if best_c is None:
                    cleaned.append([s])
                else:
                    cleaned[best_c].append(s)
                seen.add(s)
    clusters = cleaned

    # Coverage assertion to ensure every original name must appear exactly once
    original_set = set(stock_names)
    assigned_set = set(s for cl in clusters for s in cl)
    if assigned_set != original_set:
        missing = sorted(original_set - assigned_set)
        extra = sorted(assigned_set - original_set)
        raise AssertionError(f"[GBS_Agglom][COVERAGE] Lossless mode coverage failure. "
                             f"Missing={missing} Extra={extra}")

    clusters_dict = {i: c for i, c in enumerate(clusters)}

    # Singleton clusters are a violation in this mode (lossless, min_nodes=2)
    if any(len(cl) < min_nodes for cl in clusters):
        raise RuntimeError("[GBS_Agglom][STRICT] Found cluster below min_nodes.")

    print(f"[GBS_Agglom] Final clusters: {[len(v) for v in clusters_dict.values()]}")
    if DEBUG_DISPLACEMENT:
        alloc_entries = [e for e in iteration_log if e.get("disp_cluster_alloc")]
        if alloc_entries:
            print("  [GBS] per-mode displacement allocation (final clusters):")
            for e in alloc_entries:
                print(f"    cluster {e.get('cluster_id')}: {e.get('disp_cluster_alloc')}")

    if raw_clicks_count_total > 0:
        avg_sample_density = float(raw_clicks_sum_total / raw_clicks_count_total)
    else:
        avg_sample_density = (float(np.nanmean(sample_density_accum))
                              if sample_density_accum and not np.all(np.isnan(sample_density_accum))
                              else np.nan)
    sampler_success_rate = (np.mean(sampler_success_flags) if sampler_success_flags else 0.0)
    if sampler_success_rate == 0.0:
        avg_sample_density = np.nan
    return clusters_dict, iteration_log, avg_sample_density, sampler_success_rate
