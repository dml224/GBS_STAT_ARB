import numpy as np
import random
import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.decompositions import takagi as _sf_takagi
from strawberryfields.apps import sample as sf_sample
import inspect
import warnings

def gbs_sampling(B: np.ndarray,
                 num_samples: int,
                 target_mean_photons: float | None = None,
                 loss_rate: float = 0.0,
                 retries: int = 1,
                 base_seed: int | None = None,
                 verbose: bool = False,
                 displacement: complex | float | None = 0.0,
                 squeezing: float | None = None, # No additional squeezed beyond contribution from Takagi-Autonne
                 target_eff_samples: int | None = None,
                 batch_size: int | None = None,
                 max_batches: int = 1,
                 stats_out: dict | None = None,
                 compensate_via_displacement: bool = False,
                 disp_distribution: str = "degree") -> list[list[int]]:
    
    # Returns threshold-click masks sampled from a Gaussian Boson Sampling model
    # Each returned sample is a length-m binary vector indicating which modes clicked
    
    # Convert loss into transmission with a floor
    transmission = max(1.0 - float(loss_rate), 0.05) 
    m = B.shape[0]

    # Target detected photons; this sets the squeezing calibration objective
    n_mean_detect = float(target_mean_photons) if target_mean_photons is not None else float(np.sqrt(m))
    n_mean_inject = float(n_mean_detect)

    # Takagi decomposition of the (real) adjacency gives singular spectrum used for squeezing
    A = np.asarray(B, float).real
    U_tak, s_vals = _sf_takagi(A)
    sig = np.real(s_vals).astype(float) # Ensure real floats
    sig = np.clip(sig, 0.0, 1.0 - 1e-12)

    def _mean_ph_from_z(z_vec):
        z2 = z_vec**2
        return np.sum(z2 / (1.0 - z2))

    # Solve for a global scale c so expected photons from z=c*sig match target
    # Keeps z in [0,1) for physically valid Gaussian parameters
    c_max = (1.0 - 1e-6) / (sig.max() if sig.max() > 0 else 1.0)
    lo, hi = 0.0, c_max
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if _mean_ph_from_z(np.clip(mid * sig, 0, 1)) > n_mean_inject:
            hi = mid
        else:
            lo = mid
    c_opt = lo

    z = np.clip(c_opt * sig, 0.0, 1.0 - 1e-12)
    r_params = np.arctanh(z)

    # Optional coherent displacement used to compensate for expected loss
    disp_vec = None
    per_mode_ph = None
    tau = max(1e-12, 1.0 - float(loss_rate))
    n_squeezed = float(_mean_ph_from_z(z))
    extra_total = 0.0
    # For bookkeeping, intended total input photons (squeezed + displacement contributions)
    target_input_total = float(n_squeezed)

    if compensate_via_displacement:
        # Aim for detected mean photons approx n_mean_detect after loss, but cap total input
        target_input_total_raw = float(n_mean_detect) / float(tau)
        max_total = float(n_mean_detect) * 1000
        target_input_total = min(float(target_input_total_raw), max_total)
        extra_total = max(0.0, float(target_input_total) - float(n_squeezed))
        if extra_total > 0.0:
            try:
                if str(disp_distribution).lower() == "degree":
                    weights = np.sum(np.abs(A), axis=1)
                else:
                    weights = np.ones(m, dtype=float)
                wsum = float(np.sum(weights))
                if not np.isfinite(wsum) or wsum <= 0.0:
                    weights = np.ones(m, dtype=float)
                    wsum = float(m)
                per_mode_ph = (float(extra_total) * (weights / wsum)).astype(float)
                disp_vec = np.sqrt(np.maximum(per_mode_ph, 0.0)).astype(complex)
            except Exception:
                disp_vec = None
                per_mode_ph = None

    # Optional explicit uniform displacement 
    if disp_vec is None and displacement is not None:
        try:
            disp_vec = np.asarray([displacement] * m, dtype=complex)
            per_mode_ph = (np.abs(disp_vec) ** 2).astype(float)
            extra_total = float(np.sum(per_mode_ph))
            target_input_total = float(n_squeezed) + float(extra_total)
        except Exception:
            disp_vec = None
            per_mode_ph = None

    # Populate diagnostics early so callers always get displacement metadata
    if stats_out is not None:
        stats_out["n_mean_detect"] = float(n_mean_detect)
        stats_out["n_squeezed"] = float(n_squeezed)
        stats_out["target_input_total"] = float(target_input_total)
        stats_out["extra_disp_total"] = float(extra_total)
        stats_out["disp_distribution"] = str(disp_distribution)
        stats_out["disp_vec_norm"] = (0.0 if disp_vec is None else float(np.linalg.norm(disp_vec)))
        stats_out["displacement_kwarg"] = None
        stats_out["disp_vec_applied"] = False
        if per_mode_ph is not None:
            stats_out["disp_photons_per_mode"] = per_mode_ph.tolist()

    # Rebuild adjacency implied by chosen squeezing strengths for sampler consistency
    try:
        B_scaled = U_tak @ np.diag(z) @ U_tak.T
        B_scaled = 0.5 * (B_scaled + B_scaled.T)
        np.fill_diagonal(B_scaled, 0.0)
        B_scaled[np.abs(B_scaled) < 1e-12] = 0.0
    except Exception:
        B_scaled = None

    try:
        if base_seed is not None:
            seed_val = int(base_seed)
            np.random.seed(seed_val)
            try:
                sf_sample.set_seed(seed_val)
            except Exception:
                pass
            try:
                random.seed(seed_val)
            except Exception:
                pass
    except Exception:
        pass

    target_needed = int(target_eff_samples) if target_eff_samples is not None else int(num_samples)
    batch_sz = max(1, int(batch_size if batch_size is not None else num_samples))
    
    masks = []
    raw_clicks_sum = 0
    raw_clicks_count = 0
    raw_photons_sum = 0
    raw_photons_count = 0
    raw_attempted = 0
    batches_run = 0

    # SF sampler kwarg names vary across versions
    try:
        _sig = inspect.signature(sf_sample.sample)
        _params = set(_sig.parameters.keys())
    except Exception:
        _params = set()
    disp_kw = None
    for name in ("displacements", "displacement", "alphas", "coherent"):
        if name in _params:
            disp_kw = name
            break

    def _try_sample_with_disp(target, base_kwargs, disp_vector):
        # Try accepted displacement kwarg variants until one succeeds
        if disp_vector is None:
            return None, None
        cand_names = [disp_kw] if disp_kw else ["displacements", "displacement", "alphas", "coherent"]
        for name in cand_names:
            if not name:
                continue
            try:
                return sf_sample.sample(target, **{**base_kwargs, name: disp_vector}), name
            except Exception:
                continue
        return None, None

    def _explicit_threshold_sample(n_shots: int):
        nonlocal use_explicit_disp
        try:
            A_embed = B_scaled if B_scaled is not None else A
            mean_photon_per_mode = float(n_mean_inject) / float(m) if m > 0 else 0.0
            prog = sf.Program(m)
            with prog.context as q:
                # Explicit circuit path: graph embed -> displacement -> loss -> threshold measure
                ops.GraphEmbed(A_embed, mean_photon_per_mode=mean_photon_per_mode) | q
                if disp_vec is not None:
                    disp_op = getattr(ops, "Displacement", None) or getattr(ops, "Dgate", None)
                    for i in range(m):
                        alpha = disp_vec[i]
                        if np.abs(alpha) > 0 and disp_op is not None:
                            disp_op(alpha) | q[i]
                if float(loss_rate) > 0.0:
                    for i in range(m):
                        ops.LossChannel(float(transmission)) | q[i]
                ops.MeasureThreshold() | q
            eng = sf.LocalEngine(backend="gaussian")
            
            with warnings.catchwarnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Cannot simulate non-")
                res = eng.run(prog, shots=int(n_shots))
            use_explicit_disp = True
            return res.samples
        except Exception as e:
            if stats_out is not None:
                stats_out["explicit_disp_error"] = str(e)
            return None

    use_explicit_disp = False

    for attempt in range(int(max_batches)):
        try:
            # Apps sampler with loss parameter
            sf_kwargs = {
                "n_samples": int(batch_sz),
                "loss": float(1.0 - transmission),
            }

            # Fallback construction of a uniform displacement vector if needed
            displacement_kwarg_used = None
            if disp_vec is None:
                if displacement is not None:
                    try:
                        disp_vec = np.asarray([displacement] * m, dtype=complex)
                    except Exception:
                        disp_vec = None

            raw = None
            if disp_vec is not None and np.linalg.norm(disp_vec) > 0:
                raw = _explicit_threshold_sample(int(batch_sz))
                if raw is not None:
                    displacement_kwarg_used = "explicit_gaussian"

            if raw is None:
                if B_scaled is not None:
                    raw, displacement_kwarg_used = _try_sample_with_disp(B_scaled, sf_kwargs, disp_vec)
                    if raw is None:
                        raw = sf_sample.sample(B_scaled, **sf_kwargs)
                else:
                    sf_kwargs["n_mean"] = float(n_mean_inject)
                    raw, displacement_kwarg_used = _try_sample_with_disp(A, sf_kwargs, disp_vec)
                    if raw is None:
                        raw = sf_sample.sample(A, **sf_kwargs)

            # Record whether and how displacement was actually applied this attempt
            try:
                if use_explicit_disp and displacement_kwarg_used is None:
                    displacement_kwarg_used = "explicit_program"
                if stats_out is not None:
                    stats_out.setdefault("n_mean_detect", float(n_mean_detect))
                    stats_out.setdefault("n_squeezed", float(n_squeezed))
                    stats_out.setdefault("target_input_total", float(target_input_total))
                    stats_out.setdefault("extra_disp_total", float(extra_total))
                    stats_out.setdefault("disp_distribution", str(disp_distribution))
                    if disp_vec is None:
                        stats_out.setdefault("disp_vec_norm", 0.0)
                    else:
                        stats_out.setdefault("disp_vec_norm", float(np.linalg.norm(disp_vec)))
                    if per_mode_ph is not None:
                        stats_out.setdefault("disp_photons_per_mode", per_mode_ph.tolist())
                    # Overwrite with actual application status for this attempt
                    stats_out["displacement_kwarg"] = displacement_kwarg_used
                    stats_out["disp_vec_applied"] = (displacement_kwarg_used is not None and disp_vec is not None)
                if verbose:
                    print(f"[GBS] displacement_kwarg_used={displacement_kwarg_used}, disp_vec_norm={(0.0 if disp_vec is None else float(np.linalg.norm(disp_vec)))}")
            except Exception:
                pass

            # Normalise backend output into an iterable of integer count vectors
            raw_samples = None
            if isinstance(raw, (int, float)):
                raw_samples = [np.array([int(raw)] * m, dtype=int)]
            elif isinstance(raw, np.ndarray):
                if raw.ndim == 1:
                    raw_samples = [raw.astype(int)]
                elif raw.ndim == 2:
                    raw_samples = [row.astype(int) for row in raw]
                else:
                    raise RuntimeError(f"[GBS] Unexpected ndarray shape {raw.shape}")
            elif isinstance(raw, (list, tuple)):
                norm = []
                for x in raw:
                    if isinstance(x, (int, float)):
                        norm.append(np.array([int(x)] * m, dtype=int))
                    else:
                        arr = np.asarray(x, int).reshape(-1)
                        if arr.size == m:
                            norm.append(arr)
                raw_samples = norm
            else:
                raise RuntimeError(f"[GBS] Unsupported sample type: {type(raw)}")
            
            # If loss channel was not applied in-circuit
            if float(loss_rate) > 0.0 and not use_explicit_disp:
                try:
                    tau_post = max(0.0, 1.0 - float(loss_rate))
                    if tau_post < 1.0:
                        post = []
                        for arr in raw_samples:
                            a = np.asarray(arr, int).reshape(-1)
                            if a.size != m:
                                continue
                            a_loss = np.random.binomial(a, tau_post).astype(int)
                            post.append(a_loss)
                        raw_samples = post if post else raw_samples
                        if stats_out is not None:
                            stats_out["post_loss_applied"] = True
                except Exception:
                    if stats_out is not None:
                        stats_out["post_loss_applied"] = False

            last_raw = raw_samples

            raw_attempted += len(raw_samples)

            for counts in raw_samples:
                a = np.asarray(counts, int).reshape(-1)
                if a.size != m:
                    continue
                # Convert photon counts to threshold clicks and keep non-trivial events
                mask = (a > 0).astype(int)
                clicks = int(mask.sum())
                raw_clicks_sum += clicks
                raw_clicks_count += 1
                tot = int(a.sum())
                raw_photons_sum += tot
                raw_photons_count += 1
                if tot == 0:
                    continue
                if clicks < 2:
                    continue
                masks.append(mask.tolist())
                if len(masks) >= target_needed:
                    break
            last_raw = raw_samples

            raw_attempted += len(raw_samples)

            for counts in raw_samples:
                a = np.asarray(counts, int).reshape(-1)
                if a.size != m:
                    continue
                mask = (a > 0).astype(int)
                clicks = int(mask.sum())
                raw_clicks_sum += clicks
                raw_clicks_count += 1
                tot = int(a.sum())
                raw_photons_sum += tot
                raw_photons_count += 1
                if tot == 0:
                    continue
                if clicks < 2:
                    continue
                masks.append(mask.tolist())
                if len(masks) >= target_needed:
                    break

            raw_attempted += len(raw_samples)

            for counts in raw_samples:
                a = np.asarray(counts, int)
                # Convert photon counts to binary click mask
                mask = (a > 0).astype(int)
                
                clicks = int(mask.sum())
                raw_clicks_sum += clicks
                raw_clicks_count += 1
                raw_photons_sum += int(a.sum())
                raw_photons_count += 1
                
                # Filter trivial samples 
                if clicks < 2:
                    continue
                
                masks.append(mask.tolist())
                if len(masks) >= target_needed:
                    break
            
            batches_run += 1
            if len(masks) >= target_needed:
                break

        except Exception as e:
            if attempt >= retries:
                if verbose: print(f"[GBS Circuit] Error: {e}")
                break

    if stats_out is not None:
        stats_out.update({
            "raw_attempted": raw_attempted,
            "kept_masks": len(masks),
            "batches": batches_run,
            "raw_clicks_sum": raw_clicks_sum,
            "raw_clicks_count": raw_clicks_count,
            "raw_mean_clicks": (raw_clicks_sum / raw_clicks_count) if raw_clicks_count else 0.0,
            "raw_photons_sum": raw_photons_sum,
            "raw_photons_count": raw_photons_count,
            "raw_mean_photons": (raw_photons_sum / raw_photons_count) if raw_photons_count else 0.0,
            "tau": float(transmission),
            "n_mean_in": float(n_mean_inject),
        })

    return masks[:target_needed]
