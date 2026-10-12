"""Locked internal holdout analysis; never infer sample size from training runs.

Input is final/{family}/test_metrics.csv plus metadata.json. Each CSV contains
exactly one row per held-out patient, computed AFTER probability-averaging all
5-fold x 3-seed checkpoints. Duplicate patient rows are rejected: averaging
Dice over seeds is not equivalent to evaluating a probability ensemble, and
seeds/folds/eyes/pixels are not independent patients.

The prespecified primary endpoint is patient-macro Dice, with proposed versus
U-Net as the primary comparison. Patient-paired percentile bootstrap estimates
the mean Dice difference and its 95% CI. The two-sided sign-flip permutation
test assumes exchangeability of method labels within patient under the null.
Holm correction across all proposed-versus-baseline Dice comparisons is also
reported; other metrics, subgroups, and corruptions are descriptive. This is
single-centre INTERNAL holdout validation, not external or clinical validation.
The bootstrap conditions on fitted ensembles and does not capture training-
sample uncertainty. A narrow CI cannot repair selection bias, label noise,
test reuse, or a missed anatomical target. No clinical superiority threshold
or power claim is generated automatically.

Boundary metric definitions and empty-mask policies are documented in
conjseg.metrics. Metrics Reloaded: https://doi.org/10.1038/s41592-023-02151-z.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRIC_NAMES


def _finite_vector(values, name="values"):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a non-empty finite 1-D vector")
    return x


def bootstrap_mean_ci(values, bootstrap=10000, seed=20261009):
    """Patient bootstrap percentile 95% CI; reusable for paired differences."""
    x = _finite_vector(values)
    if int(bootstrap) != bootstrap or bootstrap < 100:
        raise ValueError("bootstrap must be an integer >= 100")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(bootstrap), dtype=np.float64)
    # Keep memory bounded even if the function is used on a larger cohort.
    for start in range(0, int(bootstrap), 1000):
        end = min(start + 1000, int(bootstrap))
        samples = rng.integers(0, len(x), size=(end - start, len(x)))
        estimates[start:end] = x[samples].mean(axis=1)
    lo, hi = np.percentile(estimates, (2.5, 97.5))
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(proposed, baseline, bootstrap=10000, seed=20261009):
    """Resample patient-paired differences, never the two methods separately."""
    p, b = _finite_vector(proposed, "proposed"), _finite_vector(baseline, "baseline")
    if p.shape != b.shape:
        raise ValueError("Paired method vectors must have identical length")
    mean, lower, upper = bootstrap_mean_ci(p - b, bootstrap, seed)
    return {"mean_delta_dice": mean, "ci95_lower": lower, "ci95_upper": upper,
            "n_patients": len(p)}


def sign_flip_pvalue(differences, permutations=20000, seed=20261009):
    """Two-sided Monte Carlo paired sign-flip p, with add-one correction.

    All-zero differences return 1. Small n still uses the prespecified Monte
    Carlo estimator; the returned p-value is not labelled an exact test.
    """
    delta = _finite_vector(differences, "differences")
    if int(permutations) != permutations or permutations < 100:
        raise ValueError("permutations must be an integer >= 100")
    observed = abs(delta.mean())
    if observed == 0 and np.all(delta == 0):
        return 1.0
    rng, extreme = np.random.default_rng(seed), 0
    for start in range(0, int(permutations), 1000):
        count = min(1000, int(permutations) - start)
        signs = 2 * rng.integers(0, 2, size=(count, len(delta))) - 1
        simulated = np.abs((signs * delta).mean(axis=1))
        extreme += int(np.count_nonzero(simulated >= observed - 1e-14))
    return float((extreme + 1) / (int(permutations) + 1))


def holm_adjust(pvalues):
    """Holm step-down family-wise adjustment, returned in input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("pvalues must be a finite 1-D vector in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilson_rate_ci(events, n_patients):
    """95% binomial Wilson score interval, including zero/all-event samples."""
    if n_patients < 1 or not 0 <= events <= n_patients:
        raise ValueError("Events must be between 0 and a positive patient count")
    z = 1.959963984540054
    rate = events / n_patients
    denominator = 1 + z * z / n_patients
    centre = (rate + z * z / (2 * n_patients)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n_patients + z * z / (4 * n_patients ** 2)) / denominator
    return float(rate), float(max(0, centre - half)), float(min(1, centre + half))


def _read_final(family_dir, expected_patients=54):
    family_dir = Path(family_dir)
    family = family_dir.name
    meta_path = family_dir / "metadata.json"
    if not meta_path.is_file():
        raise ValueError(f"Missing ensemble provenance: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("aggregation") != "probability_mean" or metadata.get("n_checkpoints") != 15:
        raise ValueError(f"{family}: final metrics must follow probability-mean ensemble of 15 checkpoints")
    seeds, folds = metadata.get("seeds", []), metadata.get("folds", [])
    if len(set(seeds)) != 3 or len(seeds) != 3 or len(set(folds)) != 5 or len(folds) != 5:
        raise ValueError(f"{family}: ensemble provenance must list three unique seeds and five unique folds")
    if metadata.get("threshold") != 0.5:
        raise ValueError(f"{family}: frozen primary threshold must be 0.5")
    for key in ("manifest_sha256", "freeze_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(metadata.get(key, ""))):
            raise ValueError(f"{family}: missing or malformed {key}")

    df = pd.read_csv(family_dir / "test_metrics.csv", dtype={"patient_id": str})
    required = {"patient_id", *METRIC_NAMES, "empty_prediction", "empty_ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{family}: missing columns {sorted(missing)}")
    if df.patient_id.isna().any() or (df.patient_id.str.strip() == "").any():
        raise ValueError(f"{family}: missing patient IDs")
    if df.patient_id.duplicated().any():
        raise ValueError(f"{family}: repeated patient IDs are not independent observations. "
                         "Ensemble probability maps first; do not average fold/seed metric rows.")
    if expected_patients is not None and len(df) != expected_patients:
        raise ValueError(f"{family}: expected {expected_patients} locked holdout patients, found {len(df)}")
    ids = metadata.get("holdout_patient_ids", [])
    if len(ids) != len(set(map(str, ids))) or set(map(str, ids)) != set(df.patient_id):
        raise ValueError(f"{family}: metric patient IDs differ from locked ensemble metadata")
    if "family" in df and set(df.family.astype(str)) != {family}:
        raise ValueError(f"{family}: family column disagrees with result directory")
    for metric in METRIC_NAMES:
        values = pd.to_numeric(df[metric], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{family}: non-finite or negative {metric}")
        if metric != "component_count_error" and (values > 1 + 1e-12).any():
            raise ValueError(f"{family}: {metric} outside [0,1]")
        if metric == "component_count_error" and not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{family}: component_count_error must be an integer")
        df[metric] = values
    for flag in ("empty_prediction", "empty_ground_truth"):
        vals = pd.to_numeric(df[flag], errors="raise")
        if not vals.isin((0, 1)).all():
            raise ValueError(f"{family}: {flag} must be 0/1")
        df[flag] = vals.astype(int)
    if df.empty_ground_truth.any():
        raise ValueError(f"{family}: unexpected empty GT in positive-only cohort; audit before analysis")
    for optional in ("corrupt_dice_mean", "uncertainty_mean"):
        if optional in df:
            df[optional] = pd.to_numeric(df[optional], errors="raise")
            if not np.isfinite(df[optional]).all() or (df[optional] < 0).any():
                raise ValueError(f"{family}: invalid {optional}")
            if optional == "corrupt_dice_mean" and (df[optional] > 1).any():
                raise ValueError(f"{family}: invalid corrupt Dice")
    return df.sort_values("patient_id").reset_index(drop=True), metadata


def _binary_meta(series, name):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalised = series.astype(str).str.lower()
    if not normalised.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"{name} must have unambiguous binary values")
    return normalised.isin(("true", "1"))


def _descriptive(df, family, bootstrap, seed, subgroup="all", subgroup_value="all"):
    rows = []
    metrics = [*METRIC_NAMES, "failure_dice_below_0_5", "empty_prediction_rate"]
    metrics += [c for c in ("corrupt_dice_mean", "uncertainty_mean") if c in df]
    derived = df.assign(failure_dice_below_0_5=(df.dice < 0.5).astype(float),
                        empty_prediction_rate=df.empty_prediction.astype(float))
    for metric in metrics:
        values = derived[metric].to_numpy(dtype=float)
        is_rate = metric in ("failure_dice_below_0_5", "empty_prediction_rate")
        events = int(values.sum()) if is_rate else None
        if is_rate:
            mean, lo, hi = wilson_rate_ci(events, len(df))
        else:
            mean, lo, hi = bootstrap_mean_ci(values, bootstrap, seed)
        rows.append({"family": family, "subgroup": subgroup, "subgroup_value": subgroup_value,
                     "metric": metric, "n_patients": len(df), "mean": mean,
                     "ci95_lower": lo, "ci95_upper": hi, "median": float(np.median(values)),
                     "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                     "min": float(values.min()), "max": float(values.max()),
                     "n_events": events, "analysis_role": "descriptive",
                     "ci_method": "wilson_score" if is_rate else "patient_percentile_bootstrap",
                     "small_subgroup": len(df) < 10})
    return rows


def _paired_plot(proposed, unet, comparisons, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    baseline, method = unet.dice.to_numpy(), proposed.dice.to_numpy()
    # All patients shown; technical failure flags never exclude a difficult case.
    for b, p in zip(baseline, method):
        axes[0].plot((0, 1), (b, p), color="#8896a4", alpha=0.30, lw=0.7)
    axes[0].scatter(np.zeros(len(baseline)), baseline, s=12, color="#4679a4", zorder=3)
    axes[0].scatter(np.ones(len(method)), method, s=12, color="#d06c3f", zorder=3)
    axes[0].set(xticks=(0, 1), xticklabels=("U-Net", "Proposed"), ylabel="Patient Dice",
                ylim=(-0.02, 1.02), title=f"Locked internal holdout (n={len(method)})")
    axes[0].set_xlim(-0.3, 1.3)
    table = comparisons.reset_index(drop=True)
    y = np.arange(len(table))
    delta = table.mean_delta_dice.to_numpy()
    error = np.array((delta - table.ci95_lower, table.ci95_upper - delta))
    axes[1].errorbar(delta, y, xerr=np.maximum(error, 0), fmt="o", color="#d06c3f", capsize=3)
    axes[1].axvline(0, color="#333333", lw=0.8, linestyle="--")
    axes[1].set(yticks=y, yticklabels=table.baseline.tolist(), xlabel="Mean paired Dice difference",
                title="Proposed minus baseline; bootstrap 95% CI")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"paired_dice.{extension}", dpi=300)
    plt.close(fig)


def analyze(output_root, bootstrap=10000, seed=20261009):
    """Write reviewable tables/figures for real final outputs; return a summary.

    No fallback to development validation is allowed. Missing final results,
    missing provenance, mismatched test IDs, and duplicate patients raise errors.
    Optional subgroup fields are n_polygons (annotation proxy, not eye count)
    and has_dng (paired-RAW availability/acquisition-period proxy). Neither can
    establish a causal subgroup effect or separate camera and time confounding.
    """
    output_root = Path(output_root)
    paths = sorted((output_root / "final").glob("*/test_metrics.csv"))
    if not paths:
        raise FileNotFoundError("No final/*/test_metrics.csv: complete frozen holdout evaluation first")
    models, provenance = {}, {}
    for path in paths:
        df, metadata = _read_final(path.parent)
        models[path.parent.name], provenance[path.parent.name] = df, metadata
    if not {"proposed", "unet"}.issubset(models):
        raise ValueError("The prespecified primary comparison requires proposed and unet results")
    reference = models["proposed"]
    for family, df in models.items():
        if df.patient_id.tolist() != reference.patient_id.tolist():
            raise ValueError(f"{family}: patient IDs do not match proposed paired holdout set")
        for key in ("manifest_sha256", "freeze_sha256", "seeds", "folds"):
            if provenance[family][key] != provenance["proposed"][key]:
                raise ValueError(f"{family}: shared evaluation provenance differs in {key}")
        for col in ("n_polygons", "has_dng"):
            if col in df and col in reference and df[col].astype(str).tolist() != reference[col].astype(str).tolist():
                raise ValueError(f"{family}: patient subgroup metadata differs in {col}")

    baselines = ["unet"] + [m for m in sorted(models) if m not in ("unet", "proposed")]
    comparisons = []
    for i, baseline in enumerate(baselines):
        row = paired_bootstrap(reference.dice, models[baseline].dice, bootstrap, seed + i)
        row.update({"proposed": "proposed", "baseline": baseline,
                    "p_sign_flip_two_sided": sign_flip_pvalue(
                        reference.dice.to_numpy() - models[baseline].dice.to_numpy(), 20000, seed + i),
                    "comparison_role": "primary" if baseline == "unet" else "secondary",
                    "permutations": 20000, "bootstrap_replicates": int(bootstrap)})
        comparisons.append(row)
    compare_df = pd.DataFrame(comparisons)
    compare_df["p_holm_all_baselines"] = holm_adjust(compare_df.p_sign_flip_two_sided)

    descriptive, subgroup_rows, failure_rows = [], [], []
    missing_subgroups = []
    for family, df in models.items():
        descriptive.extend(_descriptive(df, family, bootstrap, seed))
        if "n_polygons" in df:
            counts = pd.to_numeric(df.n_polygons, errors="raise")
            if ((counts < 1) | (counts != np.floor(counts))).any():
                raise ValueError(f"{family}: invalid n_polygons")
            groups = counts.map(lambda v: "one_polygon" if v == 1 else "two_polygons" if v == 2 else "three_or_more_polygons")
            for group in sorted(groups.unique()):
                subgroup_rows.extend(_descriptive(df.loc[groups == group], family, bootstrap, seed,
                                                   "annotation_polygon_count_proxy", group))
        else:
            missing_subgroups.append(f"{family}: n_polygons")
        if "has_dng" in df:
            groups = _binary_meta(df.has_dng, "has_dng")
            for group in sorted(groups.unique()):
                subgroup_rows.extend(_descriptive(df.loc[groups == group], family, bootstrap, seed,
                                                   "dng_availability_proxy", str(bool(group)).lower()))
        else:
            missing_subgroups.append(f"{family}: has_dng")
        failing = (df.dice < 0.5) | (df.empty_prediction == 1) | (df.gt_component_recall_0_5 < 1)
        for _, row in df.loc[failing].iterrows():
            failure_rows.append({"family": family, "patient_id": row.patient_id,
                                 "dice": row.dice, "empty_prediction": row.empty_prediction,
                                 "gt_component_recall_0_5": row.gt_component_recall_0_5,
                                 "dice_below_0_5": int(row.dice < 0.5)})

    analysis = output_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(descriptive).to_csv(analysis / "metric_summary.csv", index=False)
    compare_df.to_csv(analysis / "paired_dice_comparisons.csv", index=False)
    subgroup_columns = [*pd.DataFrame(descriptive).columns]
    pd.DataFrame(subgroup_rows, columns=subgroup_columns).to_csv(analysis / "subgroup_summary.csv", index=False)
    pd.DataFrame(failure_rows, columns=["family", "patient_id", "dice", "empty_prediction",
                                      "gt_component_recall_0_5", "dice_below_0_5"]).to_csv(
        analysis / "failure_cases.csv", index=False)
    _paired_plot(reference, models["unet"], compare_df, analysis / "figures")

    primary = compare_df.loc[compare_df.baseline == "unet"].iloc[0].to_dict()
    summary = {"status": "ANALYZED", "validation_type": "locked_internal_holdout",
               "n_patients": len(reference), "primary_endpoint": "patient_macro_dice",
               "primary_comparison": primary, "families": list(models),
               "bootstrap_replicates": int(bootstrap), "permutations": 20000, "seed": int(seed),
               "unit_of_resampling": "patient", "missing_subgroup_fields": missing_subgroups,
               "failure_definition": "technical Dice < 0.5; not a clinical acceptability threshold",
               "manifest_sha256": provenance["proposed"]["manifest_sha256"],
               "freeze_sha256": provenance["proposed"]["freeze_sha256"],
               "input_file_sha256": {str(p.relative_to(output_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in paths},
               "notes": ["Single prespecified primary comparison uses the raw two-sided p value.",
                         "Holm-adjusted p values across all baseline Dice comparisons are supplied alongside raw p values.",
                         "Secondary metrics and subgroup analyses are descriptive; no clinical inference is made.",
                         "DNG availability may confound acquisition period and device; raw files are not extra patients.",
                         "Polygon count is an annotation proxy and does not establish anatomical eye count.",
                         "Continuous metric percentile CIs for small subgroups can be degenerate; failure rates use Wilson score CIs and event counts.",
                         "Inference conditions on trained ensembles and does not include training-sample uncertainty."]}
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = (
        "## Material Passport\n\n"
        "- Artifact: locked internal holdout statistical analysis\n"
        "- Source: final probability-ensemble test metrics and provenance\n"
        "- Verification status: ANALYZED; model training was not independently reproduced here\n"
        "- Independent unit: one patient / one original photo\n\n"
        f"Primary endpoint: patient-macro Dice. Proposed minus U-Net mean difference = "
        f"{primary['mean_delta_dice']:.6f}; paired bootstrap 95% CI "
        f"[{primary['ci95_lower']:.6f}, {primary['ci95_upper']:.6f}]; "
        f"two-sided sign-flip p = {primary['p_sign_flip_two_sided']:.6g}.\n\n"
        f"All {len(reference)} held-out patients were evaluated once after probability averaging "
        "15 checkpoints per family. Training seeds, folds, both eyes and pixels are never counted "
        "as independent patients. The confidence intervals condition on these fitted ensembles.\n\n"
        "The raw p value addresses the single prespecified primary comparison. "
        "paired_dice_comparisons.csv also supplies Holm-adjusted p values across all baseline "
        "Dice comparisons. Secondary metrics, subgroup estimates, corruption scores, uncertainty "
        "and failure summaries are descriptive. Dice < 0.5 is a technical failure summary threshold "
        "and has no established clinical interpretation.\n\n"
        "This cohort supports internal validation. It does not establish external generalisation, "
        "clinical benefit, performance in healthy populations, or independent annotation reliability. "
        "DNG availability is an acquisition proxy; polygon count is an annotation proxy. "
        "Small-subgroup continuous-metric bootstrap intervals can be degenerate; "
        "technical failure rates use Wilson score 95% CIs with event counts.\n\n"
        "Metrics definitions, empty-mask policies and boundary-distance approximations are documented "
        "in metrics.py. Full paired results and every technical failure are retained in CSV tables.\n"
    )
    (analysis / "report.md").write_text(report, encoding="utf-8")
    return summary
