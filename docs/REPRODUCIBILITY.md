# Reproducibility scope

## Executable analyses

`python reproduce.py` verifies the release and runs scalar and synthetic checks. It recomputes integer-overlap Dice for all 212 public source records and 15 procedures, the 15 mean confidence intervals, all eight final paired contrasts, event transitions and the five synthetic reconstruction conditions. It uses the original record order, random seed, function bodies and floating-point operation order.

The result family matches manuscript Table 4. Public mean Dice and failure counts map to Table 3. The other public and internal tables are supplied as manuscript exports; their inclusion is not a claim that every table is newly recomputed by this entry point.

## Figure reproduction

`python reproduce.py --figures` also redraws Figures 1–5. Figure 3 uses the supplied aggregate internal and public means/intervals. Figure 4 computes its descriptive subgroup intervals from all 212 records, using seeds 20261013 for the 125-record whole-image group and 20261012 for the 87-record partial-proposal group. These subgroup seeds differ from the main comparison seed by design. Figure 5 recomputes four full-canvas Dice values per method from the saved binary masks and uses the original four-case selection. The display crop is shared across methods within a row and is not used to calculate Dice.

Fonts may differ across platforms; Arial is preferred, with DejaVu Sans as fallback. The submitted PNG/PDF files are retained in `figures/`. Scientific quantities must agree; byte-identical vector files are not expected across plotting-library versions.

## Interpretation and boundaries

Resampling describes source-record uncertainty conditional on the fitted checkpoints. Public source records are treated as independent evaluation units; unique patient and acquisition-batch mappings were not independently established. The intervals and tests do not model adaptive analysis selection or repeated model-training variability.

The synthetic reference accepts already mapped arrays. It does not validate image interpolation or network inference. The 125 real probability identities remain server-recorded checks; complete probability arrays are not reread by the scalar replay. The four sets of Figure 5 binary masks can be checked directly, but their saved probabilities and model weights are not in this release.

The original Linux inference environment is documented in `software/inference_environment.json`. The release validation receipt separately states the actual environment used for this replay. No CUDA installation or environment management is included.
