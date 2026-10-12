# Scalar analysis and reconstruction reference

Run `python run_reproduce.py` in this directory, or `python reproduce.py` from the repository root. Existing Python >=3.10 and NumPy are sufficient; no installation, downloads or GPU operations occur.

The runner verifies its input manifest before importing the archived statistics functions. It recomputes 3,180 Dice values from TP/FP/FN, 15 mean intervals, the complete final family of eight paired contrasts, casewise declines and technical-event transitions. Five mapped-array scenarios compare the portable reconstruction reference with actual function bodies extracted from the archived server source: one whole-image box, partial box, overlapping boxes with summed weights above one, near-whole-image box and no fine regions.

`configuration.json` preserves procedure order, source-record order and the analysis seed. Bootstrap and sign-flip counts are 10,000 and 20,000; Holm adjustment includes all eight contrasts. The random generator is `np.random.default_rng(20261011)`, recreated for each vector/test; draws occur in blocks of 1,000. The bootstrap uses NumPy's linear percentile convention. Sign-flip p values use add-one correction.

`data/reconstruction_receipts.json` preserves proposal geometry and server-recorded identity checks. The 125 eligible records have a recorded zero probability difference; the other 87 retain null identity checks. The runner checks these receipts against geometry and integer overlap, but does not reconstruct the original full probability arrays. Boundary metrics are archived scalars; they cannot be recomputed for all records without masks.

The mathematical reference operates on arrays already mapped to a common canvas. The synthetic adapter deliberately performs no interpolation. Actual image resizing, network inference and training are documented in `frozen_source/`; these snapshots depend on the larger original project. Historical schema labels and source identifiers identify the unchanged analysis implementation.

`formal_training_configuration.json`, `protocol_settings.json` and `inference_environment.json` document the actual 40-fit study. They contain configuration and hash bindings, not institutional image identities or checkpoints. Original server and local replay environments are distinct and are recorded separately.

See `../docs/REPRODUCIBILITY.md` and `../docs/ACCESS.md` for result mapping and access conditions.
