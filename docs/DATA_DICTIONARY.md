# Data dictionary

`software/data/casewise_metrics.csv` has 3,180 rows: 212 source records × 15 procedures. `case_key` is a stable coded public-source key, not a verified unique-patient identifier. `method` names the archived inference procedure. `country` is the dataset source country. `source_encoding`, `source_reference` and `reference_grid` document reference decoding; they are not independent reannotations.

| Field | Definition / unit |
|---|---|
| `tp_pixels`, `fp_pixels`, `fn_pixels` | Integer foreground overlap counts on the reference canvas |
| `dice`, `iou`, `precision`, `recall` | Dimensionless overlap metrics |
| `foreground_gt_pixels`, `foreground_pred_pixels` | Reference and prediction foreground pixel counts |
| `empty_prediction`, `empty_ground_truth` | Binary flags: 1 means empty |
| `surface_dice_0_005`, `surface_dice_0_01` | Boundary-pixel surface Dice at fractions of the valid-image diagonal |
| `hd95_norm`, `assd_norm` | Boundary distances divided by the valid-image diagonal |
| `empty_distance_penalty` | Flag for the assigned empty-mask distance penalty |
| `gt_components`, `pred_components`, `component_count_error` | Connected-component counts and absolute count error |
| `gt_component_recall_0_5` | Fraction of reference components with at least 50% foreground coverage |
| `threshold` | Foreground probability threshold, 0.5 |

`software/data/three_path_casewise.csv` has one row per public record. Prefixes `original_`, `corrected_` and `direct_` identify the three combined-model inference paths. `single_fullframe_folds` is an integer from 0 to 5; `all_five_single_fullframe` is 1 exactly when the value is 5. Empty predictions against a nonempty reference have Dice 0. Precision is defined as 1 for empty predictions, as described in the manuscript.

`software/data/reconstruction_receipts.json` retains native image sizes, all member proposal boxes and archived identity checks. Missing identity checks for ineligible records are JSON null, not zeros. Coordinates use exclusive right and bottom endpoints.

`data/figure_inputs/figure_1_source.csv` contains 16 aggregate rows, eight per dataset. `n` is 54 internal patients or 212 public source records. Blank internal failure counts indicate unavailable values in that summary, not zero. Other figure input tables preserve raw Dice differences, whereas manuscript Table 4 and Figure 4a display percentage points (100 × raw difference).

`data/qualitative/selection.json` identifies the four original public source records, selection subgroup, subgroup size, median-selection field and expected full-canvas Dice. All images and masks have width 800 and height 1067. PNG masks use 0 for background and nonzero for foreground. Source-record names are the public dataset labels; they are not institutional patient identifiers.
