# Figure and table map

| Manuscript item | Source / entry point |
|---|---|
| Figure 1: inference paths | `scripts/plot_figures_1_to_4.py`, deterministic schematic |
| Figure 2: reconstruction mechanism | Same script, `data/figure_inputs/synthetic_source.npz` |
| Figure 3: internal and public performance | Same script, `data/figure_inputs/figure_1_source.csv` |
| Figure 4: paired effects and subgroups | Same script, final contrasts and all 212 rows in `software/data/three_path_casewise.csv` |
| Figure 5: real-image comparisons | `scripts/plot_figure_5.py`, `data/qualitative/` |
| Tables 1–4 | `data/manuscript_tables/table_1.csv` through `table_4.csv` |
| Tables S1–S5 | `data/manuscript_tables/table_S1.csv` through `table_S5.csv` |

Figure 5 retains all four preselected records. A: whole-image improvement; B: partial-proposal residual; C: whole-image decline; D: empty cascade predictions. Selection used the source record nearest the specified subgroup median, with lexicographic source order resolving ties. The figure displays source reference contours in cyan dashes and prediction contours in solid yellow. Cropping is for display only; no brightness, contrast, gamma or color adjustment is applied.

Full figure captions are in `data/figure_captions.txt`. Reference labels within those captions refer to the accompanying manuscript. The source dataset attribution in Figure 5 and NOTICE.md must be retained.
