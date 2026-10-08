# Evaluation protocol

## Fixed sample lists

| File under manifests/ | Count | Use |
|---|---:|---|
| lite3000.txt | 3,000 | Evaluate the complete paired benchmark |
| audit300.txt | 300 | Frozen non-box audit subset |
| primary_nonbox.txt | 2,499 | Main non-box sample pool |
| box_pool.txt | 501 | Complete box-style stress pool |
| paper_lite_common_nonbox.txt | 2,298 | Six-baseline common evaluable non-box snapshot |
| paper_lite_strict_complete_nonbox.txt | 80 | Six-baseline fully complete non-box snapshot |
| paper_lite_common_all_styles.txt | 2,762 | Six-baseline common evaluable snapshot including box styles |
| paper_box_common.txt | 464 | Six-baseline common box-style snapshot |

Audit300 was copied from the original ordered selection file. Its SHA-256 is:

~~~text
7b33b4368c03c359aa02a51179bebc6bc7540360bbf1586fcbd6277cfe28c95f
~~~

audit300_selection.csv and audit300_selection_report.json document the selection. A machine-specific inventory-file path was removed from the report copy.

The paper_common lists describe the archived six-method results. They are not substitutes for the full fixed lists when evaluating a new method. With new methods, report their native support and the common intersection for the methods being compared.

## Metrics and sampling

Both profiles use the original stratified fixed-budget sampler: up to 32 subtitle-active plus 16 global frame draws, and 24 subtitle-active plus 8 global pair draws. The actual number of unique usable frames or pairs can be lower.

| Profile | Metrics |
|---|---|
| Lite3000 | PSNR, LPIPS, and flow-free temporal change (TC) |
| Audit300 | PSNR, SSIM, LPIPS, DISTS, TC, and flow-based TWE/TWE-Excess |

Spatial regions include overall, subtitle glyph, subtitle bounding box, background inside the bounding box, and outside the bounding box. The region matrix in configs/evaluation.example.json determines which metrics run in each region.

The temporal flow backend is RAFT-Large using clean-target motion and forward/backward consistency filtering. TWE-Excess and TC are reported for bbox and outside regions in the profile configuration. DISTS uses the original same-bbox context and conditioned-region settings.

Higher PSNR/SSIM is better; lower LPIPS/DISTS/TC/TWE-Excess is better. The bbox background region is a mixed diagnostic for box-style examples, where removing the box can differ from preserving its background. Interpret it alongside the dedicated box-style pool.

## Support and failures

Report quality alongside temporal/spatial coverage and failure rates. A common evaluable video may still have an undefined value for a specific metric, so use matched finite support for direct metric comparisons.

manifests/metric_support contains ten matched finite-support Audit300 lists derived from the six archived baselines. Nine contain 300 IDs; audit_bbox_twe_excess.txt contains 299. These support lists document the existing snapshot and should be recomputed for new method comparisons.

Do not replace failed evaluations, missing predictions, or undefined metrics with zero. Do not silently enlarge Audit300 or change its selection seed. The provided primary release tables use individual metrics and reliability summaries; a fitted aggregate score is not required.
