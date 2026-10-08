# Archived results

Two result folders correspond to the final server evaluation:

- results/lite3000: 18,000 rows, six methods × 3,000 samples.
- results/audit300: 1,800 rows, six methods × 300 fixed samples.

Methods are clear, propainter, minimax, omnimattezero, effecterase, and diffueraser. clear is blind; the remaining methods use the mask-guided setting in the archived experiment.

## Files

| File | Contents |
|---|---|
| per_sample_metrics.csv | Individual sample metrics, coverage, and statuses |
| quality_common_and_native.csv | Aggregated quality, common/native support, dimensions, and uncertainty |
| coverage_common_and_native.csv | Coverage under corresponding comparison scopes |
| reliability.csv | Overall success/failure and incompleteness |
| reliability_by_dimension.csv | Reliability stratified by dataset attributes |
| dataset_dimensions.csv | Counts for the analyzed dataset |
| intersection_retention.csv | How common-support selection changes attribute retention |
| analysis_summary.json | Compact counts and record-audit summary |

Six Lite3000 native evaluable counts are:

| Method | Evaluable samples |
|---|---:|
| clear | 2,999 |
| propainter | 2,949 |
| minimax | 2,871 |
| omnimattezero | 2,885 |
| effecterase | 2,927 |
| diffueraser | 2,970 |

All six methods have 300 evaluable Audit300 samples. Metric-specific finite support can still differ, as documented in [protocol.md](protocol.md).

The Lite3000 main common non-box support is 2,298. The 2,762 all-style intersection and 464 common box-style samples are separate scopes; do not use them interchangeably.

## Export handling

Aggregate CSVs were copied byte-for-byte. Per-sample CSVs preserve every retained cell as exported by the server, with seven machine-specific path columns removed:

- record_path
- frame_record_path
- input.prediction.path
- input.ground_truth.path
- input.glyph_mask.path
- input.bbox_json.path
- input.inference_info.path

No metric values were rounded, recomputed, or filled. Analysis summaries omit optional fitted-score fields. The original server files remain unchanged.

Per-sample exports are standalone result tables. They do not contain the complete JSON/frame records required to rerun the analysis command; new evaluation produces those runtime records.

## Audit300 detailed tables

The following baseline results use the frozen **Audit300** subset. Quality is measured on aligned evaluable output, with joint spatial/temporal coverage reported separately. **Bold** and *italics* mark the best and second-best means among mask-guided methods; blind results are shown in their own block.

### 🖼️ Restoration and Preservation

Restoration metrics use the **Subtitle** region; preservation metrics use the **Outside** region. PSNR is in dB. cDISTS denotes conditioned DISTS, and joint coverage is a percentage.

#### Blind

| Method | Subtitle PSNR ↑ | Subtitle LPIPS ↓ | Subtitle cDISTS ↓ | Outside PSNR ↑ | Outside LPIPS ↓ | Joint cov. (%) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLEAR | 11.99 | 0.621 | 0.236 | 21.48 | 0.221 | 85.4 |

#### Mask-guided

| Method | Subtitle PSNR ↑ | Subtitle LPIPS ↓ | Subtitle cDISTS ↓ | Outside PSNR ↑ | Outside LPIPS ↓ | Joint cov. (%) ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ProPainter | 19.66 | 0.364 | 0.138 | **34.11** | **0.050** | 100.0 |
| DiffuEraser | **21.36** | *0.269* | **0.116** | *32.87* | *0.062* | 99.9 |
| MiniMax-Remover | 20.32 | 0.321 | *0.132* | 30.73 | 0.086 | 99.4 |
| OmnimatteZero | 17.69 | 0.369 | 0.160 | 26.00 | 0.131 | 98.5 |
| EffectErase* | *20.58* | **0.251** | 0.138 | 25.77 | 0.120 | 43.1 |

### ⏱️ Temporal Fidelity

| Setting | Method | BBox TC ↓ | Outside TC ↓ | BBox TWE-X ↓ | Outside TWE-X ↓ |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Blind | CLEAR | 5.00 | 1.77 | 15.042 | 0.076 |
| Mask-guided | ProPainter | 2.57 | **0.70** | 0.683 | 0.072 |
| Mask-guided | DiffuEraser | 2.53 | *0.73* | 0.686 | 0.068 |
| Mask-guided | MiniMax-Remover | *2.52* | 1.12 | 1.142 | **0.028** |
| Mask-guided | OmnimatteZero | 2.77 | 1.51 | *0.653* | *0.048* |
| Mask-guided | EffectErase* | **2.39** | 1.51 | **0.634** | 0.129 |

TC values are multiplied by 100; TWE-X (TWE-Excess) values are multiplied by 1,000. BBox TWE-X uses the same 299 finite-support videos for all six methods; the other metrics above use 300.

*EffectErase is evaluated on its verified prefix. Its lower coverage should be considered alongside its conditional quality.*

The [Lite3000 CSVs](../results/lite3000/) contain the full-set results, including failures and partial outputs. The [Audit300 CSVs](../results/audit300/) provide detailed metrics and confidence intervals. See [result documentation](./results.md) for file descriptions.
