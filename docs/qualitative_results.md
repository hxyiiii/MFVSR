# Qualitative results

The README displays three diagnostic cases from Appendix E of the final submitted paper. The input, clean target, and six-method images were extracted directly from the embedded PDF images at native resolution.

## Cases

| Case | Paper page | View | Frames |
| :--- | ---: | :--- | :--- |
| R1: Shadowed bilingual subtitles | 26 | Local restoration crop | 0 |
| P1: Abstract geometry | 28 | Full-frame preservation view | 47 |
| T1: Bouncing neon | 30 | Shared temporal crop; three-frame strip | 0, 1, 2 |

Each case includes Subtitled input, Clean target, CLEAR, ProPainter, DiffuEraser, MiniMax-Remover, OmnimatteZero, and EffectErase. CLEAR uses blind input; the remaining methods use BBox-derived masks.

The paper selects cases using inter-method variation in the corresponding regional metric. These examples illustrate diagnostic differences rather than the frequency of failure across the benchmark.

## Image handling

Images in imgs/qualitative retain the PDF-embedded pixels and aspect ratios. The README changes only their presentation order and displayed size. Opening an image shows its native resolution. There is no retouching, sharpening, frame interpolation, or generated replacement content.

The temporal strips preserve their original left-to-right frame ordering. They are static consecutive-frame comparisons, not video playback.

The complete quantitative results remain in the two result folders and [detailed results documentation](results.md).
