# Dataset

## Paired benchmark

The manifest contains 3,000 paired samples, evenly divided between Real-HQ, Real-LQ, and AI-Generated sources. Each row identifies the subtitled input, clean ground truth, subtitle glyph mask, and bounding-box annotation.

The 953 auxiliary videos mentioned in the paper are separate from these 3,000 paired samples. They are not counted in the evaluation lists or included in this prepared repository.

Video payloads are hosted on [Hugging Face](https://huggingface.co/datasets/siiian/MFVSR). Follow the dataset page to download, verify, and extract the archives.

## Directory layout

Preserve the relative paths in manifests/benchmark_manifest.json when extracting the dataset:

~~~text
MFVSR/
  data_select/
    AI_gen/
    origin_clean/
  subtitled_data/
    AI_gen/
      bilingual/
      monolingual/<language>/
    origin_clean/
      ...
~~~

The manifest is the authoritative path mapping. Do not reconstruct paths from split alone: some monolingual AI samples have an additional language directory.

## Manifest fields

| Field | Meaning |
|---|---|
| video_id | Unique paired sample identifier |
| split | Existing source/layout folder label; not a train/validation partition |
| input_video_path | Subtitled input video |
| gt_video_path | Clean target video |
| glyph_mask_path | Video mask for subtitle glyph pixels |
| bbox_json_path | Per-frame bounding-box annotations |
| bbox_mask_video_path | Bounding-box mask video used by mask-guided pipelines |
| mask_frame_root | Directory of per-frame mask images |

Paths are relative to the extracted dataset root. scripts/configure.py resolves them into a machine-local manifest without changing the public file.

The evaluator reads the input, target, glyph mask, and bounding-box JSON. The bbox mask video and mask frames are provided for preparing mask-guided predictions; inference code is not part of this release.

## Metadata

manifests/sample_metadata.json is the metadata used by the server evaluation. sample_metadata.csv provides a readable view, including subtitle languages. Metadata includes source, resolution, duration, layout, animation, visual style, language mode, difficulty tier, and box-style status.

The JSON also contains cached ground-truth file size and modification-time fingerprints from its original generation. These are provenance fields, not portable file checksums.

The fixed evaluation lists and their roles are described in [protocol.md](protocol.md).
