# Evaluation

## Conda environment

Create the mfvsr environment using conda-forge. The same shared dependencies are listed in environment.yml:

~~~bash
conda create -n mfvsr python=3.10 pip "setuptools>=68" ffmpeg \
  --override-channels -c conda-forge -y
conda activate mfvsr
~~~

Install a PyTorch/torchvision build matching your hardware, then run python -m pip install -e . from the repository root. The README's [Conda setup](../README.md) provides NVIDIA, AMD/ROCm, CPU, and macOS instructions.

environment.yml contains the shared system dependencies. PyTorch is installed separately because CUDA, ROCm, CPU, and macOS require different builds. The remaining Python dependencies are declared in pyproject.toml and installed with the editable package.


## Prediction interface

For a sample with the input path:

~~~text
subtitled_data/AI_gen/monolingual/Persian/<video_id>/<video_id>.mp4
~~~

The recommended output layout is:

~~~text
predictions/your_method/
  AI_gen/monolingual/Persian/<video_id>/
    info.json
    <video_id>_cleaned.mp4
~~~

The evaluator also accepts an additional <video_id> directory containing both files, matching existing server outputs.

A minimal info.json is:

~~~json
{
  "status": "success"
}
~~~

With no explicit output path, the evaluator reads <video_id>_cleaned.mp4 next to info.json. To use another filename or location, add output_raw.video_path with an absolute path:

~~~json
{
  "status": "success",
  "output_raw": {
    "video_path": "/absolute/path/to/prediction.mp4"
  }
}
~~~

A video without a valid successful info.json is not treated as a successful prediction. Record genuine inference failures with a non-success status rather than fabricating a cleaned output.

## Configuration

Use scripts/configure.py as shown in the repository README. It changes only paths and the user method entry; the public metric and sampling settings come from configs/evaluation.example.json.

Choose:

- setting: blind or mask-guided, according to your method's inputs.
- geometry: full_frame_resize_to_gt for spatially aligned full-frame predictions; clear_center_crop_inverse only for the corresponding center-crop preprocessing.
- temporal: exact for equal frame counts; verified_prefix for predictions that are a correctly aligned prefix of the target.

The prefix policy evaluates the available aligned frames and reports coverage. It does not pad missing frames. The configured FPS tolerance is 0.01. Declaring an alignment policy does not validate the semantics of a model's preprocessing.

Relative output and configuration paths in configure.py are resolved from the repository root. It refuses to overwrite existing runtime files unless --overwrite is supplied. Use another --config-out and --output-root for separate runs.

For multiple methods, add entries to the generated config's models dictionary, each with its own result_root and alignment policy. Then use --model all, or a comma-separated list of method names, for both run and analyze. Their geometry can differ while metric settings remain shared.

## Evaluation and analysis

Always pass the same profile and ID list to run and analyze. Lite3000 uses manifests/lite3000.txt; Audit300 uses manifests/audit300.txt. The profile alone does not select the fixed sample list.

doctor checks executable availability, dependencies, and configuration. With --deep, it also initializes the metric backends and weights. Run's --strict flag returns an error if an evaluation task fails.

Analysis requires current terminal evaluation records by default. Do not bypass that requirement to silently aggregate missing or stale results. Re-evaluate affected samples when paths, inputs, alignment settings, or metric settings change.

Outputs include per-sample JSON records, optional frame-level Parquet records, and aggregate CSVs. --output-dir controls the analysis output folder.

For a small functional check, select an existing video_id using run's --video-id option. An analysis restricted to that sample should receive an ID file containing the same single ID. Such a check is not a full Lite3000 or Audit300 result.

Baseline prediction videos, model weights, and inference configurations are not bundled. The six-baseline CSV snapshot can be inspected without rerunning inference.
