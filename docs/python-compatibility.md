# Python 3.14 feasibility experiment

This draft is for evaluation. It is kept separate from `main` and `personal` and must not be merged as part of this work. Its base is the personal CUDA/model-download integration, commit `6e775f392e1c3e208e28bb544e9a21c7cc67e3a1`; those prerequisites are separate from the port. A future upstream contribution would need rebasing/splitting once those prerequisites are handled upstream.

A compact [validation receipt](python314-validation.json) contains the exact tested versions, hashes, settings, paired metrics and qualified timing/memory results. Raw arrays, environment directories, weights and logs are excluded from Git.

## Technical profile

The Python 3.11 installer profile remains unchanged. The additional profile targets ordinary CPython 3.14 on Windows x64. It rejects 3.14 on Linux/macOS, Windows ARM64, 32-bit interpreters, non-CPython implementations and free-threaded builds (`3.14t`). No broader platform support is claimed.

| Dependency | Python 3.11 | Python 3.14 on Windows x64 |
| --- | --- | --- |
| PyTorch / torchvision | 2.5 / 0.20 | 2.9.1 / 0.24.1 |
| NVIDIA runtime selected from driver capability | Existing profile retained | CUDA 12.6 or 12.8; CPU fallback below 12.6 |
| GDAL / Rasterio | 3.9.2 / 1.3.11, v2024.9.22 | 3.12.2 / 1.5.0, v2026.2.26 |

The torch wheels were verified in official [CPU](https://download.pytorch.org/whl/cpu/), [CUDA 12.6](https://download.pytorch.org/whl/cu126/), and [CUDA 12.8](https://download.pytorch.org/whl/cu128/) indexes. Windows GDAL/Rasterio wheels come from the [versioned geospatial-wheels release](https://github.com/cgohlke/geospatial-wheels/releases/expanded_assets/v2026.2.26). Other dependencies remain selected by pip.

The production changes are small: admit the 3.14 profile and select matching wheels; use `shutil.copytree(..., dirs_exist_ok=True)` instead of the removed `distutils` API; remove an unused `snuggs` import because Rasterio 1.5 no longer supplies it transitively; escape an existing correspondence-column backslash without changing the string. No segmentation algorithm, pretrained weight, augmentation preset, slope method, or scientific default was deliberately changed in application code.

## Verified execution, 2026-10-10

Real `install.py no-sam` completed in both isolated, activated Conda environments, with the user site disabled. Both passed `pip check`. Python 3.11.17 used torch 2.5.0+cu124 / torchvision 0.20.0+cu124 / GDAL 3.9.2 / Rasterio 1.3.11; ordinary Python 3.14.8 used torch 2.9.1+cu128 / torchvision 0.24.1+cu128 / GDAL 3.12.2 / Rasterio 1.5.0. NumPy was 2.4.6 versus 2.5.3; SciPy 1.17.1 versus 1.18.1. PyQt5 5.15.11, Albumentations 2.0.8 and scikit-image 0.26.0 were unchanged.

The 38 standard-library tests passed in each profile. Functional checks covered real Qt entry-point initialization, event-loop rendering, annotated project/correspondence save/load, GDAL/Rasterio roundtrip and TagLab georeference copying, training data, six legacy checkpoints, RITM and DEXTR CPU/CUDA inference, all three configured classifiers, DeepLab backward/QHAdam and SAM CPU/CUDA with the complete verified checkpoint. CUDA used an RTX 4070 Laptop GPU and driver advertising CUDA UMD 13.4. CPU checks used CUDA-enabled torch wheels on the CPU; CPU-only and CUDA 12.6 installs were not separate tested environments.

The initial GUI smoke imported the class without entry-point style initialization; its harness was corrected to execute the real startup body. A real 3.14 GUI import then failed on the unused snuggs dependency and was fixed by removing that import. These failures are retained in preliminary receipts. The original SAM test copy was truncated. The orchestrator repaired it from the official Meta mirror: 2,564,550,879 bytes, valid ZIP CRC, SHA256 `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e`. Both profiles now load and infer successfully.

## Scientific results

Working execution does not establish unchanged scientific results.

**Neural fixtures.** Three fixed synthetic images, identical checkpoint hashes and inputs, produced bitwise-identical CPU outputs for RITM, DEXTR and three map classifiers. With the real TagLab startup flags, CUDA probability differences crossed thresholds: DEXTR changed 2 pixels on each of two fixtures. The three map classifiers retained identical thresholded class maps after the controlled nine-tile aggregation. On the DEXTR ellipse, TagLab's actual region area stayed 11,500 pixels while perimeter changed 431.264834 to 431.850620 pixels. The texture fixture changed 11,833 to 11,835 pixels of region area. The selected inference outputs and loss repeated bitwise in each profile across two runs. The 3.11 updated head also matched; the 3.14 updated head differed by at most 5.82e-11 in one float, despite cuDNN deterministic mode. This is much smaller than the measured cross-profile weight difference, but backward bitwise determinism must not be claimed. The backward fixture used evaluation mode to disable dropout/BN updates; it is not a full training-equivalence test. Loss differed by 6.26e-6 and the updated classifier-head weights by at most 3.26e-7.

**Existing annotated data.** Four deterministic, median-area eligible regions from the official VOS sample (one per annotated class, 100..450-pixel bounding box, 60-pixel context margin) were compared using identical data/contour hashes and reference-derived clicks: RITM uses the maximum internal distance-transform point plus a fixed negative corner; DEXTR uses four contour extremes. These are oracle prompts, not a manual interaction study. CPU outputs again matched exactly. CUDA changed 0..14 mask pixels per result. RITM IoU against one annotation increased by 0.000175; DEXTR IoU increased by 0.0000602 on one region and decreased by 0.0000959 and 0.0000431 on two others. The maximum surface difference was 0.106032 cm2 at the sample's 0.94 mm/pixel calibration. These tiny mixed changes provide no general evidence of better or worse segmentation. The existing contours are references, not independently adjudicated ground truth, and train/test separation is unknown. One convenience sample is not a quality benchmark. Attribution: Stuart Sandin Lab / Scripps Institution of Oceanography, CC BY-NC-SA 4.0.

**DEM surface measurements are materially affected in some unit configurations.** Starting with GDAL 3.11, [slope defaults infer horizontal/vertical unit ratios](https://gdal.org/en/stable/programs/gdaldem.html#slope). TagLab calls `gdal.DEMProcessing(..., 'slope')` without a scale, then derives surface area through `1/cos(slope)` and zeroes slopes above 87 degrees. Analytic planes with a true 26.565-degree slope showed:

| Coordinates / heights | Legacy surface / planar area | New surface / planar area | Change |
| --- | ---: | ---: | ---: |
| Metres / metres | 1.118034 | 1.118034 | 0% |
| WGS84 equator / metres | 1.000000 | 1.118034 | +11.80% |
| WGS84 latitude 24 / metres | 1.000000 | 1.118157 | +11.82% |
| US survey feet / metres | 1.011546 | 1.118034 | +10.53% |
| Metres / feet | 1.921192 | 1.118034 | -41.81% |

The planar analytic surface factor is sqrt(1.25), about 1.118034. The new mixed-unit results improve agreement with that physical reference. Geographic auto-scaling remains approximate (including the latitude-24 result). Explicit `scale=1` reproduces the old slopes, isolating this default change; it is a diagnostic control, not a proposed silent application fix. Correct metadata is required. Historical surface measurements cannot be treated as directly comparable without recording the GDAL version/unit policy.

The central 64x64 crop of the official Plot18 DEM (projected metre units, no declared band unit) gave identical slope and surface factor 1.321152. A NoData-hole control revealed an existing common issue: TagLab treats the -9999 slope sentinel through cosine, yielding factor 1.633112 despite a valid-plane factor 1.118034. This behavior was identical in both profiles; it was not fixed in this port.

## Changelog interpretation and limits

The demonstrated GDAL unit change is a scientific correctness improvement on correctly specified mixed-unit fixtures, with a real comparability cost. CUDA/cuDNN changed from 12.4/9.1.0 to 12.8/9.10.2. TagLab.py imports models.training at startup, which sets cuDNN deterministic mode for all tools. The final comparisons import the same module: deterministic=True, benchmark=False, cuDNN TF32=True and matmul TF32=False in both profiles. Earlier standalone controls with deterministic=False are retained separately; their pixel counts are not the final application-state results. Equal flags do not imply identical numerical outputs across releases; [PyTorch documents that limitation](https://docs.pytorch.org/docs/2.9/notes/randomness.html).

Other release improvements require their triggering path: Rasterio inverse-affine/boundless-read fixes were not demonstrated on an active TagLab call; torch's very-large-tensor interpolation fix is an edge case. TorchInductor Welford, MPS-specific batch-normalization and cuDNN attention fixes do not establish gains for this Windows eager/SAM-1 path. Albumentations, OpenCV, pandas, scikit-image, scikit-learn and SAM versions did not change; their old release improvements are not gains from this port.

The existing Albumentations preset has obsolete parameters ignored by 2.0.8 in both profiles. An initial eight-8x8-hole correction was removed from this branch: restoring historical intent changes current training behavior and must be evaluated independently. No full retraining, multi-seed convergence comparison, production orthomap tiling, SAM quality comparison, manual interactive session, CoralNet service workflow, broad reprojection/nodata audit or non-Windows validation was completed. No blanket `torch.load(weights_only=False)` change was required for the official checkpoints.

## Hardware and exploratory performance

The requested release policy is one application release with an established/legacy dependency profile and an optional modern profile, rather than forcing Python 3.14/latest onto all users. This prototype retains the 3.11 torch/geospatial profile and adds the opt-in 3.14 interpreter profile; it does not implement a complete hardware-qualified, locked release matrix. Exact environment manifests are necessary because common requirements remain unpinned.

The test PC is a Core Ultra 7 155H, 63.46 GiB system RAM and RTX 4070 Laptop with 8 GiB VRAM. It is not a representative five-year-old/IMR machine. A GPU purchase date does not define compute capability or VRAM. The installed cu128 binary lists sm_70/75/80/86/90/100/120, while the legacy cu124 binary also lists sm_50/60/61. [PyTorch's packaging policy](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128) excludes Maxwell/Pascal from cu128 builds. The current installer selects by driver CUDA capability only and can select an unsuitable cu128 wheel on an old GPU with a recent driver. The cu126 alternative and older hardware were not qualified. A driver below 12.6 also makes this modern profile fall back to CPU. Do not promise broad older-GPU compatibility.

Eight same-PC actions used the same official VOS region 62, checkpoint hashes, startup flags and 16 torch CPU threads, one warmup and three timed repeats, with CUDA synchronization. Initial median latencies:

| Action | 3.11 | 3.14 |
| --- | ---: | ---: |
| RITM CPU, image + two clicks | 0.302 s | 0.255 s |
| DEXTR CPU | 2.220 s | 3.525 s |
| SAM CPU, new image + click | 39.460 s | 37.465 s |
| SAM CPU, cached-image click | 86 ms | 117 ms |
| RITM CUDA, image + two clicks | 86 ms | 118 ms |
| DEXTR CUDA | 64 ms | 55 ms |
| SAM CUDA, new image + click | 1.223 s | 1.231 s |
| SAM CUDA, cached-image click | 7.13 ms | 30.13 ms |

These are exploratory timings from a sequential run; power, thermal state and process-level retention were not controlled. Targeted fresh-process controls were then performed by the orchestrator. DEXTR CPU used two warmups/seven repeats: final clean 3.11 median 2.967 s (2.390..4.670), 3.14 median 4.108 s (3.159..7.848). Fresh SAM CUDA used three warmups/50 cached clicks: 3.11 median 9.29 ms (6.74..22.03), 3.14 median 15.96 ms (8.84..26.74). The latter controls may have overlapped a child CPU control and must be qualified accordingly; their ratio is not a guaranteed regression factor. The substantial ranges and changed medians show why the initial ratios should not be generalized or attributed to Python alone. No further campaign or kernel optimization was performed.

The sequential process peak working set was 6.44 versus 9.70 GiB. Fresh SAM CUDA construction peaks were much closer (about 5.53 versus 5.64 GiB); ending working sets and private committed memory differed, suggesting retained/cache effects that were not isolated. Sequential RAM peaks are not isolated per-model requirements or a minimum-RAM specification. SAM new-image inference peaked at about 5.61 GiB GPU allocation and 6.13 GiB reserved in both profiles, which does not validate a 6 GiB GPU. Non-regression of speed on older machines is **not validated**, and these mixed same-PC results do not justify a general recommendation to migrate.

## Reproduction

Use a separate environment, activate Conda and disable user-site packages before tests. The original installation is not a target for this experiment.

```powershell
conda create -n taglab314 -c conda-forge python=3.14 pip
conda activate taglab314
$env:PYTHONNOUSERSITE = '1'
$env:PIP_USER = 'false'
python install.py
python -m pip check
python -m unittest discover -s tests -v
python tests/smoke_runtime.py --report runtime.json
```

Installer options `cpu` and `no-sam` may be combined. The functional harness supports `--skip-sam`, `--skip-cuda`, `--only` and an explicit `--sam-checkpoint`. Its remote version check is stubbed to the local version, while dependency imports, event loop, model loads and inference are real. Temporary settings/projects and ignored model/backbone caches are used.

Generate paired neural fixtures with `tests/compare_scientific.py --report <json> --arrays <npz>`, postprocess each with `--postprocess <raw-npz> --arrays <final-npz> --report <json>`, then compare with `--compare <baseline-final-npz> <candidate-final-npz> --metadata <baseline-json> <candidate-json> --report <json>`. `--figure` saves the pixel-change plot; `--select` supports targeted repeatability controls and `--allow-subset` their comparison.

Run `tests/compare_samples.py --samples <official-sampleProjects-directory> --report <json> --arrays <npz>` in each profile, then `--compare <baseline-json> <candidate-json> --report <json>`. The local validation cache contains only read-only copies of the specified official examples, excluded from Git. `tests/compare_dem.py --report <json>` uses analytical fixtures and `models/.samples/Plot18_DEM.tiff`. `tests/benchmark_inference.py --report <json>` records sequential latency/memory measurements; `--model`, `--device`, `--warmup`, `--repeats` and `--cached-only` support bounded fresh-process controls. `--compare` consolidates two stored receipts without rerunning inference.

`update.py` still downloads the original project's version-based release and can replace fork changes. Update this experimental branch through Git from the fork. No upstream Python 3.14 issue, comment, review or PR was submitted by this work.
