# Integration verification record

## Source identity

- Dynamic source: `cc2270424cdcf205ffeccb8e6e8aec63e628ff63`
- Multi-floor behavior snapshot: `8fb27d2f861a3602608ca26d3ec56cca6a3cc958`
- ASCENT reference: `8f7bbf906c8433a47c4d2c46a268cd1258c4057c`
- VLFM submodule: `584ed56008754fde7997d904983607def8328322`
- frontier_exploration submodule: `8523aa857e4b81d15e14a506fe94a77d5f2e7d5d`

## Environment and model endpoints

- GPU: NVIDIA GeForce RTX 3080, 10240 MiB; driver 580.82.07.
- Habitat/Habitat-Sim 0.3.1; PyTorch 2.4.1+cu121.
- Text endpoint: `https://integrate.api.nvidia.com/v1`, default model
  `nvidia/nvidia-nemotron-nano-9b-v2`.
- Verification model: `meta/llama-3.2-11b-vision-instruct`.
- Secrets are read from the environment and are not recorded here.
- The flattened default config is pinned in
  `tests/unit/golden/config_snapshot.json`; every experiment's composed SHA-256
  is pinned in `tests/unit/golden/experiment_fingerprints.json`.

## Ignored staged assets

All paths in this table remain ignored and are not integration commits.

| Asset | SHA-256 / identity | Validation |
|---|---|---|
| HM3D ObjectNav v1 episodes zip | `24d37da5a3919ca15c81251f9420eec5fef32a48ab26a6eeabda723e8387bee1` | extracted `train`, `val`, `val_mini` under `data/datasets/objectnav/hm3d/v1` |
| PointNav weights | `55898285b3d101e61f7accb27d5989e42579b3a8b792de48ea627163b1012295` | converted to weights-only, strict policy load passed (80 tensors) |
| RedNet MPCAT40 | `f94d1c62a73bc05690ae29200d3dbd033ff243e7ce91755d1cd928bde844f995` | checkpoint loaded; final output head has 40 classes |
| OpenAI CLIP ViT-B/32 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` | `clip.load(..., device="cpu")` completed |
| MobileCLIP | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` | existing TorchScript asset |
| YOLOE-11s | `8e439445c87338b79d9ce21dec109f4621e26df67e94d26ea1a98c1e64dce3e3` | existing detector asset |
| YOLOE-11l | `a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f` | existing detector asset |

Generated ASCENT priors are under `data/priors/`; the Places365 category file
is under `data/place365/`. The mounted HM3D v0.2 scene data was reused and was
not downloaded again.

## Baselines and deterministic gates

- Dynamic source unit baseline before integration: `486 passed, 1 xfailed`.
- Isolated upstream snapshot unit baseline after staging its public priors:
  `448 passed, 3 skipped`.
- Integrated unit suite during implementation: `842 passed, 2 skipped,
  1 xfailed` (the final result is updated before the ancestry merge).
- All 27 experiment presets compose; their complete resolved fingerprints are
  pinned.
- Constructor/import coverage exercises `costmap`, `navmesh`, and `pointnav`
  navigation plus `nav_agent`, `ascent`, and `ascentnav` policy selection
  without initializing a GPU model.
- Cross-floor authored manifest generation produced ten valid deterministic
  cases at `start_min_geodesic_m=1.0`: every prior/destination floor differs,
  every start lies within 0.5 m of the prior floor, and the set includes both
  upward and downward moves. This includes the requested downward 00844 case.

## Long-running paired gates

The 96-episode dynamic A/B, the fixed 100-episode ASCENT A/B, cross-floor
subset, and full combined navigation campaign are intentionally not replaced by
unit tests. Their acceptance thresholds and commands remain in the integration
plan and `docs/AB_RESULTS.md`. Run artifacts belong under ignored `outputs/`;
the first combined full run establishes its quantitative baseline.
