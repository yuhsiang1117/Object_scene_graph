# configs/experiment/archive

Exploratory presets: the rungs of ablation ladders and the variants that lost an A/B. None of
them is run by a current script or named by the reference documentation, but each was
measured once and the document that reports the measurement (under `docs/history/`) names it,
so they are kept composable rather than deleted.

They compose exactly as before with the directory in the name:

```bash
python scripts/run_eval.py +experiment=archive/dualmap_protocol_osg_sensor_v3
```

Their bases are named by absolute group path (`/experiment/<name>`) when the base stayed at the
top level and by `archive/<name>` when it moved here, so a preset here still composes to what
it was measured with. `tests/unit/golden/experiment_fingerprints.json` pins every one of them
under its `archive/<name>` key, the same lock the top-level presets have.

| preset | what it tried |
|---|---|
| `ascent_sensor_all3`, `_arrival`, `_band`, `_carrot`, `_fix1`, `_full`, `_rednet` | the `ascent_sensor` ladder (S8–S30 in `AB_RESULTS.md`) |
| `dualmap_protocol_ascent` | ASCENT's policy on DualMap's protocol |
| `dualmap_protocol_osg_look_flat_anchor_v2_island_close_feat`, `_featp`, `_featsz` | feature-memory variants of the `look_flat` chain |
| `dualmap_protocol_osg_sensor_v3`, `_v3_region` | the sensor arm between `_v2` and `_v4_fuse` |
| `dualmap_protocol_osg_sensor_v4_observe` | `_v4_fuse` with the observe-only proposer |
| `ycb_authored_navmesh` | the authored benchmark on privileged navigation (diagnostic; not comparable to sensor-only arms) |
| `ycb_dynamic_sat`, `ycb_dynamic_grounded` | the saturation and grounded conditions of the dynamic ladder |
