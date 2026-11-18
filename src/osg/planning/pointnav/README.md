# Vendored PointNav policy

The frozen point-goal controller ASCENT uses as its mover
(`ascent/ascent_policy.py:837 _pointnav`). It takes `(rho, theta)` to a goal
plus a 224x224 depth image and returns one of habitat's four navigation
actions. It is sensor-only: no navmesh, no map, no privileged geometry.

## Verbatim from VLFM

`nh_pointnav_policy.py`, `resnet.py` and `rnn_state_encoder.py` are copied
unmodified (BDAI copyright headers intact) from

    vlfm/policy/utils/non_habitat_policy/

at vlfm commit `584ed56008754fde7997d904983607def8328322`, vendored in this repo
under `relative_work/ascent/third_party/vlfm`. They import only `torch`.

This is the required path, not a convenience: `habitat_baselines` is not
installed in the nav container, so `PointNavResNetPolicy` from habitat-baselines
cannot be constructed here.

## First-party: `discrete_policy.py`

VLFM's `PointNavResNetPolicy` in `nh_pointnav_policy.py` is the **Spot /
continuous-control** variant -- a `GaussianNet(512, 2)` head emitting linear and
angular velocity. `pointnav_weights.pth`, which is what ASCENT actually runs, is
the **discrete habitat** checkpoint:

    net.prev_action_embedding.weight   (5, 32)   -> nn.Embedding(4 + 1, 32)
    action_distribution.linear.weight  (4, 512)  -> Categorical over 4 actions
    critic.fc.weight                   (1, 512)

Loading that checkpoint through VLFM's non-habitat path silently keeps only the
ResNet encoder and leaves the action head randomly initialised (its loader
filters unmatched keys and prints them as "unused"). `discrete_policy.py` is the
matching discrete head, so the whole checkpoint loads strictly. Only the shared
encoder body is reused -- `PointNavResNetNet(discrete_actions=True)` from the
vendored file, whose 78 parameter tensors match the checkpoint exactly (verified
key-by-key and shape-by-shape).

The one key rename, `net.prev_action_embedding` -> `net.prev_action_embedding_discrete`,
is applied once by `scripts/download_weights.py --pointnav`, which also strips
the pickled habitat-baselines config object so the runtime load needs neither a
shim nor `weights_only=False`.
