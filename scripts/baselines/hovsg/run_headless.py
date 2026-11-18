"""Run one of HOV-SG's application scripts without a display, and with the faster mask merge.

`hovsg/graph/navigation_graph.py` calls `plt.switch_backend("TkAgg")` at import, which raises
on a headless machine. The backend only draws debug figures, so it is swapped for Agg here.
`merge_3d_masks` is replaced by fast_merge.py's, which returns the same masks (checked bit for
bit there) in a fraction of the time. The floor-pose assignment gets 10 cm of slack on its sensor
height (block below). HOV-SG's source is left as it is.

    python run_headless.py <HOV-SG>/application/create_graph.py [hydra overrides...]
"""
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_switch_backend = plt.switch_backend
plt.switch_backend = lambda name: _switch_backend("Agg" if str(name).lower() == "tkagg" else name)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_merge  # noqa: E402

fast_merge.install()

# HOV-SG assigns camera poses to a floor by `pose height - camera_height >= lowest floor point`,
# with camera_height 1.5 m, its own walks' sensor height. The release's camera is also 1.5 m above
# the agent, but in 00829 the reconstructed floor sits 6 cm above the navmesh the agent stands on
# (camera 1.436 m, lowest floor point 0.001 m), so every pose fell below the floor and the
# navigation graph had none ("Found array with 0 sample(s)"). 10 cm of slack absorbs that offset;
# it only decides which floor a pose belongs to.
from hovsg.graph.navigation_graph import NavigationGraph  # noqa: E402

_get_floor_poses = NavigationGraph.get_floor_poses
NavigationGraph.get_floor_poses = (
    lambda self, floor, poses_list, upper_floor_min_height=None, camera_height=1.5:
    _get_floor_poses(self, floor, poses_list, upper_floor_min_height, camera_height - 0.1))

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
