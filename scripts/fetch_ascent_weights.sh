#!/usr/bin/env bash
# Fetch the model weights native ASCENT needs into pretrained_weights/.
#
#   bash scripts/fetch_ascent_weights.sh [DEST]
#
# Four of the seven are already in this repo under data/weights and are linked
# rather than re-downloaded. Qwen2.5-7B-Instruct (~15 GB) is the bulk of it.
set -euo pipefail
DEST="${1:-relative_work/ascent/pretrained_weights}"
mkdir -p "$DEST"
here() { [ -s "$DEST/$1" ]; }

link() {  # $1 name in DEST, $2 path we already have
    if [ -f "$2" ] && ! here "$1"; then ln -sf "$(realpath "$2")" "$DEST/$1"; echo "  linked $1"; fi
}
get() {   # $1 name, $2 url
    here "$1" && { echo "  have   $1"; return; }
    echo "  fetch  $1"; curl -fL --retry 3 -o "$DEST/$1" "$2"
}

echo "already in this repo:"
link mobile_sam.pt                data/weights/mobile_sam.pt
link rednet_semmap_mp3d_40.pth    data/weights/rednet_semmap_mp3d_40.pth
link resnet50_places365.pth.tar   data/place365/resnet50_places365.pth.tar
link pointnav_weights.pth         data/weights/pointnav_weights.pth

echo "downloading:"
get groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
get dfine_x_obj2coco.pth \
    https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth
get ram_plus_swin_large_14m.pth \
    https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# Qwen2.5-7B-Instruct: a repo, not a file. Needs `pip install huggingface_hub`.
if [ ! -d "$DEST/Qwen2.5-7b" ]; then
    echo "  fetch  Qwen2.5-7B-Instruct (~15 GB)"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$DEST/Qwen2.5-7b')"
else
    echo "  have   Qwen2.5-7b"
fi

echo
echo "done. Expected layout:"
ls -la "$DEST"
