"""The Dockerfile and its build context have to agree.

Two failures here cost a real build each. The first merged Dockerfile
`COPY`d `relative_work/ascent` while compose set `context: docker/`, so the
path was not in the context at all. The second hazard is subtler and has the
same symptom: a `COPY` of a path that *is* in the tree but is excluded by
`/.dockerignore`. Both fail late, after the expensive conda and torch layers.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "Dockerfile"
COMPOSE = ROOT / "docker" / "compose.yaml"
DOCKERIGNORE = ROOT / ".dockerignore"
# Every image built from this repo, each with the compose file that builds it.
# The Jetson pair is a separate deployment, not an override of the x86 one:
# different base, different architecture, no Habitat (docs/THOR.md).
DOCKERFILES = {
    "docker/Dockerfile": "docker/compose.yaml",
    "docker/Dockerfile.thor": "docker/compose.thor.yaml",
    "docker/Dockerfile.bridge": "docker/compose.thor.yaml",
}

# Everything the image needs at build time. Excluding any of these silently
# breaks the build a long way in.
BUILD_INPUTS = (
    "relative_work/ascent/model_api/dfine_out.py",
    "relative_work/ascent/third_party/GroundingDINO/setup.py",
    "relative_work/ascent/third_party/MobileSAM/setup.py",
    "relative_work/ascent/third_party/habitat-lab/habitat-lab/setup.py",
    "docker/entrypoint.sh",
    "docker/modules/install_claude_code.sh",
    "docker/modules/install_ros2_humble.sh",
    "docker/modules/install_x11_opengl_vulkan.sh",
)
# Directories that must never reach the daemon: together they are ~34 GB.
MUST_EXCLUDE = ("data/x.pth", "outputs/run/episodes.jsonl",
                ".conda-envs/ascent/bin/python",
                "relative_work/ascent/pretrained_weights/mobile_sam.pt",
                "docker/.env")


def _patterns() -> list:
    return [l.strip() for l in DOCKERIGNORE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def ignored(rel: str) -> bool:
    """Approximate docker's matcher: enough to catch the mistakes above."""
    for pat in _patterns():
        q = pat.rstrip("/")
        if q.startswith("**/"):
            q = q[3:]
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, q):
            return True
        if rel.startswith(q + "/"):
            return True
        if any(fnmatch.fnmatch(seg, q) for seg in rel.split("/")):
            return True
    return False


def copy_sources(dockerfile=None) -> list:
    # Join line continuations: `COPY --chown=x \<newline>  src dst` is one
    # instruction, and reading it line-wise captures the backslash.
    text = (dockerfile or DOCKERFILE).read_text().replace("\\\n", " ")
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(?:--chown=\S+\s+)?(\S+)", line)
        if m:
            out.append(m.group(1).lstrip("./"))
    return out


def all_copy_sources() -> list:
    """(dockerfile, source) for every COPY in every image."""
    return [(name, src) for name in DOCKERFILES
            for src in copy_sources(ROOT / name)]


def test_compose_builds_from_the_repo_root():
    """The COPY paths below are written relative to the root; a narrower
    context makes every one of them wrong."""
    text = COMPOSE.read_text()
    assert re.search(r"context:\s*\.\.\s*$", text, re.M), \
        "docker/compose.yaml must set `context: ..` for the COPYs to resolve"
    assert re.search(r"dockerfile:\s*docker/Dockerfile\s*$", text, re.M), \
        "with a root context the dockerfile path must be docker/Dockerfile"


def test_every_copy_source_exists():
    missing = [f"{f}: {s}" for f, s in all_copy_sources() if not (ROOT / s).exists()]
    assert not missing, f"COPY sources not in the tree: {missing}"


def test_no_copy_source_is_excluded_by_dockerignore():
    """A path that exists but is ignored fails the build the same way a missing
    one does, and reads as a typo rather than an ignore rule."""
    blocked = [f"{f}: {s}" for f, s in all_copy_sources() if ignored(s)]
    assert not blocked, f"COPY sources excluded by /.dockerignore: {blocked}"


def test_every_dockerfile_is_built_from_the_repo_root():
    """Same trap as the x86 image, once per deployment: a COPY path written
    relative to the root and a context narrower than the root."""
    for dockerfile, compose in DOCKERFILES.items():
        text = (ROOT / compose).read_text()
        assert re.search(r"context:\s*\.\.\s*$", text, re.M), \
            f"{compose} must set `context: ..`"
        assert f"dockerfile: {dockerfile}" in text, \
            f"{compose} does not build {dockerfile}"


def test_the_jetson_images_carry_no_habitat():
    """There is no aarch64 build of habitat-sim, and the robot path does not
    need one -- `eval.mode=ros2` never constructs a Habitat env. An image that
    tried would fail late, after the expensive layers."""
    for name in ("docker/Dockerfile.thor", "docker/Dockerfile.bridge"):
        text = (ROOT / name).read_text()
        installs = [l for l in text.splitlines()
                    if re.match(r"\s*(RUN|ENV)\b", l) and "habitat-sim" in l]
        assert not installs, f"{name} installs habitat-sim: {installs}"


def test_the_build_inputs_survive_the_ignore_file():
    blocked = [p for p in BUILD_INPUTS if ignored(p)]
    assert not blocked, f"/.dockerignore excludes build inputs: {blocked}"


def test_the_heavy_directories_are_excluded():
    """Without these the context is 35 GB instead of ~213 MB."""
    leaked = [p for p in MUST_EXCLUDE if not ignored(p)]
    assert not leaked, f"/.dockerignore lets these into the context: {leaked}"


REQUIREMENTS = ROOT / "docker" / "ascent-requirements.txt"


def _requirement_lines() -> list:
    return [l.strip() for l in REQUIREMENTS.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def test_the_ascent_requirements_parse_and_are_uniquely_named():
    """pip refuses a file that names one package twice -- and the environment
    this was generated from really does carry two setuptools metadata dirs (a
    live .egg-info beside a stale .egg), which is exactly how that shipped."""
    from packaging.requirements import Requirement

    seen = {}
    for line in _requirement_lines():
        req = Requirement(line)          # raises on anything pip cannot read
        key = re.sub(r"[-_.]+", "-", req.name).lower()
        seen.setdefault(key, []).append(line)
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dups, f"pip will reject these duplicate pins: {dups}"


def test_the_bootstrap_packages_are_not_pinned():
    """pip, setuptools and wheel come with the interpreter; pinning them under
    --no-deps fights the copies already installed."""
    names = {re.sub(r"[-_.]+", "-", l.split("==")[0]).lower() for l in _requirement_lines()}
    clash = names & {"pip", "setuptools", "wheel", "habitat-sim"}
    assert not clash, f"these must not be in the requirements file: {sorted(clash)}"


def test_every_requirement_is_pinned():
    """The list is installed with --no-deps, so an unpinned entry would resolve
    to whatever is newest and silently diverge from the measured environment."""
    loose = [l for l in _requirement_lines() if "==" not in l]
    assert not loose, f"unpinned requirements: {loose}"


def test_the_submodule_is_a_recorded_pin():
    """The build COPYs a submodule, so the commit it ships is whatever git
    records -- there is no second copy of the pin to drift."""
    gitmodules = (ROOT / ".gitmodules").read_text()
    assert "relative_work/ascent" in gitmodules
    assert "Zeying-Gong/ascent" in gitmodules
