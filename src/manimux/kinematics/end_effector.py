"""Swappable end effectors mounted on an arm flange.

A gripper, hand or tool is described once under
the owning component's ``assets/<name>/`` as a standalone URDF plus
``end_effector.yaml``, independent of the arm that carries it. The same
description yields both the tool transform (flange -> TCP) that pose-space
kinematics use and the combined arm + end-effector URDF the viewer renders, so
a drawn fingertip and the IK tool frame cannot drift apart.

``end_effector.yaml``::

    name: umi_follower
    urdf: end_effector.urdf          # relative to this directory
    root_link: base_link             # the link bolted to the flange
    mount: {xyz: [...], rpy: [...]}  # flange -> root_link (metres, URDF rpy)
    tcp: {xyz: [...], rpy: [...]}    # root_link -> tool centre point
    inputs: 1                        # values appended to each arm's joint vector
    rest_inputs: [1.0]               # drawn when a state carries no inputs
    aperture_input: 0                # optional: input that is a 0-closed/1-open aperture
    joints:                          # how inputs drive the URDF's actuated joints
      - {joint: joint1, input: 0, at_0: -0.44, at_1: 0.0}

An actuated joint takes ``at_0 + v * (at_1 - at_0)`` for its input ``v``, which
``clip`` (default true) first limits to [0, 1]. Every non-fixed, non-mimic URDF
joint must be driven exactly once; mimic joints follow their leader.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

from manimux.kinematics.base import FloatArray

DEFAULT_END_EFFECTOR_ROOT = (
    Path(__file__).resolve().parents[1] / "embodiments" / "end_effector" / "taccap" / "assets"
)
SPEC_FILENAME = "end_effector.yaml"

# Names of end-effector links and joints inside a combined URDF.
LINK_PREFIX = "ee_"
MOUNT_JOINT = f"{LINK_PREFIX}mount"
TCP_LINK = f"{LINK_PREFIX}tcp"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Frame(_Strict):
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> FloatArray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_euler("xyz", self.rpy).as_matrix()
        transform[:3, 3] = self.xyz
        return transform


class JointDrive(_Strict):
    joint: str = Field(min_length=1)
    input: int = Field(ge=0)
    at_0: float
    at_1: float
    clip: bool = True

    def position(self, value: float) -> float:
        fraction = float(np.clip(value, 0.0, 1.0)) if self.clip else float(value)
        return self.at_0 + fraction * (self.at_1 - self.at_0)


class EndEffectorSpec(_Strict):
    name: str = Field(min_length=1)
    urdf: str = "end_effector.urdf"
    root_link: str = Field(min_length=1)
    mount: Frame = Frame()
    tcp: Frame = Frame()
    inputs: int = Field(default=0, ge=0)
    rest_inputs: tuple[float, ...] = ()
    aperture_input: int | None = Field(default=None, ge=0)
    joints: tuple[JointDrive, ...] = ()

    @model_validator(mode="after")
    def _inputs_are_consistent(self) -> EndEffectorSpec:
        if len(self.rest_inputs) != self.inputs:
            raise ValueError(
                f"rest_inputs must have {self.inputs} values, got {len(self.rest_inputs)}"
            )
        if self.aperture_input is not None and self.aperture_input >= self.inputs:
            raise ValueError(f"aperture_input {self.aperture_input} is not one of the inputs")
        for drive in self.joints:
            if drive.input >= self.inputs:
                raise ValueError(
                    f"joint {drive.joint!r} reads input {drive.input}, "
                    f"but only {self.inputs} inputs exist"
                )
        names = [drive.joint for drive in self.joints]
        if len(set(names)) != len(names):
            raise ValueError("each joint may be driven only once")
        return self


@dataclass(frozen=True, slots=True)
class EndEffector:
    spec: EndEffectorSpec
    directory: Path
    # Driven joints in URDF order, which is the order a combined URDF expects.
    actuated_joints: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def inputs(self) -> int:
        return self.spec.inputs

    @property
    def urdf_path(self) -> Path:
        return self.directory / self.spec.urdf

    def tool_transform(self) -> FloatArray:
        """Flange -> tool centre point."""

        return self.spec.mount.matrix() @ self.spec.tcp.matrix()

    def joint_positions(self, inputs: Sequence[float] | FloatArray | None = None) -> FloatArray:
        """End-effector inputs (``rest_inputs`` when omitted) -> actuated joint positions."""

        values = np.asarray(
            self.spec.rest_inputs if inputs is None else inputs, dtype=np.float64
        ).reshape(-1)
        if values.size != self.spec.inputs:
            raise ValueError(f"{self.name} expects {self.spec.inputs} inputs, got {values.size}")
        by_joint = {drive.joint: drive.position(values[drive.input]) for drive in self.spec.joints}
        return np.array([by_joint[name] for name in self.actuated_joints], dtype=np.float64)


def _movable_joints(robot: ET.Element) -> list[str]:
    return [
        joint.get("name", "")
        for joint in robot.findall("joint")
        if joint.get("type") != "fixed" and joint.find("mimic") is None
    ]


def available_end_effectors(root: Path | str | None = None) -> tuple[str, ...]:
    base = Path(root or DEFAULT_END_EFFECTOR_ROOT).expanduser()
    return tuple(sorted(path.parent.name for path in base.glob(f"*/{SPEC_FILENAME}")))


def load_end_effector(name_or_path: str | Path, root: Path | str | None = None) -> EndEffector:
    """Load a bundled end effector by name, or one from a directory path."""

    candidate = Path(name_or_path).expanduser()
    if (candidate / SPEC_FILENAME).is_file():
        directory = candidate
    else:
        directory = Path(root or DEFAULT_END_EFFECTOR_ROOT).expanduser() / str(name_or_path)
    spec_path = directory / SPEC_FILENAME
    if not spec_path.is_file():
        available = ", ".join(available_end_effectors(root)) or "none"
        raise FileNotFoundError(
            f"end effector {str(name_or_path)!r} not found at {spec_path}; available: {available}"
        )
    spec = EndEffectorSpec.model_validate(yaml.safe_load(spec_path.read_text()) or {})
    urdf_path = directory / spec.urdf
    robot = ET.parse(urdf_path).getroot()
    if spec.root_link not in {link.get("name") for link in robot.findall("link")}:
        raise ValueError(f"{spec.name}: root_link {spec.root_link!r} is not a link of {urdf_path}")
    movable = _movable_joints(robot)
    driven = {drive.joint for drive in spec.joints}
    if set(movable) != driven:
        raise ValueError(
            f"{spec.name}: joints must drive exactly the URDF's actuated joints {movable}, "
            f"got {sorted(driven)}"
        )
    return EndEffector(spec, directory.resolve(), tuple(movable))


def _absolutize_meshes(robot: ET.Element, base: Path) -> None:
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename", "")
        if "://" in filename:
            raise ValueError(f"mesh {filename!r} must be a path relative to its URDF")
        mesh.set("filename", str((base / filename).resolve()))


def _fixed_joint(name: str, parent: str, child: str, frame: Frame) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": " ".join(map(repr, frame.xyz)), "rpy": " ".join(map(repr, frame.rpy))},
    )
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def attach_end_effector(
    arm_urdf: Path | str,
    flange_link: str,
    end_effector: EndEffector | None,
    *,
    cache_dir: Path | str | None = None,
) -> Path:
    """Return a URDF of the arm with the end effector bolted to ``flange_link``.

    End-effector links and joints are renamed with ``ee_`` and a ``ee_tcp`` link
    marks the tool centre point. The combined file is written once per content
    hash; without an end effector the arm URDF itself is returned.
    """

    arm_path = Path(arm_urdf).resolve()
    if end_effector is None:
        return arm_path
    arm = ET.parse(arm_path).getroot()
    arm_names = {element.get("name") for element in [*arm.findall("link"), *arm.findall("joint")]}
    if flange_link not in {link.get("name") for link in arm.findall("link")}:
        raise ValueError(f"flange link {flange_link!r} is not a link of {arm_path}")
    _absolutize_meshes(arm, arm_path.parent)

    tool = ET.parse(end_effector.urdf_path).getroot()
    _absolutize_meshes(tool, end_effector.urdf_path.parent)
    for element in [*tool.findall("link"), *tool.findall("joint")]:
        element.set("name", LINK_PREFIX + element.get("name", ""))
        if element.tag == "joint":
            for tag in ("parent", "child"):
                reference = element.find(tag)
                if reference is not None:
                    reference.set("link", LINK_PREFIX + reference.get("link", ""))
            mimic = element.find("mimic")
            if mimic is not None:
                mimic.set("joint", LINK_PREFIX + mimic.get("joint", ""))
        if element.get("name") in arm_names:
            raise ValueError(f"end-effector name {element.get('name')!r} collides with the arm")
        arm.append(element)
    root_link = LINK_PREFIX + end_effector.spec.root_link
    arm.append(_fixed_joint(MOUNT_JOINT, flange_link, root_link, end_effector.spec.mount))
    ET.SubElement(arm, "link", {"name": TCP_LINK})
    arm.append(_fixed_joint(f"{TCP_LINK}_joint", root_link, TCP_LINK, end_effector.spec.tcp))

    text = ET.tostring(arm, encoding="unicode")
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    directory = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "manimux-urdf"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{arm_path.stem}-{end_effector.name}-{digest}.urdf"
    if not path.is_file():
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(text)
        partial.replace(path)
    return path
