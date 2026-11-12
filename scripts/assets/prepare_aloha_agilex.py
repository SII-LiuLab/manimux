"""Extract portable follower-arm visuals from the pinned RoboTwin ALOHA assets."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

COLLADA = "http://www.collada.org/2005/11/COLLADASchema"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "manimux/embodiments/arm/aloha_agilex/assets/robotwin"
)
ET.register_namespace("", COLLADA)


def prepare(source: Path, destination: Path) -> None:
    """Preserve geometry; omit unused bodies, collisions and missing textures."""
    robot = ET.parse(source / "urdf/arx5_description_isaac.urdf").getroot()
    meshes = source / "urdf/aloha_maniskill_sim/meshes"
    destination.mkdir(parents=True, exist_ok=True)
    required_meshes: set[str] = set()
    for side, prefix in (("left", "fl"), ("right", "fr")):
        result = ET.Element("robot", name=f"aloha_agilex_{side}_follower")
        links = {f"{prefix}_base_link", *(f"{prefix}_link{i}" for i in range(1, 9))}
        for element in robot:
            if element.tag == "link" and element.get("name") in links:
                link = deepcopy(element)
                for collision in link.findall("collision"):
                    link.remove(collision)
                for mesh in link.findall(".//mesh"):
                    filename = Path(mesh.attrib["filename"]).name
                    if not (meshes / filename).is_file():
                        raise FileNotFoundError(meshes / filename)
                    required_meshes.add(filename)
                    mesh.set("filename", f"meshes/{filename}")
                result.append(link)
            elif element.tag == "joint":
                parent, child = element.find("parent"), element.find("child")
                if (
                    parent is not None
                    and child is not None
                    and parent.get("link") in links
                    and child.get("link") in links
                ):
                    result.append(deepcopy(element))
        if len(result.findall("link")) != 9 or len(result.findall("joint")) != 8:
            raise ValueError(f"unexpected RoboTwin follower-arm layout: {side}")
        # The RoboTwin recipe places its grasp centre 0.12 m along link6's x axis.
        # Keep link6 orientation; policy-specific EEF rotations belong to adapters.
        ET.SubElement(result, "link", name=f"{prefix}_tcp")
        joint = ET.SubElement(result, "joint", name=f"{prefix}_tcp_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=f"{prefix}_link6")
        ET.SubElement(joint, "child", link=f"{prefix}_tcp")
        ET.SubElement(joint, "origin", xyz="0.12 0 0", rpy="0 0 0")
        ET.indent(result)
        ET.ElementTree(result).write(
            destination / f"{side}_arm.urdf", encoding="utf-8", xml_declaration=True
        )
    output_meshes = destination / "meshes"
    output_meshes.mkdir(exist_ok=True)
    for filename in sorted(required_meshes):
        mesh = ET.parse(meshes / filename)
        root = mesh.getroot()
        images = root.find(f"{{{COLLADA}}}library_images")
        if images is not None:
            root.remove(images)
        # Image_269.png / Image_290.png are absent from the upstream asset folder.
        # Retain existing material colours; replace unavailable texture channels.
        for parent in root.iter():
            for texture in list(parent):
                if texture.tag == f"{{{COLLADA}}}texture":
                    parent.remove(texture)
                    ET.SubElement(parent, f"{{{COLLADA}}}color").text = "0.65 0.65 0.65 1"
        for profile in root.findall(f".//{{{COLLADA}}}profile_COMMON"):
            for parameter in profile.findall(f"{{{COLLADA}}}newparam"):
                profile.remove(parameter)
        mesh.write(output_meshes / filename, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="RoboTwin aloha-agilex folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="follower-arm asset directory (default: bundled assets/robotwin)",
    )
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
