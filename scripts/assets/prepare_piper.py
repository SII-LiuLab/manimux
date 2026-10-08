"""Prepare standard PiPER visuals from an official piper_ros checkout.

Source: agilexrobotics/piper_ros, revision ac41fcbcdda598f01b51cf6175ed9a24d0dacadc.
Retain joint geometry, inertia and visual materials; remove collision geometry
and replace ROS package references with paths relative to the bundled URDF.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "manimux/embodiments/arm/piper"


def prepare(upstream: Path, output: Path) -> None:
    description = upstream / "src/piper_description"
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    model = ET.parse(description / "urdf/piper_description.urdf", parser=parser)
    for link in model.getroot().findall("link"):
        for collision in link.findall("collision"):
            link.remove(collision)

    meshes = {}
    for mesh in model.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        prefix = "package://piper_description/meshes/"
        if not filename.startswith(prefix):
            raise ValueError(f"unexpected PiPER mesh reference: {filename}")
        name = filename.removeprefix(prefix)
        if Path(name).name != name:
            raise ValueError(f"mesh must be directly inside the upstream meshes directory: {name}")
        source = description / "meshes" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        meshes[name] = source
        mesh.set("filename", f"assets/meshes/{name}")
    license_path = upstream / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError(license_path)

    destination = output / "assets/meshes"
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in meshes.items():
        shutil.copyfile(source, destination / name)
    ET.indent(model)
    model.write(output / "model.urdf", encoding="utf-8", xml_declaration=True)
    shutil.copyfile(license_path, output / "LICENSE.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", type=Path, required=True, help="Root of the pinned piper_ros source"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="PiPER component directory"
    )
    args = parser.parse_args()
    prepare(args.upstream, args.output)


if __name__ == "__main__":
    main()
