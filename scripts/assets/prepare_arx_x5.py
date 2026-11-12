"""Prepare the official X5 kinematic URDF without vendoring its SDK or meshes."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def prepare(source: Path, license_path: Path, destination: Path) -> None:
    model = ET.parse(source)
    for link in model.getroot().findall("link"):
        for element in list(link):
            if element.tag in {"visual", "collision"}:
                link.remove(element)
    destination.mkdir(parents=True, exist_ok=True)
    ET.indent(model)
    model.write(destination / "model.urdf", encoding="utf-8", xml_declaration=True)
    shutil.copyfile(license_path, destination / "LICENSE.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "manimux/embodiments/arm/arx_x5",
        help="destination component directory (default: bundled arx_x5 component)",
    )
    args = parser.parse_args()
    prepare(args.urdf, args.license, args.output)


if __name__ == "__main__":
    main()
