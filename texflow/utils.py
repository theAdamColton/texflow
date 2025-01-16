from pathlib import Path
import toml

ROOT_PATH = Path(__file__).parent.parent.absolute()


_pyproject_toml_file = ROOT_PATH / "pyproject.toml"
with open(_pyproject_toml_file, "r") as f:
    _data = toml.load(f)

VERSION_STRING = _data["project"]["version"]
VERSION_TUPLE = tuple(int(d) for d in VERSION_STRING.split("."))
DESCRIPTION = _data["project"]["description"]
PYTHON_VERSION = _data["project"]["requires-python"].removeprefix("==")


def make_blender_manifest_dict():
    return dict(
        schema_version="1.0.0",
        id="adam_colton_texflow",
        version=VERSION_STRING,
        name="texflow",
        tagline=DESCRIPTION,
        maintainer="Adam Colton <atcolton@tutanota.com>",
        type="add-on",
        tags=["Paint"],
        blender_version_min="4.2.0",
        license=[
            "SPDX:GPL-3.0-or-later",
        ],
        platforms=["windows-x64", "macos-arm64", "linux-x64"],
    )
