import re
import toml
import os
import subprocess
import shutil

from ..utils import (
    PYTHON_VERSION,
    ROOT_PATH,
    VERSION_STRING,
    make_blender_manifest_dict,
)

if __name__ == "__main__":
    # Set up paths
    build_dir = ROOT_PATH / "build" / VERSION_STRING
    src_root = ROOT_PATH / "texflow"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    # Copy source directory to build directory excluding specific files and directories
    shutil.copytree(
        src_root,
        build_dir / src_root.name,
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "tests"),
    )

    shutil.copyfile(ROOT_PATH / "pyproject.toml", build_dir / "pyproject.toml")

    # Move client/__init__.py to the root of the build directory
    init_file = build_dir / src_root.name / "client" / "__init__.py"
    shutil.move(init_file, build_dir / "__init__.py")
    # Create an empty __init__.py in the original location
    (build_dir / src_root.name / "client" / "__init__.py").touch()

    # Change import statements in the moved __init__.py file
    init_content = (build_dir / "__init__.py").read_text()
    # fix single dot relative imports
    init_content = re.sub(
        r"^from \.(?=[a-zA-Z])",
        f"from .{src_root.name}.client.",
        init_content,
        flags=re.MULTILINE,
    )
    # fix double dot relative imports
    init_content = re.sub(
        r"^from \.\.(?=[a-zA-Z])",
        f"from .{src_root.name}.",
        init_content,
        flags=re.MULTILINE,
    )

    # no other relative imports should exist
    bad_imports = re.search(rf"^from \.(?!{src_root.name}\.)", init_content)
    if bad_imports is not None:
        raise ValueError(f"unsupported relative import! {bad_imports}")

    (build_dir / "__init__.py").write_text(init_content)

    subprocess.run(["python", "-m", "ensurepip", "--upgrade"], check=True)

    # Compile requirements for Linux
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--python-version",
            PYTHON_VERSION,
            "-o",
            str(build_dir / "requirements-linux.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "download",
            "-r",
            str(build_dir / "requirements-linux.txt"),
            "-d",
            str(build_dir / "wheels"),
            "--no-deps",
            "--only-binary=:all:",
            "--no-input",
        ],
        check=True,
    )

    # Compile requirements for Windows
    env = os.environ.copy()
    torch_index_url = "https://download.pytorch.org/whl/cu124"
    env["UV_EXTRA_INDEX_URL"] = torch_index_url
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--python-platform",
            "windows",
            "--python-version",
            PYTHON_VERSION,
            "-o",
            str(build_dir / "requirements-windows.txt"),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "download",
            "-r",
            str(build_dir / "requirements-windows.txt"),
            "-d",
            str(build_dir / "wheels"),
            "--platform",
            "win_amd64",
            "--no-deps",
            "--only-binary=:all:",
            "--extra-index-url",
            torch_index_url,
            "--no-input",
        ],
        check=True,
    )

    # Compile requirements for macOS
    subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "pyproject.toml",
            "--python-platform",
            "macos",
            "--python-version",
            PYTHON_VERSION,
            "-o",
            str(build_dir / "requirements-macos.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "pip",
            "download",
            "-r",
            str(build_dir / "requirements-macos.txt"),
            "-d",
            str(build_dir / "wheels"),
            "--platform",
            "macosx_11_0_arm64",
            "--no-deps",
            "--only-binary=:all:",
            "--no-input",
        ],
        check=True,
    )

    # This was my old hack to get universal macOS wheels to work
    # Because of blender not recognizing the universal2 name https://projects.blender.org/blender/blender/issues/125091
    # I'm using a similar fix to this https://github.com/IfcOpenShell/IfcOpenShell/commit/8f2e8ec64da31de81d72fc63ad2569bc5a3d328b
    # But it looks like MarkupSafe changed their wheels to ones that are named macos
    # So this hack is not necessary anymore
    # prev_whl_name = list((build_dir / "wheels").glob("MarkupSafe-*universal2.whl"))[0]
    # whl_name = str(prev_whl_name).replace("_universal2", "_arm64")
    # shutil.move(str(prev_whl_name), whl_name)

    # manually appends the wheel paths to the blender manifest
    manifest_file = build_dir / "blender_manifest.toml"
    manifest_dict = make_blender_manifest_dict()
    wheels = list((build_dir / "wheels").glob("*.whl"))
    manifest_dict["wheels"] = [f"./wheels/{wheel.name}" for wheel in wheels]

    with open(manifest_file, "w") as f:
        toml.dump(manifest_dict, f)

    dist_dir = ROOT_PATH / "dist"
    dist_dir.mkdir(exist_ok=True)

    blender_cmd = os.getenv("BLENDER", "blender")
    subprocess.run(
        [
            blender_cmd,
            "--command",
            "extension",
            "build",
            "--source-dir",
            str(build_dir),
            "--output-dir",
            str(dist_dir),
            "--split-platforms",
        ],
        check=True,
    )
