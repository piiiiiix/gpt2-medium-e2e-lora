"""Prepare pinned data/scorer; verify Java and Perl before expensive experiments."""

import argparse
import shutil
import subprocess
import zipfile

from config import OUTPUT_ROOT
from utils.paper_resources import data_files, fetch, preflight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--install-java",
        action="store_true",
        help="Download pinned portable Windows x64 JRE into outputs/resources",
    )
    options = parser.parse_args()
    if options.install_java:
        root = OUTPUT_ROOT / "resources/runtime"
        archive = fetch(
            {
                "url": "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jre_x64_windows_hotspot_17.0.20.1_1.zip",
                "sha256": "bc21a93923103cdaac93ee337b0ae4365e739fde36df823dd456bc67c8a9d352",
            },
            root / "temurin17.zip",
        )
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                if not (root / item.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe Java archive path")
            zipped.extractall(root)
    print(data_files())
    scorer, environment = preflight()
    print(scorer)
    subprocess.run(
        [shutil.which("java", path=environment["PATH"]), "-version"], env=environment, check=True
    )
    subprocess.run(
        [shutil.which("perl", path=environment["PATH"]), "-v"], env=environment, check=True
    )
