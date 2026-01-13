from pathlib import Path
import urllib.request
from src.constants import ROOT_DIR


URLS = {
    "metadata.csv": "https://seisbench.gfz-potsdam.de/mirror/datasets/iquique/metadata.csv",
    "waveforms.hdf5": "https://seisbench.gfz-potsdam.de/mirror/datasets/iquique/waveforms.hdf5",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    out_dir = ROOT_DIR / "iquique"
    out_dir.mkdir(exist_ok=True)

    for filename, url in URLS.items():
        download(url, out_dir / filename)

    print("Done.")


if __name__ == "__main__":
    main()
