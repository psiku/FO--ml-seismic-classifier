from pathlib import Path


def get_root_dir() -> Path:
    ROOT_DIR = Path(__file__).parent.parent
    return ROOT_DIR
