from src.get_root_dir import get_root_dir
from loguru import logger
from enum import Enum

DATASET_NAME = "nadeeshafdo/SeismicDataWorldwide"
DATASET_SPLITS_TRAIN = "train"
DATASET_SPLITS_TEST = "test"

ROOT_DIR = get_root_dir()

logger.info(f"Root directory set to: {ROOT_DIR}")

SRC_DIR = ROOT_DIR / "src"
DATA_FOLDER = ROOT_DIR / "data"

RAW = "raw"
PROCESSED = "processed"

RAW_DATA_DIR = DATA_FOLDER / RAW
PROCESSED_DATA_DIR = DATA_FOLDER / PROCESSED

SESMIC_DATA_CSV = RAW_DATA_DIR / "seismic_data.csv"
IQUIQUE_FOLDER_PATH = ROOT_DIR / 'iquique'
PHASENET_MODELS_DIR = ROOT_DIR / 'models' / "phasenet"

MODELS_DIR = ROOT_DIR / "models"

class SeismicEventsMapper(Enum):
    EARTHQUAKE = 0
    EXPLOSION = 1
    NATURAL_EVENT = 2
    MINING_ACTIVITY = 3
    OTHER = 4
    VOLCANIC = 5
