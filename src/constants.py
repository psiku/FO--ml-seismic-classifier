from src.get_root_dir import get_root_dir

DATASET_NAME = "nadeeshafdo/SeismicDataWorldwide"
DATASET_SPLITS_TRAIN = "train"
DATASET_SPLITS_TEST = "test"

ROOT_DIR = get_root_dir()
SRC_DIR = ROOT_DIR / "src"
DATA_FOLDER = ROOT_DIR / "data"

RAW = "raw"
PROCESSED = "processed"

RAW_DATA_DIR = DATA_FOLDER / RAW
PROCESSED_DATA_DIR = DATA_FOLDER / PROCESSED

SESMIC_DATA_CSV = RAW_DATA_DIR / "seismic_data.csv"
IQUIQUE_FOLDER_PATH = ROOT_DIR / 'iquique'
PHASENET_MODELS_DIR = ROOT_DIR / 'models' / "phasenet"
