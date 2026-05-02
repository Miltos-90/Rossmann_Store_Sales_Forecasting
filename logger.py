
import logging
import os

LOG_DIR = "./artifacts"
LOG_FILE = "hypertuning.log"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,  # ensure no duplicate handlers if this cell is re-run in a notebookcls
)

logger = logging.getLogger(__name__)
