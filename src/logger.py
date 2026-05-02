
import logging

from src.constants import LOG_FILE

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,  # ensure no duplicate handlers if this cell is re-run in a notebookcls
)
