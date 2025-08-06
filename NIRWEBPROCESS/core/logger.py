import time
import logging, os, json
from datetime import datetime
from .config import settings

os.makedirs(settings.logging_dir, exist_ok=True)
LOG_FILE = os.path.join(settings.logging_dir, f"nir_api_{datetime.now().strftime('%Y%m%d')}.log")

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name
        }
        return json.dumps(log_record)

handler = logging.FileHandler(LOG_FILE)
handler.setFormatter(JsonFormatter())

logger = logging.getLogger("nir_logger")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def log_info(msg: str):
    logger.info(msg)

def log_error(msg: str):
    logger.error(msg)

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        log_info(f"Func {func.__name__} executada em {duration:.3f} segundos")
        return result
    return wrapper
