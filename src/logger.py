import time
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def log_execution_time(logger: logging.Logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start) * 1000  # in milliseconds
            logger.info(f"{func.__qualname__} executed in {duration:.2f} ms")
            return result
        return wrapper
    return decorator
