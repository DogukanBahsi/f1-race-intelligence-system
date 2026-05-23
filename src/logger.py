"""
src/logger.py
-------------
F1 Race Intelligence System - Merkezi Logging Modülü
Tüm modüller bu logger'ı kullanır.
"""

import logging
import sys
from pathlib import Path

# Config'i import et (döngüsel import olmaması için sadece path)
try:
    from config import LOG_FILE, LOG_LEVEL
except ImportError:
    LOG_FILE  = Path(__file__).resolve().parent.parent / "f1_system.log"
    LOG_LEVEL = "INFO"


class ColoredFormatter(logging.Formatter):
    """Terminal'de renkli log çıktısı için özel formatter."""

    COLORS = {
        "DEBUG":    "\033[36m",   # Cyan
        "INFO":     "\033[32m",   # Yeşil
        "WARNING":  "\033[33m",   # Sarı
        "ERROR":    "\033[31m",   # Kırmızı
        "CRITICAL": "\033[35m",   # Magenta
        "SUCCESS":  "\033[92m",   # Parlak yeşil
    }
    RESET = "\033[0m"

    def format(self, record):
        color    = self.COLORS.get(record.levelname, self.RESET)
        message  = super().format(record)
        return f"{color}{message}{self.RESET}"


def get_logger(name: str = "F1System") -> logging.Logger:
    """
    Verilen isimde bir logger döndürür.
    İlk çağrıda handler'ları kurar.
    """
    logger = logging.getLogger(name)

    # Zaten kurulmuşsa tekrar kurma
    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # ── Terminal (stdout) handler ─────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = ColoredFormatter(
        "[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ── Dosya handler ────────────────────────────────────────
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except Exception:
        # Dosya yazılamazsa sadece console ile devam et
        pass

    # SUCCESS seviyesini ekle (logging'de varsayılan yok)
    logging.SUCCESS = 25  # INFO (20) ile WARNING (30) arası
    logging.addLevelName(logging.SUCCESS, "SUCCESS")

    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.SUCCESS):
            self._log(logging.SUCCESS, message, args, **kwargs)

    logging.Logger.success = success

    return logger


# ─────────────────────────────────────────────────────────────
# Kullanışlı kısayol fonksiyonlar
# ─────────────────────────────────────────────────────────────
_default_logger = get_logger("F1System")

def log_info(msg: str):    _default_logger.info(msg)
def log_warning(msg: str): _default_logger.warning(msg)
def log_error(msg: str):   _default_logger.error(msg)
def log_debug(msg: str):   _default_logger.debug(msg)

def log_success(msg: str):
    """Başarı mesajı - yeşil renkte gösterilir."""
    _default_logger.log(logging.SUCCESS, msg)


if __name__ == "__main__":
    logger = get_logger("Test")
    logger.debug("Bu bir debug mesajıdır.")
    logger.info("Loading Monza GP 2025...")
    logger.info("Raw data saved.")
    logger.info("Preprocessing started.")
    logger.info("Feature engineering completed.")
    logger.info("Decision Tree model trained.")
    log_success("Dashboard data is ready.")
    logger.warning("Veri eksikliği tespit edildi.")
    logger.error("FastF1 data could not be loaded.")
