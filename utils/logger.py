import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

enable_console = False


if not logger.handlers:
    if enable_console:
        shell_handler = RichHandler(console=Console(file=sys.stderr))
        shell_handler.setLevel(logging.DEBUG)
        shell_formatter = logging.Formatter("%(message)s")
        shell_handler.setFormatter(shell_formatter)
        logger.addHandler(shell_handler)

    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

logger.propagate = False
