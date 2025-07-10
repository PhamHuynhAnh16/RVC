"""Configuration of logging for various libraries and modules."""

import logging
import os
import warnings


def configure_logging(enable_configure_logging=True, global_logger=False, logging_level="WARNING"):
    """
    This function sets logging levels for various libraries and modules
    to reduce the number of output messages and improve log readability.

    Parameters:
    - enable_configure_logging (bool, optional):
        Main switch for the entire logging configuration.
        If set to False, the function will not perform any actions.
      Default: True

    - global_logger (bool, optional):
        Flag to enable or disable the global logger.
        If set to True, configures the logging level for all loggers.
        If False, configures the logging level only for specified libraries.
      Default: False

    - logging_level (str, optional):
        Custom logging level.
        Must be one of the following: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
        If an invalid value is provided, the default "WARNING" level will be used.
      Default: "WARNING"

    Logging Levels:
    - 0 | DEBUG: Detailed information, typically of interest only when debugging issues.
    - 1 | INFO: Confirmation that things are working as expected.
    - 2 | WARNING: An indication that something unexpected happened, or an indication
                   of a problem in the near future (e.g., 'disk space low').
                   The program still operates as expected.
    - 3 | ERROR: Due to a more serious problem, the program cannot perform some functions.
    - 4 | CRITICAL: Indicates that the program may not be able to continue running.

    In this case, we set the WARNING logging level for all libraries and modules
    to ignore DEBUG and INFO level messages.
    """

    if enable_configure_logging:
        # ===== Configuration of environment variables for dependencies ===== #
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

        # ===== Handling system warnings ===== #
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Get the logging level from the string
        level = getattr(logging, logging_level, logging.WARNING)

        # ===== Configuration of third-party library loggers ===== #
        if global_logger:
            logging.basicConfig(level=level)
        else:
            logging.getLogger("pydub").setLevel(level)
            logging.getLogger("numba").setLevel(level)
            logging.getLogger("faiss").setLevel(level)
            logging.getLogger("torio").setLevel(level)
            logging.getLogger("httpx").setLevel(level)
            logging.getLogger("urllib3").setLevel(level)
            logging.getLogger("fairseq").setLevel(level)
            logging.getLogger("asyncio").setLevel(level)
            logging.getLogger("httpcore").setLevel(level)
            logging.getLogger("matplotlib").setLevel(level)
            logging.getLogger("onnx2torch").setLevel(level)
            logging.getLogger("python_multipart").setLevel(level)


"""
Example usage of the configure_logging function in the main file:

1. With full parameters:
from logging_config import configure_logging
configure_logging(enable_configure_logging=True, global_logger=False, logging_level="DEBUG")

2. With abbreviated parameters (using default values for named arguments):
from logging_config import configure_logging
configure_logging(True, False, "DEBUG")

3. With default parameters (if no specific configuration is required):
from logging_config import configure_logging
configure_logging()
"""
