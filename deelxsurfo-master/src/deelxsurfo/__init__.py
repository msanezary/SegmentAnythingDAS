"""File: __init__.py | Author: agarc | date: Sat Sep 21 2024

Description: Logger configuration
"""

__version__ = "0.1.0"

# You may want to perform packages imports here if you are using builders.

# from templator.module.foo import foo  # noqa: ERA001
# from templator.settings.settings import Settings  # noqa: ERA001

# __all__ = [  # noqa: ERA001, RUF100
#     "foo",  # noqa: ERA001
#     "Settings",  # noqa: ERA001
#     ] # noqa: ERA001, RUF100

# Setting up logging options for the project
import logging


# ANSI escape codes for colors
class LogFormatter(logging.Formatter):
    """Enable different color modes for different log levels."""

    # Color codes for different log levels
    COLORS = {  # noqa: RUF012
        "DEBUG": "\033[36m",  # Cyan.
        "INFO": "\033[32m",  # Green.
        "WARNING": "\033[33m",  # Yellow.
        "ERROR": "\033[31m",  # Red.
        "CRITICAL": "\033[1;31m",  # Bold Red.
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003, RUF100
        """Standard log formatter with color support."""
        log_color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{log_color}{message}{self.RESET}"


def setup_logging() -> None:
    """Logging setup with color support."""
    # Create file handler (no color for file logs)
    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    color_formatter = LogFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(color_formatter)

    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,  # Set the default logging level
        handlers=[
            file_handler,  # Log to a file
            console_handler,  # Log to the console with colors
        ],
    )


# Call the setup_logging to configure logging
setup_logging()

if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
