import logging
import os


def setup_logging(log_file="vertical_resolution_test.log"):
    """
    Set up centralized logging configuration for all scripts in the test suite.

    Args:
        log_file (str): Name of the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, log_file)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info(f"Logging initialized. Log file: {log_path}")
