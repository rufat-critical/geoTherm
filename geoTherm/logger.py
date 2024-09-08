import logging


class makeLager:
    """
    A logger class to create and manage a logging instance for geoTherm.
    """

    def __init__(self, log_level=logging.DEBUG):
        """
        Initialize the logger instance with the specified logging level.

        Args:
            log_level (int): Logging level to use (default is logging.DEBUG).
        """
        # Create a logger instance with the specified name
        self.logger = logging.getLogger('geoTherm')
        self.logger.setLevel(log_level)

        # Create a console handler for logging to the console
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(log_level)

        # Create a formatter and set it to the console handler
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        self.console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        self.logger.addHandler(self.console_handler)

    def set_level(self, level):
        """
        Set the logging level.

        Args:
            level (str or int): Logging level to use. Can be 'silent', 'debug',
                                'info', 'warning', 'error', or 'critical'.
        """
        if isinstance(level, str):
            level = level.lower()
            levels = {
                # A level higher than CRITICAL to silence logging
                'silent': logging.CRITICAL + 1,
                'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL
            }
            log_level = levels.get(level, logging.DEBUG)
        else:
            log_level = level

        self.logger.setLevel(log_level)
        self.console_handler.setLevel(log_level)

    def warn(self, msg):
        """
        Log a warning message.

        Args:
            msg (str): The message to log.
        """
        self.logger.warning(msg)

    def debug(self, msg):
        """
        Log a debug message.

        Args:
            msg (str): The message to log.
        """
        self.logger.debug(msg)

    def info(self, msg):
        """
        Log an informational message.

        Args:
            msg (str): The message to log.
        """
        self.logger.info(msg)

    def error(self, msg):
        """
        Log an error message.

        Args:
            msg (str): The message to log.
        """
        self.logger.error(msg)

    def critical(self, msg):
        """
        Log a critical error message and raise a RuntimeError.

        Args:
            msg (str): The message to log.

        Raises:
            RuntimeError: Always raised after logging the critical error
            message.
        """
        self.logger.critical(msg)
        raise RuntimeError(
            "A critical error has occurred in geoTherm. Please check the log "
            "messages above for details and fix the issues before re-running "
            "geoTherm."
        )


logger = makeLager()

# Example usage:
if __name__ == "__main__":
    logger = makeLager(log_level=logging.DEBUG)
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    try:
        logger.error("This is an error message.")
    except RuntimeError as e:
        print(e)
    try:
        logger.critical("This is a critical message.")
    except RuntimeError as e:
        print(e)
