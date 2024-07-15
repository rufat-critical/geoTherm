import logging

class makeLager:
    # Create a lager instance to use in geoTherm

    def __init__(self, log_level=logging.DEBUG):
        # Create Logger
        # log_file = 'app.log'
        self.logger = logging.getLogger('geoTherm')
        self.logger.setLevel(log_level)

        # Create a file handler
        #file_handler = logging.FileHandler(log_file)
        #file_handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #console_handler.setFormatter(formatter)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def warn(self, msg):
        self.logger.warn(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

# Create a logger to use in geoTherm
logger = makeLager()
