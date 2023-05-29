import logging

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('logs.log')

# Set the logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


def trim_string(string, count: int):
    words = string.split()
    trimmed_words = words[:count]
    trimmed_string = ' '.join(trimmed_words)
    return trimmed_string