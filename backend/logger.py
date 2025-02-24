import os
import logging
import configparser

config = configparser.ConfigParser()
config.read("../config/config.ini")

# Ensure the logs directory exists
LOG_DIR = config["directory"]["log"]
os.makedirs(LOG_DIR, exist_ok=True)

BACKEND_LOG_FILE = os.path.join(LOG_DIR, "backend.log")
GRAPH_LOG_FILE = os.path.join(LOG_DIR, "graph.log")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Backend Logger
backend_logger = logging.getLogger("backend")
backend_logger.setLevel(logging.DEBUG)

backend_file_handler = logging.FileHandler(BACKEND_LOG_FILE, mode="a")
backend_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

backend_console_handler = logging.StreamHandler()
backend_console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

backend_logger.addHandler(backend_file_handler)
backend_logger.addHandler(backend_console_handler)

# Graph Logger
graph_logger = logging.getLogger("graph")
graph_logger.setLevel(logging.DEBUG)

graph_file_handler = logging.FileHandler(GRAPH_LOG_FILE, mode="a")
graph_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

graph_console_handler = logging.StreamHandler()
graph_console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

graph_logger.addHandler(graph_file_handler)
graph_logger.addHandler(graph_console_handler)

