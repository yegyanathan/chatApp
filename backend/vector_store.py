import os
import configparser

from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

config = configparser.ConfigParser()
config.read("../config/config.ini")

MILVUS_URI = config["milvus"]["uri"]
AUTO_ID = config.getboolean("milvus", "auto_id")
INDEX_TYPE = config["milvus"]["index_type"]
METRIC_TYPE = config["milvus"]["metric_type"]
EMBED_MODEL_NAME = config["embedding"]["name"]
DB_DIR = config["directory"]["db"]

os.makedirs(DB_DIR, exist_ok=True)

vector_store = Milvus(
    embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME),
    connection_args={"uri": MILVUS_URI},
    index_params={"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE},
    auto_id=AUTO_ID
)