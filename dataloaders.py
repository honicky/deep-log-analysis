import logparser.Drain as Drain
import os
import pandas as pd
import requests
import zipfile

LOG_DIR = "data"
OUTPUT_DIR = "result"

# I took the following log formats and regex patterns from
#  https://github.com/HC-Guo/LogFormer/blob/main/parse_log.py
hdfs_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
bgl_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
openstack_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'

bgl_regex = [
    r'core\.\d+',
    r'(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'0x[0-9a-f]+(?: [0-9a-f]{8})*',  # hex
    r'[0-9a-f]{8}(?: [0-9a-f]{8})*',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

hdfs_regex = [
    r'blk_(|-)[0-9]+',  # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

openstack_regex = [
    r'(?<=\[instance: ).*?(?=\])',
    r'(?<=\[req).*(?= -)',
    r'(?<=image ).*(?= at)',
    r'(?<=[^a-zA-Z0-9])(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=\s|=)\d+(?:\.\d+)?'
]

datasets = {
    "BGL": {
        "url": "https://zenodo.org/records/8196385/files/BGL.zip?download=1",
        "zip_file_name": "BGL.zip",
        "file_name": "BGL.log",
        "log_format": bgl_format,
        "regex": bgl_regex,
    },
    "HDFS": {
        "url": "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
        "zip_file_name": "HDFS_v1.zip",
        "file_name": "HDFS.log",
        "log_format": hdfs_format,
        "regex": hdfs_regex,
    }
}

def download_data(url: str, file_name: str):
    """
    Downloads a data file from a given URL and saves it to the data directory if it doesn't exist already.

    Args:
        url (str): The URL to download the file from
        file_name (str): The name to save the downloaded file as
    """

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(f"{LOG_DIR}/{file_name}"):
        response = requests.get(url)
        with open(f"{LOG_DIR}/{file_name}", "wb") as f:
            f.write(response.content)

def unzip_data(zip_file_name: str, log_file_name: str, base_dir: str = LOG_DIR):
    """
    Extracts a log file from a zip archive in the data directory.

    Args:
        zip_file_name (str): Name of the zip file to extract from
        log_file_name (str): Name of the log file to extract
    """

    log_file_path = f"{base_dir}/{log_file_name}"
    zip_file_path = f"{LOG_DIR}/{zip_file_name}"
    if not os.path.exists(log_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extract(log_file_name, base_dir)

def parse_dataset(dataset_name: str):
    """
    Parses a log dataset using the Drain parser and saves the structured output.
    
    Args:
        dataset_name (str): Name of the dataset to parse (must be defined in datasets dict)
    
    Note:
        The parsed output will be saved as a CSV file in the output directory
        with the name pattern: {log_file_name}_structured.csv
    """
    log_format = datasets[dataset_name]["log_format"]
    regex = datasets[dataset_name]["regex"]
    log_file_name = f'{datasets[dataset_name]["file_name"]}'

    structured_file_path = f"{OUTPUT_DIR}/{log_file_name}_structured.csv"
    if not os.path.exists(OUTPUT_DIR) or \
        not os.path.exists(structured_file_path):
        
        print(f"{structured_file_path} does not exist")
        print(f"Parsing {dataset_name} dataset...")
        parser = Drain.LogParser(log_format, indir=LOG_DIR, rex=regex)
        parser.parse(log_file_name)
    
    return structured_file_path

def add_hdfs_blockid_column(structured_df: pd.DataFrame):
    """
    Adds a column to the structured dataframe that contains the HDFS block ID
    """

    structured_df["BlockId"] = structured_df.Content.str.extract(r'(blk_-?\d+)', expand=False)
