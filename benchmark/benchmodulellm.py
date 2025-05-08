import argparse
import os
import sys

import yaml
import logging

from pathlib import Path

from utils import LLMClient

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_opt(known=False):
    """
    Parse command-line options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ModuleLLM IP Address")
    parser.add_argument("--port", type=int, default=10001, help="ModuleLLM TCP Port")
    parser.add_argument("--test-items", type=str, default=ROOT / "default.yaml", help="testitems.yaml path")

    args = parser.parse_known_args()[0] if known else parser.parse_args()

    return args

def read_yaml(file_path):
    """
    Read a YAML file and return its content.
    """
    if not os.path.exists(file_path):
        logging.error(f"YAML file '{file_path}' does not exist.")
        sys.exit(1)
    
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            if data is None:
                logging.warning(f"YAML file '{file_path}' is empty.")
                return {}
            
            logging.info(f"YAML file '{file_path}' read successfully.")
            
            if "items" in data:
                return data["items"]
            else:
                logging.warning(f"'items' not found in YAML file.")
                return []
    except Exception as e:
        logging.error(f"Failed to read YAML file '{file_path}': {e}")
        sys.exit(1)

def write_yaml(file_path, data):
    """
    Write data to a YAML file.
    """
    try:
        with open(file_path, "w") as file:
            yaml.safe_dump(data, file)
            logging.info(f"YAML file '{file_path}' written successfully.")
    except Exception as e:
        logging.error(f"Failed to write YAML file '{file_path}': {e}")
        sys.exit(1)

def categorize_and_deduplicate(items):
    """
    Categorize items by 'type' and remove duplicate 'model_name'.
    """
    categorized = {}
    for item in items:
        item_type = item.get("type")
        model_name = item.get("model_name")
        if not item_type or not model_name:
            continue
        
        if item_type not in categorized:
            categorized[item_type] = set()
        
        categorized[item_type].add(model_name)
    
    # Convert sets back to lists for easier usage
    return {key: list(value) for key, value in categorized.items()}

def main(opt):
    items = read_yaml(opt.test_items)
    if not items:
        logging.warning(f"No items found in YAML file '{opt.test_items}'.")
        return
    
    categorized_items = categorize_and_deduplicate(items)
    
    logging.info("Categorized items:")
    for item_type, models in categorized_items.items():
        logging.info(f"Type: {item_type}, Models: {models}")
        
        if item_type == "llm":
            logging.info("Initializing LLMClient...")
            llm_client = LLMClient(opt.host, opt.port)
            
            for model_name in models:
                logging.info(f"Testing model: {model_name}")
                input_text = "Tell me an adventure story."
                try:
                    result = llm_client.test(model_name, input_text)
                    logging.info(f"Test result for model '{model_name}': {result}")
                except Exception as e:
                    logging.error(f"Error testing model '{model_name}': {e}")
            
            del llm_client
            logging.info("LLMClient deleted successfully.")
    
    return categorized_items

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
