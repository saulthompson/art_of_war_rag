import json
import csv
from typing import List, Dict, Any

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file and return its contents as a list of dictionaries.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: Parsed JSON content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_entities_to_csv(data: List[Dict[str, Any]], csv_path: str) -> None:
    """
    Write filtered entity data from JSON to a CSV file.

    Args:
        data (List[Dict[str, Any]]): The JSON data loaded from file.
        csv_path (str): Path to the output CSV file.
    """
    excluded_labels = {'CARDINAL', 'ORDINAL', 'QUANTITY', 'TIME', 'MONEY', 'PERCENT'}

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['chunk_id', 'entity_text', 'label'])  # Write header

        for i, item in enumerate(data):
            chunk_id = item.get('chunk_id', i)
            entities = item.get('entities', [])

            for entity in entities:
                label = entity.get('label')
                if label and label not in excluded_labels:
                    writer.writerow([
                        entity.get('chunk_id', chunk_id),
                        entity['text'],
                        label
                    ])

if __name__ == "__main__":
    json_data = load_json_file('assets/entities.json')
    write_entities_to_csv(json_data, 'assets/entities.csv')
