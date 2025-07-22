import json
import csv

# Load the JSON file
with open('assets/entities.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write to CSV
with open('assets/entities.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow(['chunk_id', 'entity_text', 'label'])
    # Write rows
    for i, item in enumerate(data):
        chunk_id = item.get('chunk_id', i)  
        entities = item.get('entities', [])
        label = None

        if entities:
            for entity in entities:
                label = entity['label']

                if label not in ['CARDINAL', 'ORDINAL', 'QUANTITY', 'TIME', 'MONEY', 'PERCENT']:
                    writer.writerow([
                        entity.get('chunk_id', chunk_id),
                        entity['text'],
                        label
                    ])
    
