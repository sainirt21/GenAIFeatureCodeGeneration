import json
import os
import random

input_folder_path = '/Users/ritusaini/Documents/OpenAI'
training_file_path = 'training_data.jsonl'
validation_file_path = 'validation_data.jsonl'

validation_split = 0.2
all_entries = []

for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.json'):
        input_file_path = os.path.join(input_folder_path, file_name)
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                for step, details in data.items():
                    system_message = "You are an AI assistant that provides accurate and concise responses."
                    user_message = step.strip()
                    assistant_message = details.get("code", "").strip()
                    
                    jsonl_entry = {
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": assistant_message}
                        ]
                    }
                    all_entries.append(jsonl_entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_name}: {e}")

random.shuffle(all_entries)
split_index = int(len(all_entries) * (1 - validation_split))
training_data = all_entries[:split_index]
validation_data = all_entries[split_index:]

with open(training_file_path, 'w', encoding='utf-8') as train_file:
    for entry in training_data:
        train_file.write(json.dumps(entry) + '\n')

with open(validation_file_path, 'w', encoding='utf-8') as val_file:
    for entry in validation_data:
        val_file.write(json.dumps(entry) + '\n')