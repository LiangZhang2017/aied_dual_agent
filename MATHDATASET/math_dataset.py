import os
import json
import random
from llm_answer_extractor import llm_answer_extract

"""
Topics:
1. algebra
2. counting_and_probability
3. geometry
4. intermediate_algebra
5. number_theory
6. prealgebra
7. precalculus
"""

topic="precalculus"
level="Level 5"

dataset_path_train = os.path.join(os.getcwd(), "MATHDATASET", "MATH", "test", topic)

print("dataset path is ", dataset_path_train)

all_json_data = []  # list to store parsed content of each JSON file
level_5_data = []  # list to store JSON objects with level == "Level 5"

# Loop through all files in dataset_path
for filename in os.listdir(dataset_path_train):
    # Check for .json extension
    if filename.endswith(".json"):
        json_file_path = os.path.join(dataset_path_train, filename)
        
        # Read and parse the JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data["id"] = os.path.splitext(filename)[0]
            
        all_json_data.append(data)
        
        # Check if this JSON file has "level" == "Level 5"
        # (assuming each file is a single JSON object with a "level" key)
        if data.get("level") == level:
            level_5_data.append(data)
        
# Now, all_json_data holds the parsed content for each .json file
print(f"Total JSON files found: {len(all_json_data)}")    
print(f"level_5_data includes: {len(level_5_data)} ")

# Set a fixed seed for repeatable random selection.
SEED_VALUE = 42
random.seed(SEED_VALUE)

# Ensure we have at least 100 items; otherwise, handle accordingly
if len(level_5_data) < 100:
    print(f"Only {len(level_5_data)} entries available, cannot randomly select 100.")
    level_5_random_100_data = level_5_data  # or handle differently
else:
    # Randomly sample 100 unique items
    level_5_random_100_data = random.sample(level_5_data, 100)

print(len(level_5_random_100_data))
# print(level_5_random_100_data[0])

# Loop over each item in level_5_random_100_data
for idx, item in enumerate(level_5_random_100_data, start=1):
    print(f"Entry #{idx}")
    print(f"  ID: {item['id']}")
    print(f"  Level: {item.get('level', 'N/A')}")
    print()
    
    print(item["problem"])
    print(item["solution"])
    
    answer=llm_answer_extract(problem=item["problem"],solution=item["solution"])
    
    # answer="answer"
    
    item["answer"]=answer

# 4. Save updated data back to a new JSON file
output_filename = f"{topic}_level_5_random_100_with_answers.json"
output_dir = os.path.join(os.getcwd(), "MATHDATASET", "math_100", topic)
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, output_filename)
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(level_5_random_100_data, outfile, indent=4)