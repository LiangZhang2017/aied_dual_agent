import os
import json
from accuracy_llm_answer_extractor import llm_accuracy_evaluation

mode="single_agent"
topic="precalculus"
iter=2

file_path=os.path.join(os.path.dirname(os.getcwd()), "results", "outputs", "single_agent_xdeepseekv3_precalculus", f"iter_{iter}")

accuracy_list=[]

# Loop through all files in dataset_path
for filename in os.listdir(file_path):
    # Check for .json extension
    if filename.endswith(".json"):
        json_file_path = os.path.join(file_path, filename)
        
        problem_id=os.path.splitext(filename)[0]
        
        # Read and parse the JSON file
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            print("data current is",data['answer'])
            
            true_answer=data['answer']
            solution=data['llm_solution']
            
            accuracy=llm_accuracy_evaluation(true_answer,solution)
            
            accuracy_list.append({
                "problem_id":problem_id,
                "accuracy":accuracy
            })

print("accuracy_list is ", accuracy_list)

# Count the total number of problems
# total_problems_accuracy = len(accuracy_list)
total_problems_accuracy = 100

# Count the number of correct solutions
correct_count = sum(1 for entry in accuracy_list if entry['accuracy'] == 'Correct')

# Calculate the percentage of correct solutions
correct_percentage = (correct_count / total_problems_accuracy) * 100

# Display the result
print(f"Correct solutions: {correct_percentage:.2f}%")

# Define the evaluation results dictionary
evaluation_results = {
    "accuracy_list": accuracy_list,
    "correct_percentage": round(correct_percentage, 2)  # Round to 2 decimal places
}

evaluation_file_path = os.path.join(os.getcwd(), "evaluation", "results", f"{mode}_{topic}_evaluation_iter_{iter}.json")

# Ensure the directory exists
os.makedirs(os.path.dirname(evaluation_file_path), exist_ok=True)
# Save evaluation results as JSON
with open(evaluation_file_path, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Evaluation results saved to {evaluation_file_path}")