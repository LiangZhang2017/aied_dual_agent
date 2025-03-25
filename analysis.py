import os
import json
from helper import single_agent_extract_answer, compare_answers

mode="dual_agent_rcp_gpt-4o"

# file_name="single_agent_o1-preview_outputs.json"
# file_name="single_agent_gpt-4o_outputs.json"

file_name="dual_agent_rcp_gpt-4o_outputs.json"
output_file=os.path.join(os.getcwd(),"results","outputs", file_name)

if os.path.exists(output_file):
    with open(output_file, 'r') as file:
        try:
            data = json.load(file)
            # print(json.dumps(data, indent=4))  # Pretty-print the JSON data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
else:
    print(f"The file '{output_file}' does not exist.")

# Initialize counters for correctness calculation
total_entries = len(data)
correct_count = 0

for entry in data:
    print(f"Problem ID: {entry['id']}")
    true_answer=entry['answer']
    
    if "single_agent" in mode:
        true_solution=entry['solution']
        llm_solution=entry['llm_solution']
        llm_answer=single_agent_extract_answer(llm_solution)
    
    else:     
        llm_answer= entry['llm_answer']
    
    print("true_answer is ", true_answer)
    print("llm_answer is ", llm_answer)
    
    # Determine correctness
    correctness = compare_answers(true_answer, llm_answer)
    if correctness:
        correct_count += 1

    print("correctness is ", correctness)
    
    # Update the entry with LLM's answer and correctness
    entry['llm_answer'] = llm_answer
    entry['correctness'] = correctness

# Calculate the percentage of correctness
correctness_percentage = (correct_count / total_entries) * 100
print(f"Correctness Percentage: {correctness_percentage:.2f}%")
    
new_output_file = os.path.join(os.getcwd(), "results", "analysis", f"{mode}_correctness.json")
os.makedirs(os.path.dirname(new_output_file), exist_ok=True)

with open(new_output_file, 'w') as file:
    json.dump(data, file, indent=4)