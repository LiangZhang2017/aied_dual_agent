import json
import re

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())
            
def single_agent_extract_answer(response_content):
    try:
        # Attempt to parse the entire response_content as JSON
        json_data = json.loads(response_content)
        if 'final_answer' in json_data:
            return json_data['final_answer']
    except json.JSONDecodeError:
        # If parsing fails, proceed to extract JSON snippet using regex
        pass

     # Regex pattern to extract JSON object enclosed within ```json and ```
    json_match = re.search(r'```json\n({.*?})\n```', response_content, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
        # Replace escaped newlines and tabs with actual characters
        json_string = json_string.replace('\\n', '\n').replace('\\t', '\t')
        # Remove any other escape sequences
        json_string = re.sub(r'\\(.)', r'\1', json_string)
        try:
            json_data = json.loads(json_string)
            if 'final_answer' in json_data:
                return json_data['final_answer']
        except json.JSONDecodeError:
            pass

    answer_match = re.search(r'"final_answer"\s*:\s*("[^"]*"|\d+)', response_content)
    if answer_match:
        return answer_match.group(1).strip('"')

    return None  # Return None if 'final_answer' is not found

def extract_numerical_values(text):
    # Regular expression to find all numerical values, including integers, decimals, and scientific notation
    return re.findall(r'[-+]?\d*\.\d+|\d+', str(text))

def compare_answers(true_answer, llm_answer):
    # Extract numerical values from both answers
    true_values = extract_numerical_values(true_answer)
    llm_values = extract_numerical_values(llm_answer)
    
    # Convert extracted strings to floats for comparison
    true_values = [float(value) for value in true_values]
    llm_values = [float(value) for value in llm_values]
    
    # Compare the lists of numerical values
    return true_values == llm_values