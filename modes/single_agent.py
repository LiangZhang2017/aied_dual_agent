import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import time
import json
import openai
from langchain_openai import AzureChatOpenAI
from azure.core.exceptions import AzureError
from langchain_deepseek import ChatDeepSeek

env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path=env_path, override=True)

AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION')
AZURE_OPENAI_MODEL=os.getenv('AZURE_OPENAI_MODEL')

class single_agent:
    def __init__(self, parameters):
        self.parameters = parameters
        
        # Get the model to use (default to 'gpt-4' if not specified)
        self.llm_model = self.parameters.get('role_1_llm')
        print("LLM model is", self.llm_model)
        
        # Retrieve the dataset from parameters
        self.data = self.parameters.get('dataset')
        print("self.data is ", self.data)
        
        self.topic=parameters['topic']
    
    def solve_math_problem(self, problem_statement):
        
        query_prompt = f"""You are a helpful assistant for solving math problems. Please solve the following problem: {problem_statement}.
                            Provide your answer in JSON format with the following keys "solution" and "final_answer". 
                            Ensure the response includes only this JSON structure without any additional text or characters.
                            Let's solve this problem step by step and provide your answer.
                            Use only the English language. 
            """
            
        if self.llm_model == "gpt-4o":
            
            llm = AzureChatOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )
            
            max_retries=5
            retry_delay=2
            
            for attempt in range(max_retries):
                try:
                    # Invoke the model
                    response = llm.invoke(query_prompt).content
                    print("Response using Azure:", response)
                    return response.strip()

                except Exception as e:  # or `Exception` if you prefer a broader catch
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Unable to get a valid response.")
                        return None
        
        # Option for using DeepSeek
        elif self.llm_model == "xdeepseekv3":
            # Retrieve DeepSeek API credentials from environment variables
            deepseek_api_key = os.getenv('DEEPEEK_API_KEY')
            if not deepseek_api_key:
                raise ValueError("DeepSeek API key not found. Please set it in the environment variables.")
            deepseek_api_base = os.getenv('DEEPEEK_API_BASE', 'http://maas-api.cn-huabei-1.xf-yun.com/v1')
            
            llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=deepseek_api_key,
                api_base=deepseek_api_base
            )
            
            max_retries = 5
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = llm.invoke(query_prompt).content
                    print("Response using DeepSeek:", response)
                    return response.strip()
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Unable to get a valid response.")
                        return None
        
        else:
            raise ValueError(f"Unsupported LLM model specified: {self.llm_model}")
        
    def run(self):
        
        mode = self.parameters.get('mode')
        role_llm_str = self.parameters.get('role_1_llm')

        for iteration in range(3):
            
            print("iteration is ", iteration+1)
            
            # Iterate through each row in the DataFrame
            for idx, row in enumerate(self.data, start=1):
                problem_id = row['id']
                problem_text = row['problem']
                
                output_dir = os.path.join(os.getcwd(), 'results', 'outputs', f'{mode}_{role_llm_str}_{self.topic}', f'iter_{iteration}')
                os.makedirs(output_dir, exist_ok=True)
                
                output_file_path = os.path.join(output_dir, f"problem_{problem_id}.json")
                
                try:
                    # Solve the problem using the provided method
                    llm_solution = self.solve_math_problem(problem_text)
                    # Assign the solution to the 'llm_solution' column for the current row
                    row['llm_solution']=llm_solution

                except Exception as e:
                    # Handle any exceptions that occur during problem-solving
                    print(f"An error occurred while solving problem ID {problem_id}: {e}")
                    row['llm_chat'] = ""

                print("current row is ", row)

                # Save final dataframe to JSON
                with open(output_file_path, 'w') as outfile:
                    json.dump(row, outfile, indent=2)