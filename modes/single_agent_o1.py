import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import time

class single_agent_o1:
    def __init__(self, parameters):
        self.parameters = parameters
        
        env_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Retrieve the OpenAI API key from environment variables
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found. Please set it in the environment variables.")
        
        self.openai_api_key=openai_api_key
        print("openai_api_key is", openai_api_key)
        
        # Get the model to use (default to 'gpt-4' if not specified)
        self.llm_model = self.parameters.get('role_1_llm')
        print("LLM model is", self.llm_model)
        
        # Retrieve the dataset from parameters
        self.data = self.parameters.get('dataset')
    
    def solve_math_problem(self, problem_statement):
        
        # answer_json = {
        #         "solution": "Step-by-step approach detailing the solution",
        #         "final_answer": "The final outcome"
        #     }
        
        query_prompt = f'''You are a helpful assistant for solving math problems. Please solve the following problem: {problem_statement}.
                            Provide your answer in JSON format with the following keys "solution" and "final_answer". 
                            Ensure the response includes only this JSON structure without any additional text or characters.
                            Let's solve this problem step by step and provide your answer.
                        '''
        
        client=OpenAI(
            api_key=self.openai_api_key,
        )
        
        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": query_prompt
                        }
                    ]
                )
                response = completion.choices[0].message.content
                print("Response:", response)
                return response

            except OpenAIError as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. Unable to get a valid response.")
                    return None
        
        print("response  is ", response)
        
        # Access the content attribute to get the text response
        return response.strip()
    
    def run(self):
        print("Dataset columns are:", self.data.columns)
        
        mode = self.parameters.get('mode')
        role_llm_str = self.parameters.get('role_1_llm')
        
        output_dir = os.path.join(os.getcwd(), 'results', 'outputs')
        output_file_path = os.path.join(output_dir, f'{mode}_{role_llm_str}_outputs.json')

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Iterate through each row in the DataFrame
        for index, row in self.data.iterrows():
            problem_id = row['id']
            problem_text = row['problem']
            
            try:
                # Solve the problem using the provided method
                llm_solution = self.solve_math_problem(problem_text)
                # Assign the solution to the 'llm_solution' column for the current row
                self.data.at[index, 'llm_solution'] = llm_solution
            except Exception as e:
                # Handle any exceptions that occur during problem-solving
                print(f"An error occurred while solving problem ID {problem_id}: {e}")
                self.data.at[index, 'llm_solution'] = None

        # Save the updated DataFrame to a JSON file
        self.data.to_json(output_file_path, orient='records', indent=4)