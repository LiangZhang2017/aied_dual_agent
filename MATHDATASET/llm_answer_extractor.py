from langchain_openai import AzureChatOpenAI
import json
import ast

AZURE_OPENAI_API_KEY = "xxxx"
AZURE_OPENAI_ENDPOINT = "xxx"
AZURE_OPENAI_VERSION = "xxx"
AZURE_OPENAI_MODEL ="gpt-4o"

llm_gpt4o = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_VERSION,
    azure_deployment=AZURE_OPENAI_MODEL,
    temperature=0.5
)

def llm_answer_extract(problem, solution): 
    
    system_message = """
                You are a specialized assistant tasked with extracting the final answer from the solution. The answer is already included in the solution.
                """
    
    extract_prompt = f"""
    Here is the problem: {problem}  
    Solution: {solution}  

    Extract and return only the final answer:  
    - Output only the answer, nothing else.  
    - Include the unit if present.  
    - If multiple answer formats exist in the solution, include all of them; otherwise, return just one.  
    """

    msg = [
            ("system", system_message),
            ("human", extract_prompt),
        ]

    answer = llm_gpt4o.invoke(msg).content
    
    answer=answer.replace("```json", "").replace("```", "").strip()
    
    print("answer is ", answer)
    
    return answer
    
    