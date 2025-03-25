from langchain_openai import AzureChatOpenAI
import json
import ast

AZURE_OPENAI_API_KEY = "xxx"
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

def llm_accuracy_evaluation(answer, solution): 
    
    system_message = """
                You are a specialized assistant tasked with evaluating the accuracy of the final answer against the ground truth.
                """
    
    extract_prompt = f"""
    Here is the final answer: {solution}
    Here is the ground truth answer: {answer}

    Please extract the final answer from the solution and compare it to the ground truth answer.

    Return a binary result:
    - Respond with either "Correct" or "Incorrect" based on the accuracy of final answer.
    - Provide only "Correct" or "Incorrect" without any additional text.
    """

    msg = [
            ("system", system_message),
            ("human", extract_prompt),
        ]

    accuracy = llm_gpt4o.invoke(msg).content
    
    accuracy=accuracy.replace("```json", "").replace("```", "").strip()
    
    print("accuracy is ", accuracy)
    
    return str(accuracy)