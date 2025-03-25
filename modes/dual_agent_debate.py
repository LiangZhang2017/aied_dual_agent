import os
from dotenv import load_dotenv
import time
import json

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek

env_path = os.path.join(os.getcwd(), '.env')        
load_dotenv(dotenv_path=env_path, override=True)

AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION')
AZURE_OPENAI_MODEL=os.getenv('AZURE_OPENAI_MODEL')

DEEPEEK_API_KEY = os.getenv('DEEPEEK_API_KEY')
DEEPEEK_API_BASE = os.getenv('DEEPEEK_API_BASE')
        
class dual_agent_debate:
    def __init__(self, parameters):
        self.parameters = parameters
        
        # Get the model to use (default to 'gpt-4' if not specified)
        self.role_1_llm_model = self.parameters.get('role_1_llm')
        print("LLM model role 1 is", self.role_1_llm_model)
        self.role_2_llm_model = self.parameters.get('role_2_llm')
        print("LLM model role 2 is", self.role_2_llm_model)
        
        # Retrieve the dataset from parameters
        self.data = self.parameters.get('dataset')
        self.topic=parameters['topic']
        
    def extract_final_answer_with_llm(self,conversation):
        """
        Uses an LLM to extract the final correct answer from the conversation.

        Parameters:
        conversation (str): The entire conversation as a single string.

        Returns:
        str: The final correct answer identified by the LLM.
        """      
        system_message = """
                You are a specialized assistant tasked with extracting the final answer in conversation records.
                """

        summary_prompt = (
                        "You are provided with a debate conversation between two debaters focused on solving a math problem. "
                        "Your task is to identify and extract the final answer provided by either debater in the conversation."
                        "Even if multiple answers are discussed during the process, extract only the final answer based on their concluding decisions. "
                        "Please provide only the final answer without any additional text.\n\n"
                        f"Conversation:\n{conversation}\n\n"
                        "Final answer:"
                    )
                            
        summary_llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_VERSION,
            azure_deployment=AZURE_OPENAI_MODEL,
            temperature=0.5
        )
        
        msg = [
            ("system", system_message),
            ("human", summary_prompt),
        ]
        
        llm_answer = summary_llm.invoke(msg).content
        llm_answer=llm_answer.replace("```json", "").replace("```", "").strip()
        
        print("llm_answer is", llm_answer)       
        return llm_answer

    def solve_math_problem(self, problem_statement):
        print("Solving math problem: ", problem_statement)
        
        # Initialize the LLM for Debater Agent A (use DeepSeek if role_1_llm_model equals "xdeepseekv3")
        if self.role_1_llm_model == "xdeepseekv3":
            debater_a_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            debater_a_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )
        
        # Initialize the LLM for Debater Agent B (use DeepSeek if role_2_llm_model equals "xdeepseekv3")
        if self.role_2_llm_model == "xdeepseekv3":
            debater_b_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            debater_b_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5,
            )
        
        debater_a_system_prompt = HumanMessagePromptTemplate.from_template(
                """
                You are **Debater Agent A**, an analytical and articulate individual specializing in mathematical problem solving.
                - Your role is to present an initial solution to the math problem below, defend your reasoning with clear, logical, and rigorous arguments, and maintain a distinct stance—even if Agent B agrees with parts of your solution. 
                - You should maintain confidence in your solution and uphold a distinct stance. Even if Agent B seems to agree or raises similar points, emphasize unique aspects of your approach or rationale to keep the debate engaging. 
                
                Present your solution to the following math problem, defend your reasoning, and critically evaluate Debater Agent B's arguments:\n\n
                {problem_statement}\n\n
                
                **Conversation History:**
                {conversation_history}
                
                 **Instructions:**
                1. **Opening Statement:** Clearly present your solution, detailing the proof or reasoning, key assumptions, and relevant theorems.
                2. **Rebuttal:** Address and counter Agent B's critiques by identifying any logical flaws or gaps and reinforcing your method with counterexamples or formal proof techniques.
                3. **Conclusion:** Summarize your key points, acknowledge valid insights from Agent B, and reaffirm why your solution is robust.
                4. If during the debate you identify errors or inaccuracies in either your solution or Agent B’s, incorporate these corrections into your final summary, ensuring that the final answer reflects the most accurate and refined reasoning.
                5. Engage in a respectful, constructive debate—challenge and refine each other's methods while keeping your presentation distinct from Agent B's. 
                
                **When the debate concludes, summarize the key points discussed and acknowledge the strengths of both solutions.**
                - **Do not repeat or paraphrase the teacher’s final message**.  
                - End with a simple and original farewell, such as "Thank you for your help!" or "Goodbye, and have a great day!" 
                - After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded.   
                - Output only the content in a dialogue tone, excluding words like 'Opening Statement,' 'Rebuttal,' 'Key Insights,' and any unrelated symbols, characters, or words. 
            
                Last Notes: 
                - Use only the English language.
                - Do not display role names; only show the generated content. 
                - Follow the provided guidance and generate content step by step.
                - Do not make assumptions or include future steps in the output.
            """
        )
                
        debater_b_system_prompt = HumanMessagePromptTemplate.from_template(
                """
                **You are Debater Agent B**, a methodical and logical thinker with expertise in mathematical problem solving.
                
                Your role is to analyze Debater Agent A's solution, present your own approach, and engage in a constructive debate to enhance the problem-solving process:\n\n
                f"{problem_statement}\n\n
                
                **Conversation History:**
                {conversation_history}
                
                **Instructions:**
                1. **Opening Statement:** Review Agent A's solution, identify any logical gaps, unstated assumptions, or potential flaws, and present your own approach or perspective. Be clear, concise, and ensure your reasoning is mathematically sound.
                2. **Rebuttal:** When responding to Agent A, directly address their points by using counterexamples, formal proof techniques (e.g., proof by contradiction or direct proof), and clear explanations to show why your perspective strengthens the argument.
                3. **Conclusion:** Summarize the key insights from the debate, acknowledge valid points made by Agent A, and reaffirm why your critique or alternative method adds value to the problem-solving process. 
                4. If during the debate you identify errors or inaccuracies in either your solution or Agent A’s, incorporate these corrections into your final summary, ensuring that the final answer reflects the most accurate and refined reasoning.
                5. Engage respectfully and constructively—challenge and refine each other's approaches without mirroring Agent A's style. After the debate concludes, do not initiate further dialogue.
                        
                **When the debate concludes, summarize the key insights gained from the discussion and recognize the merits of both solutions.**
                - **Do not repeat or paraphrase the teacher’s final message**.  
                - End with a simple and original farewell, such as "Thank you for your help!" or "Goodbye, and have a great day!" 
                - After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded.   
                - Output only the content in a dialogue tone, excluding words like 'Opening Statement,' 'Rebuttal,' 'Key Insights,' and any unrelated symbols, characters, or words. 
        
                Last Notes: 
                - Use only the English language.
                - Do not display role names; only show the generated content. 
                - Follow the provided guidance and generate content step by step.
                - Do not make assumptions or include future steps in the output.
        """
        )
        
        # Create the debaters' prompt templates with MessagesPlaceholder
        debater_a_prompt = ChatPromptTemplate.from_messages([
            debater_a_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        
        debater_b_prompt = ChatPromptTemplate.from_messages([
            debater_b_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        
        # Initialize conversation history
        conversation_history = []
        
        # Initialize a dictionary to store the conversation history for JSON output
        conversation_dict = {'chat': {}}
        
        # callback_handler = DialogueMonitorCallback()
            
        # Create the debaters' chains
        debater_a_chain = debater_a_prompt | debater_a_llm
        debater_b_chain = debater_b_prompt | debater_b_llm
        
        # Number of dialogue turns
        max_turns = 5  # You can adjust this number as needed
        
        for turn in range(max_turns):
            print(f"\nTurn {turn + 1}")

            # Debater Agent A's turn
            debater_a_input = {
                "problem_statement": problem_statement,
                "conversation_history": conversation_history
                }
            
            max_retries = 5
            retry_delay = 2

            # Debater Agent A's turn with retry mechanism
            debater_a_response = None
            
            for attempt in range(max_retries):
                try:
                    debater_a_response = debater_a_chain.invoke(debater_a_input)
                    print("Debater Agent A:", debater_a_response.content)
                    break  # Exit loop on success
                except Exception as e:
                    print(f"Debater Agent A attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for Debater Agent A.")
                        
            # If teacher_response is still None, set a default message or leave it as None.
            if debater_a_response is None:
                debater_a_content = None
            else:
                debater_a_content = debater_a_response.content

            # Update conversation history accordingly
            if debater_a_content is not None:
                conversation_history.append(HumanMessage(content=debater_a_content))
            else:
                conversation_history.append(HumanMessage(content=""))

            # Debater Agent B's turn with retry mechanism
            debater_b_response = None
            debater_b_input = {
                "problem_statement": problem_statement,
                "conversation_history": conversation_history
            }
            for attempt in range(max_retries):
                try:
                    debater_b_response = debater_b_chain.invoke(debater_b_input)
                    print("Debater Agent B:", debater_b_response.content)
                    break  # Exit loop on success
                except Exception as e:
                    print(f"Debater Agent B attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for Debater Agent B.")

            if debater_b_response is None:
               debater_b_content = None
            else:
               debater_b_content = debater_b_response.content

            # Update conversation history accordingly
            if debater_b_content is not None:
                conversation_history.append(HumanMessage(content=debater_b_content))
            else:
                conversation_history.append(HumanMessage(content=""))

            # Store both responses in the conversation dictionary
            conversation_dict['chat'][f"Turn {turn + 1}"] = {
                'Debater Agent A': debater_a_content,
                'Debater Agent B': debater_b_content
            }
        
        print("conversation_history_dict is ", conversation_dict)
        
        llm_answer=self.extract_final_answer_with_llm(conversation_dict)
        
        return conversation_dict, llm_answer.strip()

    def run(self):
        
        mode=self.parameters.get('mode')
        role_llm_str = self.parameters.get('role_1_llm')
        
        for iteration in range(3):
            for idx, row in enumerate(self.data, start=1):
                print("current row is ", row)
                
                problem_id = row['id']
                problem_text = row['problem']
                
                output_dir = os.path.join(os.getcwd(), 'results', 'outputs', f'{mode}_{role_llm_str}_{self.topic}', f'iter_{iteration}')
                os.makedirs(output_dir, exist_ok=True) 
                output_file_path = os.path.join(output_dir, f"problem_{problem_id}.json")
                
                try:
                    # Solve the problem using the provided method
                    llm_chat, llm_answer= self.solve_math_problem(problem_text)
                    
                    row['llm_chat']=llm_chat
                    row['llm_answer']=llm_answer
                    
                except Exception as e:
                    # Handle any exceptions that occur during problem-solving
                    print(f"An error occurred while solving problem ID {problem_id}: {e}")
                    row['llm_chat']=None
                    row['llm_answer']=None
            
                # Save final dataframe to JSON
                with open(output_file_path, 'w') as outfile:
                    json.dump(row, outfile, indent=2)