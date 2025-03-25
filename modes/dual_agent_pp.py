import os
from dotenv import load_dotenv
import json
import time

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder,  SystemMessagePromptTemplate, HumanMessagePromptTemplate
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
        
class dual_agent_peer2peer:
    def __init__(self, parameters):
        self.parameters = parameters
        
        env_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Get the model to use (default to 'gpt-4' if not specified)
        self.peer_a_llm_model = self.parameters.get('role_1_llm')
        print("LLM model role 1 is", self.peer_a_llm_model)
        self.peer_b_llm_model = self.parameters.get('role_2_llm')
        print("LLM model role 2 is", self.peer_b_llm_model)
        
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
                        "You are provided with a peer-peer conversation focused on solving a math problem. "
                        "Your task is to identify and extract the final answer from the conversation. "
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
        
        # Initialize the LLMs for both roles with an option for DeepSeek
        if self.peer_a_llm_model == "xdeepseekv3":
            peer_a_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            peer_a_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )
            
        if self.peer_b_llm_model == "xdeepseekv3":
            peer_b_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            peer_b_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5,
            )
        
       # Define system prompts for Peer Agent A and Peer Agent B

        peer_a_system_prompt = HumanMessagePromptTemplate.from_template(
            """
            **You are Peer Agent A**, a collaborative and knowledgeable mathematician. 
            Your role is to work alongside Peer Agent B to solve the following math problem by sharing insights, proposing ideas, and building upon each other's contributions.

            **Math Problem:**
            {problem_statement}

            **Conversation History:**
            {conversation_history}

            **Guidelines:**

            1. **Start by sharing your initial thoughts and approach to solving the problem.**
            2. **Maintain a balanced discussion—encourage Peer Agent B to contribute their own methods and ideas.**
            3. Foster a cooperative environment by actively listening to Peer Agent B's input and integrating their ideas into the problem-solving process.
            4. Provide constructive feedback and suggest alternative methods when appropriate.
            5. Keep the conversation focused on collaboratively solving the problem.
            6. Evaluate Peer Agent B’s solutions and reasoning, offering your insights to support and refine the problem-solving process.

            **Important Notes:**
            - **When you believe the problem is solved, summarize the key points discussed and acknowledge Peer Agent B’s contributions.**
            - Once the discussion concludes, do not introduce new topics. If prompted to continue, politely confirm that the session has already ended.
            - **Do not engage in role-flipping; remain in the role of a Peer Agent A throughout the dialogue.**
                    - Build upon the Peer Agent B's latest responses without repeating their statements. 
                    - **Do not repeat your own previous words; only move the dialogue forward.**
                    -  Properly response to the Peer Agent B's latest feedback.
            
            Last Notes: 
            - Use only the English language.
            - Do not display role names; only show the generated content. 
            - Follow the provided guidance and generate content step by step.
            - Do not make assumptions or include future steps in the output.
            """
        )

        peer_b_system_prompt = HumanMessagePromptTemplate.from_template(
            """
            **You are Peer Agent B**, a collaborative and knowledgeable mathematician. 
            Your role is to work alongside Peer Agent A to solve the following math problem through discussion, feedback, and joint problem-solving. 
            
            **Math Problem:**
            {problem_statement}
            
            **Conversation History:**
            {conversation_history}
            
            **Guidelines:**
            
            1. **Engage with Peer Agent A by responding to their thoughts, building on their approach, and contributing your own insights.**
            2. **Maintain a balanced conversation—avoid dominating and instead encourage Peer Agent A to actively participate.**
            3. Foster a cooperative environment by actively listening, acknowledging Peer Agent A’s ideas, and integrating them into your approach.
            4. Provide constructive feedback, suggest alternative methods when relevant, and help refine the solution.
            5. Keep the discussion focused on collaboratively solving the problem.
            6. Assess Peer Agent A’s proposed solutions and reasoning, offering support and additional perspectives to enhance problem-solving.

            **Important Notes:**
            - **When you believe the problem is solved, summarize the key points discussed and acknowledge Peer Agent A’s contributions.**  
            - Once the discussion concludes, do not introduce new topics. If prompted to continue, politely confirm that the session has already ended.
            - **Do not engage in role-flipping; remain in the role of a Peer Agent B throughout the dialogue.**
                    - Build upon the Peer Agent A's latest responses without repeating their statements. 
                    - **Do not repeat your own previous words; only move the dialogue forward.**
                    -  Properly response to the Peer Agent A's latest feedback. 
            
            Last Notes: 
            - Use only the English language.
            - Do not display role names; only show the generated content. 
            - Follow the provided guidance and generate content step by step.
            - Do not make assumptions or include future steps in the output.
            """
        )
        
        # Create the peers' prompt templates with MessagesPlaceholder
        peer_a_prompt = ChatPromptTemplate.from_messages([
            peer_a_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])

        peer_b_prompt = ChatPromptTemplate.from_messages([
            peer_b_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        
        # Initialize conversation history
        conversation_history = []
        
        # Initialize a dictionary to store the conversation history for JSON output
        conversation_dict = {'chat': {}}
        
        # callback_handler = DialogueMonitorCallback()

        # Create the peers' chains
        peer_a_chain = peer_a_prompt | peer_a_llm
        peer_b_chain = peer_b_prompt | peer_b_llm
        
        # Number of dialogue turns
        max_turns = 5  # You can adjust this number as needed
        
        for turn in range(max_turns):
            print(f"\nTurn {turn + 1}")

            # Peer Agent A's turn
            peer_a_input = {
                "problem_statement": problem_statement,
                "conversation_history": conversation_history
                }
            
            max_retries = 5
            retry_delay = 2

            # Peer Agent A's turn with retry mechanism
            peer_a_response = None
            
            for attempt in range(max_retries):
                try:
                    peer_a_response = peer_a_chain.invoke(peer_a_input)
                    print("Peer Agent A:", peer_a_response.content)
                    break  # Exit loop if successful
                except Exception as e:
                    print(f"Peer Agent A attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for Peer Agent A.")

            # Update conversation history for Peer Agent A
            if peer_a_response is None:
                peer_a_content= None
            else:
                peer_a_content=peer_a_response.content
            
            if peer_a_content is not None:
                conversation_history.append(HumanMessage(content=peer_a_content))
            else:
                conversation_history.append(HumanMessage(content=""))

            # Peer Agent B's turn with retry mechanism
            peer_b_response = None
            peer_b_input = {
                "problem_statement": problem_statement,
                "conversation_history": conversation_history
            }
            for attempt in range(max_retries):
                try:
                    peer_b_response = peer_b_chain.invoke(peer_b_input)
                    print("Peer Agent B:", peer_b_response.content)
                    break  # Exit loop if successful
                except Exception as e:
                    print(f"Peer Agent B attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for Peer Agent B.")

            # Update conversation history for Peer Agent B
            
            if peer_b_response is None:
               peer_b_content = None
            else:
               peer_b_content = peer_b_response.content
               
            if peer_b_content is not None:
                conversation_history.append(HumanMessage(content=peer_b_content))
            else:
                conversation_history.append(HumanMessage(content=""))

            # Store both responses in the conversation dictionary
            conversation_dict['chat'][str(turn+1)] = {
                'Peer Agent A': peer_a_content,
                'Peer Agent B': peer_b_content
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