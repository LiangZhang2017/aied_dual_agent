import os
from dotenv import load_dotenv
import time
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
import json
from langchain_deepseek import ChatDeepSeek

env_path = os.path.join(os.getcwd(), '.env')        
load_dotenv(dotenv_path=env_path, override=True)

AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION')
AZURE_OPENAI_MODEL=os.getenv('AZURE_OPENAI_MODEL')

DEEPEEK_API_KEY = os.getenv('DEEPEEK_API_KEY')
DEEPEEK_API_BASE = os.getenv('DEEPEEK_API_BASE')
        
class dual_agent_reciprocal:
    def __init__(self, parameters):
        self.parameters = parameters
        
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
                        "You are provided with a teacher-student conversation focused on solving a math problem. "
                        "Your task is to identify and extract the final answer from the conversation. "
                        "Even if multiple answers are discussed during the process, extract only the final answer based on their concluding decisions. "
                        "Please provide only the final answer without any additional text.\n\n"
                        f"Conversation:\n{conversation}\n\n"
                        "Final answer:"
                    )

        # Initialize the OpenAI Chat LLM with the API key
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
        
                # Initialize the LLMs for both agents based on the chosen provider
        if self.peer_a_llm_model == "xdeepseekv3":
            agent_a_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            agent_a_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )

        if self.peer_b_llm_model == "xdeepseekv3":
            agent_b_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            agent_b_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5,
            )
        
       # Define system prompts for Peer Agent A and Peer Agent B

        # Teacher Agent System Prompt
        teacher_system_prompt = HumanMessagePromptTemplate.from_template(
        """
        You are **Teacher Agent**, an articulate and patient educator with a strong understanding of mathematics.
        Your role is to explain mathematical problems, provide guidance, and ask questions to help your peer understand and solve the problem:\n\n
        {problem_statement}\n\n
        
        Here is coversation history:
        {conversation_history}
            
        **You work as the teacher role by asking and prompting how the student to approach the problem.**
        **Do not provide the full solution immediately.**
        Encourage your student to think critically and explain their reasoning.
        Provide hints and constructive feedback as needed to guide them toward the correct solution.
        Ensure your explanations are clear and tailored to your student's progress of understanding.
        
        **When your student reaches the correct solution, acknowledge their success and offer a summary of the key points discussed.**
        After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded.
        
        Last Notes: 
        - Use only the English language.
        - Do not display role names; only show the generated content. 
        - Follow the provided guidance and generate content step by step.
        - Do not make assumptions or include future steps in the output.
        """
        )

        # Student Agent System Prompt
        student_system_prompt = HumanMessagePromptTemplate.from_template(
            """
            You are **Student Agent**, a curious and motivated learner with a foundational understanding of mathematics.
            Your role is to engage with the Teacher Agent to understand and solve the given math problem by asking questions, applying concepts, and seeking clarification:\n\n
            {problem_statement}\n\n
            
            Here is coversation history:
            {conversation_history}
                    
            **You work as the student role by following the teacher's guidance and try to approach the problem.**
            If the problem hasn't been provided, kindly ask for it. 
            Once you have the problem, attempt to solve it step by step, explaining your reasoning at each stage.
            **Focus on answering the Teacher Agent's latest question or following latest instruction**, and **avoid repeating their words.**
            If you're unsure about something, ask for guidance.
            Stay on topic and aim to progress toward the solution in each response.
            When you believe you've solved the problem, present your final answer and thank the Teacher Agent for their assistance.
            
            **After the Teacher Agent acknowledges your solution, conclude the conversation politely in your own words without repeating their last message.**
            End with a simple farewell, such as 'Thank you for your help!' or 'Goodbye, and have a great day!
            After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded. 
        
            Last Notes: 
            - Use only the English language.
            - Do not display role names; only show the generated content. 
            - Follow the provided guidance and generate content step by step.
            - Do not make assumptions or include future steps in the output. 
        """
        )
        
        # Create the peers' prompt templates with MessagesPlaceholder
        self.teacher_prompt = ChatPromptTemplate.from_messages([
            teacher_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])

        self.student_prompt = ChatPromptTemplate.from_messages([
            student_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        
        # Initialize conversation history
        conversation_history = []
        
        # Initialize a dictionary to store the conversation history for JSON output
        conversation_dict = {'chat': {}}
        
        # callback_handler = DialogueMonitorCallback()
            
        # Create the prompt chains
        teacher_chain = self.teacher_prompt | agent_a_llm
        student_chain = self.student_prompt | agent_b_llm
        
        # Number of dialogue turns
        max_turns = 5  # You can adjust this number as needed
        
        # Flag to alternate roles
        is_teacher_turn = True  # Start with Teacher Agent
        
        for turn in range(max_turns):
            print(f"\nTurn {turn + 1}")

            if turn % 2 == 0:
                # Even turns: Agent A as Teacher, Agent B as Student
                teacher_agent = 'Agent A'
                student_agent = 'Agent B'
                current_teacher_chain = self.teacher_prompt | agent_a_llm
                current_student_chain = self.student_prompt | agent_b_llm
            else: 
                # Odd turns: Agent B as Teacher, Agent A as Student
                teacher_agent = 'Agent B'
                student_agent = 'Agent A'
                current_teacher_chain = self.teacher_prompt | agent_b_llm
                current_student_chain = self.student_prompt | agent_a_llm

            # Teacher Agent's response
            teacher_input = {
                "problem_statement":problem_statement,
                "conversation_history": conversation_history}
            
            max_retries = 5
            retry_delay = 2

            # Teacher Agent's turn with retry mechanism
            teacher_response = None
            for attempt in range(max_retries):
                try:
                    teacher_response = current_teacher_chain.invoke(teacher_input)
                    print(f"{teacher_agent} (Teacher): {teacher_response.content}")
                    break  # Exit loop on success
                except Exception as e:
                    print(f"{teacher_agent} attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for teacher.")

            # Update conversation history for teacher
            teacher_content = teacher_response.content if teacher_response is not None else None
            conversation_history.append(HumanMessage(content=teacher_content))

            # Student Agent's turn with retry mechanism
            student_response = None
            student_input = {
                "problem_statement": problem_statement,
                "conversation_history": conversation_history
            }

            for attempt in range(max_retries):
                try:
                    student_response = current_student_chain.invoke(student_input)
                    print(f"{student_agent} (Student): {student_response.content}")
                    break  # Exit loop on success
                except Exception as e:
                    print(f"{student_agent} attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached for student.")

            # Update conversation history for student
            student_content = student_response.content if student_response is not None else None
            conversation_history.append(HumanMessage(content=student_content))

            # Store responses in the conversation dictionary
            conversation_dict['chat'][str(turn + 1)] = {
                f"{teacher_agent} (Teacher)": teacher_content,
                f"{student_agent} (Student)": student_content
            }
        
        print("conversation_history_dict is ", conversation_dict)
        
        llm_answer=self.extract_final_answer_with_llm(conversation_dict)
        
        return conversation_dict, llm_answer.strip()


    def run(self):        
        mode=self.parameters.get('mode')
        role_llm_str = self.parameters.get('role_1_llm')
        
        for iteration in range(1):
         # Iterate through each row in the DataFrame
            for idx, row in enumerate(self.data, start=1):
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
            
                with open(output_file_path, 'w') as outfile:
                        json.dump(row, outfile, indent=2)