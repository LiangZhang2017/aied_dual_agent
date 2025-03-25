import os
from dotenv import load_dotenv
import time
import json

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from azure.core.exceptions import AzureError
from langchain_deepseek import ChatDeepSeek

env_path = os.path.join(os.getcwd(), '.env')        
load_dotenv(dotenv_path=env_path, override=True)

AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_VERSION=os.getenv('AZURE_OPENAI_VERSION')
AZURE_OPENAI_MODEL=os.getenv('AZURE_OPENAI_MODEL')

DEEPEEK_API_KEY = os.getenv('DEEPEEK_API_KEY')
DEEPEEK_API_BASE = os.getenv('DEEPEEK_API_BASE')

print("DEEPEEK_API_KEY is ", DEEPEEK_API_KEY)
print("DEEPEEK_API_BASE is ", DEEPEEK_API_BASE)

class dual_agent_teacher_student:
    def __init__(self, parameters):
        self.parameters = parameters
        
        # Get the model to use (default to 'gpt-4' if not specified)
        self.role_1_llm_model = self.parameters.get('role_1_llm')
        self.role_2_llm_model = self.parameters.get('role_2_llm')
        
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

        # Initialize teacher LLM based on the provided parameter
        if self.role_1_llm_model == "xdeepseekv3":
            teacher_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            teacher_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )
        
        # Initialize student LLM based on the provided parameter
        if self.role_2_llm_model == "xdeepseekv3":
            student_llm = ChatDeepSeek(
                model="xdeepseekv3",
                temperature=0.5,
                api_key=DEEPEEK_API_KEY,
                api_base=DEEPEEK_API_BASE
            )
        else:
            student_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_VERSION,
                azure_deployment=AZURE_OPENAI_MODEL,
                temperature=0.5
            )
        
        # # Define system prompts for teacher and student
        
        teacher_system_prompt = HumanMessagePromptTemplate.from_template(
                    """You are a supportive math teacher engaging in a dialogue to assist a student in solving the following math problem:
                    {problem_statement}
                    
                    Here is coversation history:
                    {conversation_history}
                    
                    Your objectives are to guide the student through the problem-solving process by asking insightful questions, providing helpful hints, and offering constructive feedback. 
                    You should Engage in Collaborative Dialogue in Math Problem Solving. Engage in a multi-turn dialogue to collaborate with the student as working through the math problem.
                    
                    Please adhere to the following guidelines:

                    1. **Please follow all the following points to initiate the dialogue:**:
                    - Greet the student warmly and introduce yourself as the math teacher who can assist the student in solving the math problem.
                    - At the begining, present the exact math problem to the student clearly. Provide a brief summary of the problem to ensure understanding. 
                    - **Provide Exact Guidance:** 
                        - **Offer strategic hints or clues that promote independent thinking on solving this math problem.**
                        - **Emphasize core concepts and principles needed to approach the problem.**
                        - ** Offer clear, actionable advice tailored to the particular math problem, guiding the student through the problem-solving process without revealing the complete solution.**
                        
                    2. **Assess the Student's Performance and Provide Tailored Guidance**:
                    - If the student indicates they have solved the problem:
                        - Review the student's solution and final answer to determine correctness. Avoid confirming a student's solution without thorough evaluation. If uncertain about its correctness, encourage the student to reflect on their problem-solving process and provide critical questions to assess their understanding.  
                        - If you think it's correct, confirm the solution's correctness and prompt the student to reflect their achievement.
                        - If you think it's incorrect, examine the student's approach to pinpoint misunderstandings or errors. Provide guidance that encourages the student to think critically and reflect on their reasoning without directly providing the solution. Prompt the student to articulate their thought process to foster deeper understanding and self-correction.   
                    - If the student is still working on the problem:
                        - **Offer Supportive Hints**:
                            - Evaluate the student's current approach to understand their thought process, identify any misconceptions and present supportive guidance. 
                            - Offer clear and concise suggestions to help the student overcome specific obstacles in their problem-solving process.
                            - Focus on high-level hints that encourage independent thinking.
                            - Highlight key concepts and necessary steps without revealing the full solution.

                    3. **Maintain a Natural and Focused Dialogue**:
                    - Ensure the conversation flows naturally.
                    - Avoid deviating from the topic or restarting the discussion unnecessarily.
                    - Build upon the student's latest responses without repeating their statements.
                    - **Do not engage in role-flipping; remain in the role of a student throughout the dialogue.** 
                    - **Do not repeat your own previous words; only move the dialogue forward.**
                    - Properly response to the student's latest output. 

                    4. **Conclude Appropriately**:
                    - When the student reaches the correct solution, acknowledge their success.
                    - Conclude the conversation in a supportive and encouraging manner.

                    **Important Notes**:
                    - **Provide guidance to the student without directly presenting detailed solutions and the final answer.**:
                        - Refrain from providing step-by-step solutions or revealing all necessary steps.
                        - Guide the student to focus on the critical parts of the problem-solving method.
                        
                    - **Do Not Repeat Student Responses**:
                        - In each response, build upon the student's latest answer without repetition.
                        
                    - After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded.
                    
                    Last Notes:
                    - Use only the English language.
                    - Do not display role names; only show the generated content. 
                    - Provide guidance only, **do not present a complete solution and final answer**. 
                    - Follow the provided guidance and generate content step by step.
                    - Do not make assumptions or include future steps in the output.  
                    """
            )

        student_system_prompt =  HumanMessagePromptTemplate.from_template(
                    """
                    **You are a student working on solving a math problem. When you first receive a message from the teacher, politely introduce yourself as a student and express your willingness to work on the problem.**  
                    - If the teacher has not provided the exact math problem, kindly ask for it.  
                    - Once you receive the problem, attempt to solve it step by step, explaining your reasoning at each stage.  
                    - If you're unsure how to proceed, ask the teacher for guidance. 
                    - Engage in Collaborative Dialogue in Math Problem Solving. Engage in a multi-turn dialogue to collaborate with the teacher as working through the math problem.
                    
                    Here is coversation history:
                    {conversation_history}

                    1. Respond to the Tutor's Latest Instruction:
                    - Follow the tutor's guidance to continue solving the problem.  
                    - If the tutor suggests adjustments, modify your steps accordingly.  

                    2. Explain Your Reasoning:
                    - Clearly articulate each step of your problem-solving process.  
                    - Provide detailed explanations to demonstrate your understanding.  

                    3. Engage in the Dialogue:
                    - Ask questions if any instructions or concepts are unclear.  
                    - Seek clarification to ensure you are on the right track.  

                    4. Present and Refine Your Answer:
                    - **When you believe you have solved the problem, present your final answer clearly and ask for confirmation.**  
                    - **If the teacher points out an error, follow their instructions to correct it.** 
                    - Feel free to seek help at any point during the process.  

                    5. Maintain Tone and Flow:
                    - Keep the conversation smooth, logical, and conversational.  
                    - Focus on collaborating with the teacher to reach the correct solution.  
                    - **Do not engage in role-flipping; remain in the role of a student throughout the dialogue.**
                    - Build upon the teacher's latest responses without repeating their statements. 
                    - **Do not repeat your own previous words; only move the dialogue forward.**
                    -  Properly response to the teacher's latest feedback. 

                    **When the teacher indicates that the session is over, conclude the conversation politely in your own words.**  
                    - **Do not repeat or paraphrase the teacher’s final message**.  
                    - End with a simple and original farewell, such as "Thank you for your help!" or "Goodbye, and have a great day!" 
                    - After concluding, refrain from initiating a new dialogue; if prompted, acknowledge that the session has already concluded.   
                    
                    Last Notes: 
                    - Use only the English language.
                    - Do not display role names; only show the generated content. 
                    - Follow the provided guidance and generate content step by step.
                    - Do not make assumptions or include future steps in the output.  
                    """
                    )
        
        # Create the teacher's prompt template with MessagesPlaceholder
        teacher_prompt = ChatPromptTemplate.from_messages([
            teacher_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])

        # Create the student's prompt template with MessagesPlaceholder
        student_prompt = ChatPromptTemplate.from_messages([
            student_system_prompt,
            MessagesPlaceholder(variable_name="conversation_history")
        ])
         
        # Initialize conversation history
        conversation_history = []
        
        # Initialize a dictionary to store the conversation history for JSON output
        conversation_dict = {'chat': {}}
        
        # callback_handler = DialogueMonitorCallback()

        # Create the teacher's chain
        teacher_chain = teacher_prompt | teacher_llm
        
        # Create the student's chain
        student_chain = student_prompt | student_llm
        
        # Number of dialogue turns
        num_turns = 5  # You can adjust this number as needed
        
        for turn in range(num_turns):
            print(f"\nTurn {turn+1}")
            
            # Teacher's turn
            teacher_input = {
                "problem_statement":problem_statement,
                "conversation_history": conversation_history
                }
            
            max_retries=5
            retry_delay=2
            
            # Teacher's turn
            teacher_response = None  # Initialize to None
            
            for attempt in range(max_retries):
                try:
                    teacher_response = teacher_chain.invoke(teacher_input)
                    print("Teacher:", teacher_response.content)
                    break
                
                except Exception as e:  # or `Exception` if you prefer a broader catch
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Unable to get a valid response.")
                        
            
            # If teacher_response is still None, set a default message or leave it as None.
            if teacher_response is None:
                teacher_content = None
            else:
                teacher_content = teacher_response.content
                
            # Update conversation history accordingly
            if teacher_content is not None:
                conversation_history.append(HumanMessage(content=teacher_content))
            else:
                conversation_history.append(HumanMessage(content=""))
            
            # Student's turn
            student_response = None  # Initialize to None
            student_input = {
                "conversation_history": conversation_history
                }
            
            for attempt in range(max_retries):
                try:
                    # Force an exception for testing:
                    student_response = student_chain.invoke(student_input)
                    print("Student:", student_response.content)
                    break
                
                except Exception as e:  # or `Exception` if you prefer a broader catch
                    print(f"Attempt {attempt + 1} failed with error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Unable to get a valid response.")   
            
            if student_response is None:
                student_content = None
            else:
                student_content = student_response.content

            # Update conversation history accordingly
            if student_content is not None:
                conversation_history.append(HumanMessage(content=student_content))
            else:
                conversation_history.append(HumanMessage(content=""))
            
            # Store the teacher's and student's responses in the conversation_dict
            conversation_dict['chat'][str(turn+1)] = {
                'teacher': teacher_content,
                'student': student_content
            }
        
        # # Output the final solution
        # print("\nFinal Student Solution:\n", student_response.content)
        
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
                    # Assign the llm chayt and final answer column for the current row
                    
                    print("llm_chat is ", llm_chat)
                    
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