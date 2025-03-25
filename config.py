import os
import json
import sys
sys.path.append('c:/Users/Liang Zhang/VisualStudioCode/workspace/LAK/dual-agent-math')
from helper import read_jsonl
import pandas as pd
from modes.single_agent import single_agent
from modes.single_agent_o1 import single_agent_o1
from modes.dual_agent_ts import dual_agent_teacher_student
from modes.dual_agent_debate import dual_agent_debate
from modes.dual_agent_pp import dual_agent_peer2peer
from modes.dual_agent_rpt import dual_agent_reciprocal

class mode_config:
    def __init__(self,args):
        self.args=args
    
    def generate_paradic(self):
        print(f"Generating paradigm with args: {self.args}")
        
        dataset_path = os.path.join(os.getcwd(), self.args.data_path[0], self.args.data_path[1])
        
        print("dataset_path is ", dataset_path)
        
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        para={'dataset':data,
              'topic':self.args.topic[0],
              'mode':self.args.mode[0],
              'role_1_llm':self.args.llm[0],
              'role_2_llm':self.args.llm[1]
        }
        
        return para
        
    def main(self, parameters):
        print("start the multiagent modes running")
        
        mode=parameters['mode']
        llm_model=str(parameters['role_1_llm'])
        
        print("llm_model is ", llm_model)
        
        if mode=='single_agent' and llm_model != "o1-preview":
            print(f"single agent {llm_model}")
            agent_obj=single_agent(parameters)
            agent_obj.run()
        
        if mode=='single_agent' and llm_model == "o1-preview":
            print(f"single agent {llm_model}")
            agent_obj=single_agent_o1(parameters)
            agent_obj.run()
        
        if mode=='dual_agent_ts':
            print("dual agent: teacher-student mode")
            agent_obj=dual_agent_teacher_student(parameters)
            agent_obj.run()
        
        if mode=='dual_agent_debate':
            print("dual agent: critical debate mode")
            agent_obj=dual_agent_debate(parameters)
            agent_obj.run()
        
        if mode=="dual_agent_pp":
            print("dual agent: peer-peer mode")
            agent_obj=dual_agent_peer2peer(parameters)
            agent_obj.run()
        
        if mode=="dual_agent_rcp":
            print("dual agent: Reciprocal Peer Teaching mode")
            agent_obj=dual_agent_reciprocal(parameters)
            agent_obj.run()