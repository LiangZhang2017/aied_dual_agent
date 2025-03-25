
import sys
sys.path.append('c:/Users/Liang Zhang/VisualStudioCode/workspace/LAK/dual-agent-math')
from config import mode_config
import argparse

'''

llm_model: 
gpt-3.5-turbo
gpt-4o
o1-preview
xdeepseekv3

Mode: 
single_agent
dual_agent_ts
dual_agent_debate
dual_agent_pp
dual_agent_rcp
'''

"""
MATH Dataset Topics:
1. algebra
2. counting_and_probability
3. geometry
4. intermediate_algebra
5. number_theory
6. prealgebra
7. precalculus
"""

if __name__ == '__main__':
    print("Start")
    
    parser=argparse.ArgumentParser(description='Dual Agent Setting')
    parser.add_argument("--data_path",nargs=2,type=str,default=['MATHDATASET/math_100/precalculus','precalculus_level_5_random_100_with_answers.json'])
    parser.add_argument('--mode',nargs=1,type=str,default=['dual_agent_rcp'])
    parser.add_argument('--topic',nargs=1,type=str,default=['precalculus'])
    parser.add_argument('--llm',nargs=1,type=str,default=['xdeepseekv3','xdeepseekv3']) # role1 and role2
    args = parser.parse_args()
    
    config_obj = mode_config(args)
    parameters=config_obj.generate_paradic()
    config_obj.main(parameters)