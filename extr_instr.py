import json
import os 

dev = 'dev'

with open(os.path.join('./', dev, 'edit_turns.json')) as f:
    edit_list = json.load(f)
    f.close()

instr_list = []
for item in edit_list:
    instr = item["instruction"]
    instr_list.append(instr)

with open(os.path.join('./', 'instructions.json'), 'w') as f:
    edit_list = json.dump(instr_list, f)
    f.close()