import json
import argparse

args = argparse.ArgumentParser()
args.add_argument("questions",type=str)
args.add_argument("exp",type=str)
args.add_argument("output",type=str)
args = args.parse_args()

ques = open(args.questions)
exp = open(args.exp)
out = open(args.output,'w')

exp_map = dict()

for lines in exp:
    line = json.loads(lines)
    exp_map[line['id']] = line

ques_map = dict()

for lines in ques:
    line = json.loads(lines)
    line['question']['cose'] = exp_map[line['id']]["explanation"]["open-ended"]
    ques_map[line['id']] = line

for key in ques_map:
    st = json.dumps(ques_map[key])
    out.write(st+'\n')

out.close()


