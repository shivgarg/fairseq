import json
import torch
from fairseq.models.roberta import RobertaModel
#from examples.roberta import commonsense_qa  # load the Commonsense QA task
import commonsense_qa_task
import argparse

args = argparse.ArgumentParser()
args.add_argument("ckpt_dir",type=str)
args.add_argument("json_file",type=str)
args.add_argument("out_file",type=str)
args = args.parse_args()

out = open(args.out_file,'w')
roberta = RobertaModel.from_pretrained(args.ckpt_dir, 'checkpoint_best.pt', 'data/CommonsenseQA')
roberta.eval()  # disable dropout
#roberta.cuda()  # use the GPU (optional)

nsamples, ncorrect = 0, 0

with open(args.json_file) as h:
    for line in h:
        example = json.loads(line)
        scores = []
        for choice in example['question']['choices']:
            if 'cose' in example['question']:
                input = roberta.encode(
                        'Q: ' + example['question']['stem'],
                        'E: ' + example['question']['cose'],
                        'A: ' + choice['text'],
                        no_separator=True
                        )
            else:
                input = roberta.encode(
                    'Q: ' + example['question']['stem'],
                    'A: ' + choice['text'],
                    no_separator=True
                )
            score = roberta.predict('sentence_classification_head', input, return_logits=True)
            scores.append(score)
        pred = torch.cat(scores).argmax()
        example['question']['ans_scores'] =list(map(lambda a: a.item(), scores))
        out.write(json.dumps(example)+'\n')
        answer = ord(example['answerKey']) - ord('A')
        nsamples += 1
        if pred == answer:
            ncorrect += 1

out.close()
print('Accuracy: ' + str(ncorrect / float(nsamples)))

# Accuracy: 0.7846027846027847
