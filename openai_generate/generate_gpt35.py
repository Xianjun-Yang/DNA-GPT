import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import os
import torch, openai
import random
from utils import get_openai_response, get_chatgpt_qa_response, get_gpt4_qa_response

openai_key = "" 
openai.api_key = openai_key

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="reddit_eli5.jsonl")
parser.add_argument('--model', default="chatgpt") # "gpt-3.5-turbo", "gpt-4-0314",
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--total_instances', default=200, type=int)
parser.add_argument('--temperature', default= 0.5, type=float)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n") ]
len(data)

output_file = f"results/{args.model}_" + str(args.total_instances) +"_len.jsonl"

random.seed(43)
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:args.total_instances]
print("Total instances: {}".format(len(data)))

if args.model == "davinci_003":
    openai_fn = get_openai_response
elif args.model == "chatgpt":
    openai_fn = get_chatgpt_qa_response
elif args.model == "gpt4":
    openai_fn = get_gpt4_qa_response
else:
    raise NotImplementedError

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    question = 'Answer the following question in 180-300 words: ' + dd['question'] 
    long_answer = dd['human_answer']

    gen_text = openai_fn(prompt_text=question, temperature=args.temperature, max_tokens=args.max_new_tokens)
    outputs.append(json.dumps({
        "question": question,
        "gold_completion": long_answer,
        "gen_completion": gen_text
    }))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []
