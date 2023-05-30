import argparse
import tqdm
import json
import os
import torch
import random
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import torch

tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

device = torch.device("cuda:1")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().to(device) #.half() to use FP16

def gptx_generate(prompt, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    gen_tokens = model.generate(
        input_ids = inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        do_sample=True,
        temperature=0.7,
        max_length=max_length,
        num_return_sequences = 1,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text[len(prompt):]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="reddit_eli5.jsonl")
parser.add_argument('--model', default="gptx")
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--total_instances', default= 200, type=int)
args = parser.parse_args()

with open( args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n") ]
len(data)

output_file = f"results/{args.model}_200_len.jsonl"

random.seed(43)
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:]
print("Total instances: {}".format(len(data)))


outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    question = dd['question'] 
    prefix = "Answer the following question in around 200 words: " #dd['prefix']
    gold_completion = dd['human_answer']

    prompt_input = question + prefix

    gen_text = gptx_generate(prompt_input, max_length=args.max_new_tokens)

    outputs.append(json.dumps({
        "question": question,
        "prefix": prefix,
        "gold_completion": gold_completion,
        "gen_completion": gen_text
    }))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []
