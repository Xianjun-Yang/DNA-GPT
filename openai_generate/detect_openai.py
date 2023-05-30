import json, openai
import argparse
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
import json
import os
import torch, openai
import random

openai_key = "" 
openai.api_key = openai_key

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="nature_paper.json")
parser.add_argument('--model', default="gpt35") # "gpt-3.5-turbo", "gpt-4-0314",
parser.add_argument('--min_characters', default=1000, type=int)
parser.add_argument('--total_instances', default=200, type=int)
args = parser.parse_args()

output_file = f"results/openai_detect_{args.model}.jsonl"

print("Loading model {}".format(args.model))
with open( f'results/chatgpt_200_len.jsonl', 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
len(data)
random.seed(43)

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
data = data[num_curr_outputs:args.total_instances]
print("Total instances: {}".format(len(data)))

def openai_detect(prompt):
    response = openai.Completion.create(engine="model-detect-v2",
                                        prompt=prompt,
                                        max_tokens=1,
                                        temperature=1,
                                        top_p=1,
                                        n=1,
                                        logprobs=5,
                                        stop="\n",
                                        stream=False)
    top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

    if "\"" in top_logprobs:
        quote_logprob = np.exp(top_logprobs["\""])
    elif "!" in top_logprobs:
        quote_logprob = 1.0 - np.exp(top_logprobs["!"])
    else:
        print("No quote or exclamation mark found in top logprobs")
        quote_logprob = 0.5
    return quote_logprob, response

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total= len(data) ): # 
    gen_completion = dd['gen_completion']#['choices'][0]['message']['content']
    gold_gen = dd['gold_completion']
    if len(gen_completion) <= args.min_characters or len( gold_gen ) <= args.min_characters:
        continue

    prompt_gen = gen_completion + "<|disc_score|>"
    prompt_gold = gold_gen + "<|disc_score|>"

    quote_logprob_gen, response_gen = openai_detect(prompt_gen)
    quote_logprob_gold, response_gold = openai_detect(prompt_gold)

    outputs.append(json.dumps({
        "question": dd['question'],
        "gold_gen_prob": quote_logprob_gold,
        "gold_response": response_gold,
        "gen_completion_prob": quote_logprob_gen,
        "openai_response": response_gen,
    }))

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []

with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
    outputs = []