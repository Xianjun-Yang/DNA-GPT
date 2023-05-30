import argparse
import tqdm
import json
import os
import openai
import random

openai_key = "" 
openai.api_key = openai_key

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="results/gpt4_200_len.jsonl")
parser.add_argument('--model', default= "gpt-4-0314") # gpt-3.5-turbo, gpt-4-0314
parser.add_argument('--max_new_tokens', default=300, type=int)
parser.add_argument('--regen_number', default=10, type=int)
parser.add_argument('--temperature', default=0.7, type=float)
parser.add_argument('--truncate_ratio', default=0.7, type=float)
args = parser.parse_args()

print('load model ...', args.model)
with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

random.seed(43)

output_file = f"results/regen_{args.model}_10_" + str(args.truncate_ratio) + ".jsonl"

if os.path.exists(output_file):
    with open(output_file, "r") as f:
        num_curr_outputs = len(f.read().strip().split("\n"))
else:
    num_curr_outputs = 0

print("Skipping {} instances".format(num_curr_outputs))
print('len(data): ', len(data))
data = data[num_curr_outputs:]

outputs = []
for idx, dd in tqdm.tqdm(enumerate(data), total = len(data) ): # len(data)
    gold_completion = dd['gold_completion']
    human_prefix_prompt = gold_completion[ :int( args.truncate_ratio*len(gold_completion) ) ]

    gen_completion = dd['gen_completion']
    machine_prefix_prompt = gen_completion[ :int( args.truncate_ratio*len(gen_completion) ) ]
        
    human_gen_text = openai.ChatCompletion.create(model= args.model,
                messages=[{"role": "system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                            {"role": "user", "content": dd['question']},
                            {"role": "assistant", "content": human_prefix_prompt},
                        ], 
                        temperature= args.temperature,
                        max_tokens = args.max_new_tokens,
                        n= args.regen_number)
    
    machine_gen_text = openai.ChatCompletion.create(model= args.model,
                messages=[{"role": "system", "content": "You are a helpful assistant that continues the passage from the sentences provided."},
                            {"role": "user", "content": dd['question']},
                            {"role": "assistant", "content": machine_prefix_prompt},
                        ], 
                        temperature= args.temperature,
                        max_tokens = args.max_new_tokens,
                        n= args.regen_number)
    
    outputs.append( json.dumps( { 'question': dd['question'],
                                 'human_gen_truncate': gold_completion[ int( args.truncate_ratio*len(gold_completion)): ],
                                 'machine_gen_truncate': gen_completion[ int( args.truncate_ratio*len(gen_completion) ): ], 
                                 "human_gen_text": human_gen_text, 
                                  "machine_gen_text": machine_gen_text }) )

    with open(output_file, "a") as f:
        f.write("\n".join(outputs) + "\n")
    outputs = []
    
with open(output_file, "a") as f:
    f.write("\n".join(outputs) + "\n")
outputs = []