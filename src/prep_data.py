from datasets import load_dataset

ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split='train')

"""
Plan:

1. Filter dataset to to only include rows where is_response_0_safe or is_response_1_safe is True.
DPO:
1. Concatenate prompt and response_{safer_response_id} into chosen.
2. Concatenate prompt and response_{1-safer_response_id} into rejected.
SFT:
1. Set prompt to input
2. Set response_{safer_response_id} to output.
"""

ds = ds.filter(lambda x: x['is_response_0_safe'] or x["is_response_1_safe"])

ds = ds.map(lambda x: {'safer_response': x['response_' + str(x['safer_response_id'])]})

ds = ds.map(lambda x: {'chosen': 'User: ' + x['prompt'] + ' Assistant: ' + x['safer_response']})
ds = ds.map(lambda x: {'rejected': 'User: ' + x['prompt'] + ' Assistant: ' + x['response_' + str(1 - x['safer_response_id'])]})
ds = ds.map(lambda x: {'input': 'User: ' + x['prompt'] + ' Assistant: '})
ds = ds.map(lambda x: {'output': x['safer_response']})

dpo = ds.select_columns(['chosen', 'rejected'])
sft = ds.select_columns(['input', 'output'])
print(dpo[0:5])
print(sft[0:5])

dpo.push_to_hub("excepto64/PKU-SafeRLHF-filtered-dpo")
sft.push_to_hub("excepto64/PKU-SafeRLHF-filtered-sft")