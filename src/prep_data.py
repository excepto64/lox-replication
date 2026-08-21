"""Build the matching SFT/DPO alignment datasets used by align.sh, from PKU-SafeRLHF.

Filters PKU-Alignment/PKU-SafeRLHF down to rows where at least one response is
safe (so both methods only ever train towards a safe target), then derives:
- DPO: chosen = prompt + safer response, rejected = prompt + the other response.
- SFT: input = prompt, output = safer response.
Pushes both to the Hub as excepto64/PKU-SafeRLHF-filtered-{dpo,sft}, the
datasets align.sh pulls by default. Not needed to reproduce the experiment --
see the README's "Recreating the datasets" section.
"""

from datasets import load_dataset

ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split='train')

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