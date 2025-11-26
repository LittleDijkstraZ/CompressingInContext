import os
import subprocess

# Fixed budget, varying HF_MAX_NEW_TOKENS
fixed_budget = 384
max_new_tokens_list = [32, 64, 256, 512, 1024]

print(f"Running with fixed budget: {fixed_budget}")
for max_new_tokens in max_new_tokens_list:
    print(f"  Running with HF_MAX_NEW_TOKENS: {max_new_tokens}")
    env = os.environ.copy()
    env['HF_MAX_NEW_TOKENS'] = str(max_new_tokens)
    env['budget'] = str(fixed_budget)
    subprocess.run(['python', './precompute_cache.py'], env=env)

# # Fixed HF_MAX_NEW_TOKENS, varying budget
# fixed_max_new_tokens = 256
# # Using a large number (100000) for "None" to effectively disable compression
# budget_list = [72, 128, 384, 768, 100000] 

# print(f"Running with fixed HF_MAX_NEW_TOKENS: {fixed_max_new_tokens}")
# for budget in budget_list:
#     print(f"  Running with budget: {budget}")
#     env = os.environ.copy()
#     env['HF_MAX_NEW_TOKENS'] = str(fixed_max_new_tokens)
#     env['budget'] = str(budget)
#     subprocess.run(['python', './precompute_cache.py'], env=env)

print("All precomputation runs are complete.")
