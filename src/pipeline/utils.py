import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache

import json
import os

try:
	from typing import override, Optional
except: # compatibility with python < 3.12
	from typing_extensions import override, Optional

class PatchedDynamicCache(DynamicCache):
	def __init__(self):
		super().__init__()
		self.altered_length = None

	@classmethod
	def from_legacy_cache(cls, past_key_values: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...]) -> "DynamicCache":
		cache = cls()
		if past_key_values is not None:
			for layer_idx in range(len(past_key_values)):
				key_states, value_states = past_key_values[layer_idx]
				cache.update(key_states, value_states, layer_idx)
		return cache

	def set_seq_length(self, alt_seq_len):
		self.altered_length = alt_seq_len

	@override
	def get_seq_length(self, layer_idx: Optional[int] = 0):
		if self.altered_length is not None:
			return self.altered_length
		
		return super().get_seq_length(layer_idx)
