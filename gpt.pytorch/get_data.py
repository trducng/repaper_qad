from datasets import load_dataset
from transformers import OpenAIGPTModel, OpenAIGPTConfig, GPT2LMHeadModel

dataset = load_dataset("bookcorpus")
