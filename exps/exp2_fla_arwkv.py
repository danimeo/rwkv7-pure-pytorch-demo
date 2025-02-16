import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained(
    "RWKV-Red-Team/ARWKV-7B-Preview-0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "RWKV-Red-Team/ARWKV-7B-Preview-0.1"
)
