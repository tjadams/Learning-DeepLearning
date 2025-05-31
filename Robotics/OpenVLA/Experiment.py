from transformers import AutoModelForVision2Seq
import torch

model_name = "openvla/openvla-7b-finetuned-libero-object"

# Apple Silicon
device = "mps" 

# torch.bfloat16 not supported on mps
torch_dtype = torch.float16

model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

print("Model loaded successfully!")