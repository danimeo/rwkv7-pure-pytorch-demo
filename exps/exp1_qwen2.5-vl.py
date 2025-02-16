import os
# pip install git+https://github.com/huggingface/transformers accelerate
# from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import torch

torch.backends.cuda.enable_mem_efficient_sdp(True)
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# model_dir=snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
model_dir = "/data/_models/Qwen/Qwen2.5-VL-7B-Instruct"
# model_dir = "/data/_models/Qwen/Qwen2-VL-2B-Instruct"

# try:
#     os.system('sudo echo 1 > /proc/sys/vm/drop_caches')
# except Exception as e:
#     print(e)
# gc.collect()
# torch.cuda.empty_cache()
print('torch.cuda.memory_usage:', torch.cuda.memory_usage())
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype='auto', device_map='cuda'
)
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     model_dir,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
print('torch.cuda.memory_usage:', torch.cuda.memory_usage())
print(model.dtype)
# torch.cuda.empty_cache()
print(torch.cuda.is_available())
# torch.cuda.ipc_collect()
print(torch.cuda.is_bf16_supported())

# default processer
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=64*28*28, max_pixels=256*28*28)
print(torch.cuda.memory_summary())
print('torch.cuda.memory_usage:', torch.cuda.memory_usage())

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "图中的人在做什么？"},
        ],
    }
]

print(torch.cuda.memory_summary())
print('torch.cuda.memory_usage:', torch.cuda.memory_usage())

with torch.no_grad():
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # with torch.autocast("cuda", torch.bfloat16):
    
    for i in range(10):
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)





