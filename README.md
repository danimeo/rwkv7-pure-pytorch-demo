
**Includes errors, not ready for use**

---
Just another version of pytorch implementation of RWKV-7, currently for my personal use.
Originally from https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/

---

Additionally, here's a list of different **RWKV-7** inference implementations I know so far:


### 1. Architecture & Training

#### Original

RWKV-LM

https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7

https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py


#### Modded-NanoGPT-RWKV

https://github.com/BlinkDL/modded-nanogpt-rwkv


#### Flash-Linear-Attention (FLA)

https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/rwkv7/modeling_rwkv7.py


#### Triton impl

https://github.com/TorchRWKV/flash-linear-attention


#### RWKV Block impl

https://github.com/RWKV/RWKV-block/tree/main/rwkv_block/v7_goose



### 2. Inference

#### Inference: CPU/CUDA/ROCm, C++/Python

https://github.com/RWKV/rwkv.cpp

https://github.com/OpenMOSE/RWKV-Infer/blob/main/rwkvengine/rwkv7.py

https://github.com/00ffcc/conRWKV


#### Inference: Vulkan/WebGPU

https://github.com/cryscan/web-rwkv

https://github.com/Ai00-X/ai00_server/tree/main/crates/ai00-core


#### Inference: onnx/ncnn/mlx

?

https://github.com/thegodone/mlx-rwkv


### 3. Forks (Reconstruction or Re-impl)

#### Reconstruct for better understanding

https://github.com/l15y/read_rwkv_v7

https://github.com/Triang-jyed-driung/rwkv7mini

https://github.com/Triang-jyed-driung/my-fla/blob/main/model.py

#### Re-impl for better understanding

https://github.com/SmerkyG/RWKV_Explained/blob/main/rwkv7.py

https://github.com/Triang-jyed-driung/receptivite-cle-valeur-pesee

#### For other purpose

Aimed at optimizing
https://github.com/johanwind/wind_rwkv 

https://github.com/erogol/BlaGPT/blob/main/bla_gpt/rwkv7/model.py

https://github.com/SmerkyG/hfattnconv/tree/main/rwkv_cuda

### Inference

https://github.com/Jellyfish042/rwkv_mmlu/blob/main/rwkv_mmlu_minimal.py

https://github.com/Beortext/RWKV-ZeroCoT/blob/main/v7_seq_model.py


### Other re-impl: Multimodal RWKV-7

https://github.com/xforcevesa/new-vrwkv

