# Env Setups

First, create a conda environment. 
```sh
conda env create -f env.yml
conda activate flashattn
```

Then clone the flash attention library and compile it from source
```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

Now you can run any tests you want.
HuggingFace
```
CUDA_VISIBLE_DEVICES=0 python flashattn-gen.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python flashattn-gen-mps.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python flashattn-peft-gen.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
CUDA_VISIBLE_DEVICES=0 python flashattn-peft.py -m meta-llama/Meta-Llama-3-8B --batch_size 1
```

If you want to setup timmer for the attention layer specifically, you will have to modify a file inside of transformers library (modeling_llama.py). To find the location of the file, we can use Python
```py
import inspect
import transformers
print(inspect.getfile(transformers.models.llama.modeling_llama))
```
Now you can delete the file and link the `modeling_llama.py` in this directory (below is just an example).
```sh
rm /pub/scratch/zijzhang/conda/envs/flashattn/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
ln -s /home/zijzhang/FineInfer-kernels/benchmarks/flashattn-cuda/modeling_llama.py /pub/scratch/zijzhang/conda/envs/flashattn/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
```