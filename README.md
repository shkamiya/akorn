# Artificial Kuramoto Oscillatory Neurons (AKOrN)
<p align="center">
  <img src="https://github.com/user-attachments/assets/62af0387-d410-4250-bfb0-0099f3412409">
</p>


<p align="center">
  <a href="https://takerum.github.io/">Takeru Miyato</a>
  ·
  <a href="https://sindylowe.com/">Sindy Löwe</a>
  ·
  <a href="https://www.cvlibs.net/">Andreas Geiger</a>
   ·
  <a href="https://staff.fnwi.uva.nl/m.welling/">Max Welling</a>
</p>
  <h3 align="center"> <a href="https://arxiv.org/abs/2410.13821">arXiv</a> </h3>

This page contains instructions for the initial environment setup and code for the CLEVR-Tex experiments.
- Minimal AKOrN model on Google Colab (The fish example in the paper) [here](https://colab.research.google.com/drive/1n8x2uskNxRIqJvvNaljWDuLAMvxkw0Qn)
- Code for other synthetic datasets (Tetrominoes, dSprits, CLEVR): [here](https://github.com/autonomousvision/akorn/blob/main/scripts/synths.md)
- Sudoku solving:  [here](https://github.com/autonomousvision/akorn/blob/main/scripts/sudoku.md)

## Setup Conda env

```
yes | conda create -n akorn python=3.12
conda activate akorn
pip3 install -r requirements.txt
```

## Donwload the CLEVRTex dataset
```
cd data
bash download_clevrtex.sh
cd ..
```

## Training
```
export NUM_GPUS=<number_of_gpus> # If you use a single GPU, run a command without the multi GPU option  (`--multi-gpu`).
```

### CLEVRTex

#### AKOrN 
```
export L=1 # The number of layers. L=1 or 2. This can be >2, but we only experimented with a single or two-layer model.
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_akorn --data_root=./data/clevrtex_full/ --model=akorn --data=clevrtex_full --J=attn --L=${L}

# Larger model (L=2, ch=512, bs=512)
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_large_akorn --data_root=./data/clevrtex_full/ --model=akorn --data=clevrtex_full --J=attn --L=2 --ch=512 --batchsize=512 --epochs=1024 --lr=0.0005
```

#### ItrSA
```
export L=1
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_itrsa --data_root=./data/clevrtex_full/ --model=vit --data=clevrtex_full --L=${L} --gta=False
```

## Evaluation

### CLEVRTex (-OOD, -CAMO) 

```
export DATA_TYPE=full #{full, outd, camo}
export L=1
# AKOrN
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=akorn  --data=clevrtex_${DATA_TYPE} --J=attn --L=${L} --model_path=runs/clvtex_akorn/ema_499.pth --model_imsize=128
# ItrSA
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=vit  --data=clevrtex_${DATA_TYPE} --gta=False --L=${L} --model_path=runs/clvtex_itrsa/ema_499.pth --model_imsize=128
```

### Eval with Up-tiling (See Appendix section).
```
# Might take long time depending on the CPU spec
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --saccade_r=4 --model=akorn  --data=clevrtex_${DATA_TYPE} --J=attn --L=${L} --model_path=runs/clvtex_akorn/ema_499.pth --model_imsize=128
```

#### Performance table
| Model                              | CLEVRTex FG-ARI | CLEVRTex MBO | OOD FG-ARI | OOD MBO | CAMO FG-ARI | CAMO MBO |
|------------------------------------|-----------------|--------------|------------|---------|-------------|----------|
| ViT                                | 46.4±0.6        | 25.1±0.7     | 44.1±0.5 |  27.2±0.5   | 32.5±0.6    | 16.1±1.1 |
| ItrSA (L = 1)              | 65.7±0.3        | 44.6±0.9     | 64.6±0.8 | 45.1±0.4   | 49.0±0.7    | 30.2±0.8 |
| ItrSA (L = 2)              | 76.3±0.4        | 48.5±0.1     | 74.9±0.8 | 46.4±0.5   | 61.9±1.3    | 37.1±0.5 |
| AKOrN (attn, L = 1)         | 75.6±0.2        | 55.0±0.0     | 73.4±0.4 | 56.1±1.1   | 59.9±0.1    | 44.3±0.9 |
| AKOrN (attn, L = 2)              | 80.5±1.5        | 54.9±0.6     | 79.2±1.2 | 55.7±0.5   | 67.7±1.5    | 46.2±0.9 |

##### With Up-tiling (x4)
| Model                              | CLEVRTex FG-ARI | CLEVRTex MBO | OOD FG-ARI | OOD MBO | CAMO FG-ARI | CAMO MBO |
|------------------------------------|-----------------|--------------|------------|---------|-------------|----------|
| AKOrN (attn, L = 2)            | 87.7±1.0        | 55.3±2.1     | 85.2±0.9 | 55.6±1.5   | 74.5±1.2    | 45.6±3.4 |
| Large AKOrN (attn, L = 2)       | 88.5±0.9        | 59.7±0.9     | 87.7±0.5 | 60.8±0.6  | 77.0±0.5    | 53.4±0.7 |
