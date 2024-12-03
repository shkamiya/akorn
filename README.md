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

This page contains an initial env setup and code for the CLEVR-Tex experiments. 
- Code for other synthetic datasets (Tetrominoes, dSprits, CLEVR) is [here](xx)

## Setup Conda env

```
yes | conda create -n akorn python=3.12
conda activate akorn
pip3 install -r requirements.txt
```

<<<<<<< HEAD
## Donwload the CLEVRTex dataset
```
cd data
bash download_clevrtex.sh
=======
## Donwload Synthetic datasets
```
cd data
bash download_synths.sh
bash download_clevrtex.sh
python create_shapes.py # create Shapes dataset 
>>>>>>> e6ef32db5e5ef7175f5ea8272e369ffa2319a11e
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
accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=clvtex_large_akorn --data_root=./data/clevrtex_full/ --model=akorn --data=clevrtex_full --J=attn --L=${L} --ch=512 --batchsize=512 --epochs=1024 --lr=0.0005
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
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=akorn  --data=clevrtex_${DATA_TYPE} --J=attn --L=$L$ --model_path=runs/clvtex_akorn/ema_499.pth --model_imsize=128
# ItrSA
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --model=vit  --data=clevrtex_${DATA_TYPE} --gta=False --L=$L$ --model_path=runs/clvtex_itrsa/ema_499.pth --model_imsize=128
```

### Eval with Up-tiling (See Appendix section).
```
# Might take long time depending on the CPU spec
python eval_obj.py  --data_root=./data/clevrtex_${DATA_TYPE}/  --saccade_r=4 --model=akorn  --data=clevrtex_${DATA_TYPE} --J=attn --L=$L$ --model_path=runs/clvtex_akorn/ema_499.pth --model_imsize=128
```

<<<<<<< HEAD
#### Performance table
| Model                              | CLEVRTex FG-ARI | CLEVRTex MBO | OOD FG-ARI | OOD MBO | CAMO FG-ARI | CAMO MBO |
|------------------------------------|-----------------|--------------|------------|---------|-------------|----------|
| ViT                                | 46.4$\pm$0.6   | 25.1$\pm$0.7 | 27.2$\pm$0.5 | 16.1$\pm$1.1 | 32.5$\pm$0.6 | 16.1$\pm$1.1 |
| ItrSA ($L = 1, T = 8$)            | 65.7$\pm$0.3   | 44.6$\pm$0.9 | 45.1$\pm$0.4 | 30.2$\pm$0.8 | 49.0$\pm$0.7 | 30.2$\pm$0.8 |
| ItrSA ($L = 2, T = 8$)            | 76.3$\pm$0.4   | 48.5$\pm$0.1 | 46.4$\pm$0.5 | 37.1$\pm$0.5 | 61.9$\pm$1.3 | 37.1$\pm$0.5 |
| AKOrN$^\text{attn}$ ($L = 1, T = 8$) | 75.6$\pm$0.2 | 55.0$\pm$0.0 | 56.1$\pm$1.1 | 44.3$\pm$0.9 | 59.9$\pm$0.1 | 44.3$\pm$0.9 |
| AKOrN$^\text{attn}$ ($L = 2, T = 8$) | 80.5$\pm$1.5 | 54.9$\pm$0.6 | 55.7$\pm$0.5 | 46.2$\pm$0.9 | 67.7$\pm$1.5 | 46.2$\pm$0.9 |
| (+up-tiling ($\times 4$))          |                 |              |             |          |              |          |
| AKOrN$^\text{attn}$ ($L = 2, T = 8$) | 87.7$\pm$1.0 | 55.3$\pm$2.1 | 55.6$\pm$1.5 | 45.6$\pm$3.4 | 74.5$\pm$1.2 | 45.6$\pm$3.4 |
| Large AKOrN$^\text{attn}$ ($L = 2, T = 8$) | 88.5$\pm$0.9 | 59.7$\pm$0.9 | 60.8$\pm$0.6 | 53.4$\pm$0.7 | 77.0$\pm$0.5 | 53.4$\pm$0.7 |
=======

#### Performance table
| Model                              | CLEVRTex FG-ARI | CLEVRTex MBO | OOD FG-ARI | OOD MBO | CAMO FG-ARI | CAMO MBO |
|------------------------------------|-----------------|--------------|------------|---------|-------------|----------|
| ViT                                | 46.4\spm{0.6}   | 25.1\spm{0.7}| 27.2\spm{0.5}| 16.1\spm{1.1} | 32.5\spm{0.6} | 16.1\spm{1.1} |
| ItrSA (\(L = 1\))          | 65.7\spm{0.3}   | 44.6\spm{0.9}| 45.1\spm{0.4}| 30.2\spm{0.8} | 49.0\spm{0.7} | 30.2\spm{0.8} |
| ItrSA (\(L = 2\))          | 76.3\spm{0.4}   | 48.5\spm{0.1}| 46.4\spm{0.5}| 37.1\spm{0.5} | 61.9\spm{1.3} | 37.1\spm{0.5} |
| AKOrN\(^\text{attn}\) (\(L = 1\)) | 75.6\spm{0.2} | 55.0\spm{0.0}| 56.1\spm{1.1}| 44.3\spm{0.9} | 59.9\spm{0.1} | 44.3\spm{0.9} |
| AKOrN\(^\text{attn}\) (\(L = 2\)) | 80.5\spm{1.5} | 54.9\spm{0.6}| 55.7\spm{0.5}| 46.2\spm{0.9} | 67.7\spm{1.5} | 46.2\spm{0.9} |
| (+up-tiling (\(\times 4\)))        |                 |              |            |         |             |          |
| AKOrN\(^\text{attn}\) (\(L = 2\)) | 87.7\spm{1.0} | 55.3\spm{2.1}| 55.6\spm{1.5}| 45.6\spm{3.4} | 74.5\spm{1.2} | 45.6\spm{3.4} |
| Large AKOrN\(^\text{attn}\) (\(L = 2\)) | 88.5\spm{0.9} | 59.7\spm{0.9}| 60.8\spm{0.6}| 53.4\spm{0.7} | 77.0\spm{0.5} | 53.4\spm{0.7} |
>>>>>>> e6ef32db5e5ef7175f5ea8272e369ffa2319a11e
