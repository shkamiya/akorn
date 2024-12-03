# Artificial Kuramoto Oscillatory Neurons (AKOrN)

## Donwload the synthetic datasets (Tetrominoes, dSprits, CLEVR) 
```
# Need gsutil installed. run `conda install conda-forge::gsutil` to install it or manually download datasets from https://console.cloud.google.com/storage/browser/multi-object-datasets;tab=objects?pli=1&inv=1&invt=AbjJBg&prefix=&forceOnObjectsSortingFiltering=false
cd data
bash download_synths.sh
cd ..
```


## Training
```
export NUM_GPUS=<number_of_gpus> # If you use a single GPU, run a command without the multi GPU option arguments (`--multi-gpu --num_processes=$NUM_GPUS`).
```

#### AKOrN (Attentive models)
```
#Tetrominoes 
export dataset=tetrominoes; accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_attn --model=akorn --data=${dataset}  --J=attn --L=1 --T=8 --ch=128 --psize=4 --epochs=50 --c_norm=none
#dSprites
export dataset=dsprites; accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_attn  --model=akorn --data=${dataset} --J=attn --L=1 --ch=128 --psize=4 --epochs=50 --c_norm=none
#CLEVR
export dataset=clevr accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_attn  --model=akorn --data=${dataset} --J=attn --L=1 --ch=256 --psize=8 --epochs=300 --c_norm=none
```

#### AKOrN (Convolutional models)
```
#Tetrominoes 
export dataset=tetrominoes; accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_conv --model=akorn --data=${dataset}  --J=conv --L=1 --ksize=5 --T=8 --ch=128 --psize=4 --epochs=50 --c_norm=none
#dSprites
export dataset=dsprites; accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_conv  --model=akorn --data=${dataset} --J=conv --L=1 --ksize=7 --ch=128 --psize=4 --epochs=50 --c_norm=none
#CLEVR
export dataset=clevr accelerate launch --num_processes=${NUM_GPUS} train_obj.py --exp_name=${dataset}_akorn_conv  --model=akorn --data=${dataset} --J=conv --L=1 --ksize=7 --ch=256 --psize=8 --epochs=300 --c_norm=none
```

#### ItrSA
```
export L=1
export dataset=tetrominoes; accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=${dataset}_itrsa  --data=${dataset} --model=vit  --L=1 --gta=False --T=8 --ch=128 --psize=4 --epochs=50 
export dataset=dsprites; accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=${dataset}_itrsa  --data=${dataset} --model=vit  --L=1 --gta=False  --T=8 --ch=128 --psize=4 --epochs=50 
export dataset=clevr; accelerate launch --multi-gpu --num_processes=$NUM_GPUS  train_obj.py --exp_name=${dataset}_itrsa  --data=${dataset} --model=vit  --L=1 --gta=False --T=8 --ch=256 --psize=8 --epochs=300
```

## Evaluation

```
export DATA=tetrominoes #{tetrominoes, dsprits, clevr}. Please adjust the model parameters (–model, –J, –ch, –psize) based on the dataset and model you want to evaluate.
export IMSIZE=32 # {32:tetrominoes, 64:dsprites, 128:clevr}.
python eval_obj.py --model=akorn --data=${DATA} --J=attn --L=$L$ --model_path=runs/${dataset}_akorn_attn --model_imsize=32 --J=attn --L=1 --T=8 --ch=128 --psize=4 --c_norm=none

```

#### Performance table
| Model                   | Tetrominoes FG-ARI | Tetrominoes MBO | dSprites FG-ARI | dSprites MBO | CLEVR FG-ARI | CLEVR MBO |
|-------------------------|--------------------|-----------------|-----------------|--------------|--------------|-----------|
| ItrConv                | 59.0\spm{2.9}     | 51.6\spm{2.2}   | 29.1\spm{6.2}   | 38.5\spm{5.2}| 49.3\spm{5.1}| 29.7\spm{3.0} |
| AKOrN\(^\text{conv}\)  | 76.4\spm{0.8}     | 51.9\spm{1.5}   | 63.8\spm{7.7}   | 50.7\spm{4.7}| 59.0\spm{4.3}| 44.4\spm{2.0} |
| ItrSA                  | 85.8\spm{0.8}     | 54.9\spm{3.4}   | 68.1\spm{1.4}   | 63.0\spm{1.2}| 82.5\spm{1.7}| 39.4\spm{1.9} |
| AKOrN\(^\text{attn}\)  | 88.6\spm{1.7}     | 56.4\spm{0.9}   | 78.3\spm{1.3}   | 63.0\spm{1.8}| 91.0\spm{0.5}| 45.5\spm{1.4} |
| (+up-tiling (\(\times 4\))) |                    |                 |                 |              |              |           |
| AKOrN\(^\text{attn}\)  | 93.1\spm{0.3}     | 56.3\spm{0.0}   | 87.1\spm{1.0}   | 60.2\spm{1.9}| 94.6\spm{0.7}| 44.7\spm{0.7} |

