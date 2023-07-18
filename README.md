<div align="center">   


# YOLOS with EVA-02

</div>

## Introduction

You Only Look at One Sequence (YOLOS) ([paper](https://arxiv.org/abs/2106.00666), [code](https://github.com/hustvl/YOLOS)) is a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. With pre-training on the ImageNet-1k dataset and fine-tuning on the COCO dataset, the transfer learning performance of YOLOS reflected on COCO object detection dataset can serve as a challenging transfer learning benchmark to evaluate different (label-supervised or self-supervised) pre-training strategies for ViT.

EVA-02 ([paper](https://arxiv.org/abs/2303.11331), [code](https://github.com/baaivision/EVA/tree/master/EVA-02)) is a series of visual pre-training models based on the ViT architecture and Masked Image Modeling (MIM) pre-training strategy. After fine-tuning for downstream tasks, the EVA-02 model demonstrates superior performance compared to previous models in various downstream tasks such as image classification and object detection.

This project applies the YOLOS method to EVA-02 models and evaluates their performance on the VOC2007 dataset to reveal the transferability of EVA-02 pretraining models.

## Results

|Model |Params |Pre-train Epochs |  Init Weight | Fine-tune Epochs | Eval Size | YOLOS Checkpoint / Log | AP @ VOC2007 test |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |:------------: |
|`VOC-YOLOS-Ti`|6M|300|[DeiT-tiny](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)|300|512|[checkpoint](https://huggingface.co/Robert-zwr/EVA-YOLOS/resolve/main/checkpoints/voc_yolos_ti.pth) / [Log](https://gist.github.com/Robert-zwr/c011a5b0ba5fc71e6f09abdf8cc84efc)|23.9
|`VOC-YOLOS-S`|23M|300|[DeiT-small](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)|150|512|[checkpoint](https://huggingface.co/Robert-zwr/EVA-YOLOS/resolve/main/checkpoints/voc_yolos_s.pth) / [Log](https://gist.github.com/Robert-zwr/3a12a61886b53c3e51e47bb1d00b0d53)|31.1
|`VOC-YOLOS-EVA-Ti`|6M|240+100|[EVA02-tiny](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|300|512|[checkpoint](https://huggingface.co/Robert-zwr/EVA-YOLOS/resolve/main/checkpoints/voc_yolos_eva_ti.pth) / [Log](https://gist.github.com/Robert-zwr/32ae183c4fd07244f3f7b58ee8c39903) |31.9
|`VOC-YOLOS-EVA-S`|23M|240+100|[EVA02-small](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|150|512|[checkpoint](https://huggingface.co/Robert-zwr/EVA-YOLOS/resolve/main/checkpoints/voc_yolos_eva_s.pth) / [Log](https://gist.github.com/Robert-zwr/87b0f14c966a57a64962bf81631e7b66)|42.0

**Notes**: 

- The `Pre-train Epochs` of `VOC-YOLOS-EVA` is `240+100` , which means 240 MIM pre-training epochs and 100 IN-1K fine-tuned epochs. In other words, we use [IN-1K fine-tuned EVA-02 weights](https://github.com/baaivision/EVA/tree/master/EVA-02/asuka#in-1k-fine-tuned-eva-02-wo-in-21k-intermediate-fine-tuning) as initial checkpoint. The reason why we don't choose to directly use MIM pre training weights as the initial weights is due to the small size of the VOC2007 dataset, which makes it difficult to start training from MIM models that have never seen real images before. Subsequent experiments have also proven this point: For the Tiny model, the performance difference between the two is not significant. But for Small model, model trained from MIM weights performs poorly.
- For EVA models, We interpolate the kernel size of `patch_embed` from `14x14` to `16x16`. This is useful for object detection, instance segmentation & semantic segmentation tasks.
- The comparison of these results may not be fair, as the EVA model uses more data during the pre-training process(IN-21K).

## Partial finetune results

+ Tiny models

|Model |Params(adjustable) |  Init Weight| finetune type | Fully adjustable layers | Log | AP @ VOC2007 test |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|`VOC-YOLOS-Ti`|6M|[DeiT-tiny](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)|full|12| [Log](https://gist.github.com/Robert-zwr/c011a5b0ba5fc71e6f09abdf8cc84efc)|23.9
|`VOC-YOLOS-EVA-Ti`|6M|[eva02_Ti_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|full|12| [Log](https://gist.github.com/Robert-zwr/32ae183c4fd07244f3f7b58ee8c39903) |31.9
|`VOC-YOLOS-Ti-0`|3.7M|[DeiT-tiny](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth)|ffn|0| [Log](https://gist.github.com/Robert-zwr/319519eacb981148cfc9441857926844)|9.4
|`VOC-YOLOS-EVA-Ti-0`|3.7M|[eva02_Ti_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|ffn|0| [Log](https://gist.github.com/Robert-zwr/b1330db38105819162fdb707d68dbc41) |10.9
|`VOC-YOLOS-EVA-Ti-1`|4.2M|[eva02_Ti_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|ffn|1| [Log](https://gist.github.com/Robert-zwr/c9b36c196ffc0ce7ce18b4541f002d5b) |11.6
|`VOC-YOLOS-EVA-Ti-2`|4.6M|[eva02_Ti_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|ffn|2| [Log](https://gist.github.com/Robert-zwr/1736ca4766793df72f1072f4bb7482f0) |12.4
|`VOC-YOLOS-EVA-Ti-3`|5.1M|[eva02_Ti_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_Ti_pt_in21k_ft_in1k_p14.pt)|ffn|3| [Log](https://gist.github.com/Robert-zwr/d223280c7889ced628e58b29d7a664f0) |16.2

+ Small models

|Model |Params(adjustable) |  Init Weight| finetune type | Fully adjustable layers | Log | AP @ VOC2007 test |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
|`VOC-YOLOS-S`|23M|[DeiT-small](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)|full|12| [Log](https://gist.github.com/Robert-zwr/3a12a61886b53c3e51e47bb1d00b0d53)|31.1
|`VOC-YOLOS-EVA-S`|23M|[eva02_S_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|full|12| [Log](https://gist.github.com/Robert-zwr/87b0f14c966a57a64962bf81631e7b66)|42.0
|`VOC-YOLOS-S-0`|15M|[DeiT-small](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth)|ffn|0| [Log](https://gist.github.com/Robert-zwr/e50217898587d9e40e30da559d62233d)|12.2
|`VOC-YOLOS-EVA-S-0`|15M|[eva02_S_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|ffn|0| [Log](https://gist.github.com/Robert-zwr/a2ed8397c9eba8f6687e868e04073561) |21.0
|`VOC-YOLOS-EVA-S-MIM-0`|15M|[eva02_S_pt_in21k_p14](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt)|ffn|0| [Log](https://gist.github.com/Robert-zwr/4912365ae48a64ae5714a50d5402ceb6) |23.0
|`VOC-YOLOS-EVA-S-1`|17M|[eva02_S_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|ffn|1| [Log](https://gist.github.com/Robert-zwr/2c531ebdacc20a191e1ebc2f7374ffb2) |23.4
|`VOC-YOLOS-EVA-S-MIM-1`|17M|[eva02_S_pt_in21k_p14](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt)|ffn|1| [Log](https://gist.github.com/Robert-zwr/a09391500f35eb73bdb428b6838c7fb2) |24.0
|`VOC-YOLOS-EVA-S-2`|18M|[eva02_S_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|ffn|2| [Log](https://gist.github.com/Robert-zwr/c6ba5a7bf48fd1ecd66ddd280e0f98fa) |25.8
|`VOC-YOLOS-EVA-S-MIM-2`|18M|[eva02_S_pt_in21k_p14](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt)|ffn|2| [Log](https://gist.github.com/Robert-zwr/ccfc807658305f9df47d01450b7b423e) |30.7
|`VOC-YOLOS-EVA-S-3`|20M|[eva02_S_pt_in21k_ft_in1k_p14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in1k/eva02_S_pt_in21k_ft_in1k_p14.pt)|ffn|3| [Log](https://gist.github.com/Robert-zwr/798f8f06413cc188bea12a2477036a53) |32.6
|`VOC-YOLOS-EVA-S-MIM-3`|20M|[eva02_S_pt_in21k_p14](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt)|ffn|3| [Log](https://gist.github.com/Robert-zwr/6543bdb46bc1df515111bfe1aba5192d) |35.4
|`VOC-YOLOS-EVA-S-attn-0`|8M|[eva02_S_pt_in21k_p14](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_S_pt_in21k_p14.pt)|attn|0| [checkpoint](https://huggingface.co/Robert-zwr/EVA-YOLOS/blob/main/checkpoints/VOC-YOLOS-EVA-S-attn-0.pth) / [Log](https://gist.github.com/Robert-zwr/496a5c3fa09ffb21c3724f73567bbe60) |42.4

## Requirement

Please reference to Requirement of YOLOS [here](https://github.com/hustvl/YOLOS#requirement) to build the environment.

Further, you also need to install timm and einops for EVA model:

```
pip install timm einops
```



## Data preparation

We use VOC2007 trainval to train and VOC2007 test to eval.

Download and extract Pascal VOC 2007 images and annotations:

```
# Download the data.
cd $HOME/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

Now you should see VOCdevkit folder.

Then run voc2coco.py to convert VOC annotations to COCO format.

```
python voc2coco.py /path/to/VOCdevkit
```
Now you should see voc_train.json and voc_val.json.

We expect the dataset directory structure to be the following:

```
path/to/dataset/
  annotations/
  	voc_train.json
  	voc_val.json
  images/
  	train/	# VOC 2007 trainval images
  	val/	# VOC 2007 test images
```

## Training

Before finetuning on VOC2007, you need download the pre-trained model.

<details>
<summary>To train the original <code>VOC-YOLOS-Ti</code> model on VOC2007, run this command:</summary>
<pre><code>
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 2 --lr 2.5e-5 --epochs 300 --backbone_name tiny --pre_trained path/to/deit-tiny.pth --eval_size 512 --init_pe_size 608 800 --output_dir /output/path/box_model
</code></pre>
</details>


<details>
<summary>To train the original <code>VOC-YOLOS-S</code> model on VOC2007, run this command:</summary>
<pre><code>
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --lr 2.5e-5 --epochs 150 --backbone_name small --pre_trained path/to/deit-small-300epoch.pth --eval_size 512 --init_pe_size 512 864 --mid_pe_size 512 864 --output_dir /output/path/box_model
</code></pre>
</details>


<details>
<summary>To train the <code>VOC-YOLOS-EVA-Ti</code> model on VOC2007, run this command:</summary>
<pre><code>
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 2 --lr 2.5e-5 --epochs 300 --model_name eva --backbone_name tiny --pre_trained path/to/eva02_Ti_pt_in21k_ft_in1k_p14.pt --eval_size 512 --init_pe_size 608 800 --output_dir /output/path/box_model
</code></pre>
</details>


<details>
<summary>To train the <code>EVA-YOLOS-EVA-S</code> model on VOC2007, run this command:</summary>
<pre><code>
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --lr 2.5e-5 --epochs 150 --model_name eva --backbone_name small --pre_trained path/to/eva02_S_pt_in21k_ft_in1k_p14.pt --eval_size 512 --init_pe_size 608 800 --output_dir /output/path/box_model
</code></pre>
</details>


<details>
<summary>To apply attention partial finetune on <code>EVA-YOLOS-EVA-S-MIM</code> model on VOC2007, run this command:</summary>
<pre><code>
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --lr 2.5e-5 --epochs 150 --model_name eva --backbone_name small_mim --use_partial_finetune --partial_finetune_type attn --pre_trained path/to/eva02_S_pt_in21k_p14.pt --eval_size 512 --init_pe_size 608 800 --output_dir /output/path/box_model
</code></pre>
</details>


## Evaluation

To evaluate `VOC-YOLOS-Ti` model on VOC2007 test, run:

```eval
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 2 --backbone_name tiny --eval --eval_size 512 --init_pe_size 608 800 --resume path/to/voc_yolos_ti.pth
```

To evaluate `VOC-YOLOS-S` model on VOC2007 test, run:

```eval
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --backbone_name small --eval --eval_size 512 --init_pe_size 512 864 --mid_pe_size 512 864 --resume path/to/voc_yolos_s.pth
```

To evaluate `VOC-YOLOS-EVA-Ti` model on VOC2007 test, run:

```eval
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 2 --model_name eva --backbone_name tiny --eval --eval_size 512 --init_pe_size 608 800 --resume path/to/voc_yolos_eva_ti.pth
```

To evaluate `VOC-YOLOS-EVA-S` model on VOC2007 test, run:

```eval
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --model_name eva --backbone_name small --eval --eval_size 512 --init_pe_size 608 800 --resume path/to/voc_yolos_eva_s.pth
```

To evaluate `VOC-YOLOS-EVA-S-attn-0` model on VOC2007 test, run:

```eval
python -m torch.distributed.launch --nproc_per_node=3 --use_env main.py --coco_path /path/to/dataset --dataset_file voc --batch_size 1 --model_name eva --backbone_name small_mim --eval --eval_size 512 --init_pe_size 608 800 --resume path/to/voc_eva_s_mim_frozen_attn_0.pth
```
