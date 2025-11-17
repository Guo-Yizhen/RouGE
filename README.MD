## RouGE: Learning Gated Experts for Segment Anything in the Wild

#### RouGE module

The Code for RouGE module can be found in 

```
./model/RouGE.py
```

In practice, RouGE is inserted in image encoder of efficientSAM model. Details can be found in 

```
./efficient_sam/efficient_sam_encoder.py
```

#### Trainning

To train RouGE from scratch, run

```
python train.py -e 20 --local_rank 0 --RouGE_start_index 10 -bz 16 --loss dice_sigfocal --lr 1e-4
```

Dataset and annotations will be released soon

The folder path should be

```
-anno
   |-LIS
   |-GoPro
   |-Rain200L
   |-DDN
   |-snow100k
   
-data
   |-LIS
   |-GoPro
   |-Rain200L
   |-DDN
   |-snow100k
```

#### Testing

The pretrained weights of RouGE can be found in ./ckpt/train/all/RouGE_from_10/Weights_for_RouGE

To test RouGE's performance, run

```
python mAP_cal.py
```

#### Thanks

The base model EfficientSAM and its code are from https://github.com/yformer/EfficientSAM

Thanks [Yunyang Xiong](https://github.com/yformer) and other authors for their excellent work~
