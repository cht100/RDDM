# RDDM SR

这个目录从 RDDM 官方 image restoration 实验复制而来，只改训练/测试入口。RDDM 官方 restoration 代码要求输入和 GT 同尺寸，因此 x4 SR 使用 `bicubic(LR) -> HR` 的同尺寸恢复方式训练，与当前 SR 测试适配器一致。

默认读取：

```text
data/DIV2K_x4_sr/flists/train_gt.flist
data/DIV2K_x4_sr/flists/train_input.flist
data/DIV2K_x4_sr/flists/test_gt.flist
data/DIV2K_x4_sr/flists/test_input.flist
```

准备数据：

```bash
python scripts/sr/prepare_rddm_sr_pairs.py \
  --train-lr-dir /data1/jiangtaoren/datasets/DIV2K/train/input \
  --train-gt-dir /data1/jiangtaoren/datasets/DIV2K/train/target \
  --test-lr-dir /data1/jiangtaoren/datasets/DIV2K/test/input \
  --test-gt-dir /data1/jiangtaoren/datasets/DIV2K/test/target \
  --scale 4 \
  --lr-suffix '' \
  --gt-template '{filename}' \
  --output-root /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sr/data/DIV2K_x4_sr
```

训练：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --sampling-timesteps 5 \
  --train-num-steps 120000 \
  --save-and-sample-every 10000 \
  --results-folder ./results/sr_x4
```
