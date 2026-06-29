# RDDM Denoising

这个目录从 RDDM 官方 image restoration 实验目录复制而来，仅把训练/测试入口改成 DFWB 高斯去噪任务使用的路径和参数。

数据格式保持 RDDM 官方 paired restoration 逻辑：

```text
train_gt.flist     clean 训练图
train_noisy.flist  noisy 训练图
test_gt.flist      clean 测试图
test_noisy.flist   noisy 测试图
```

默认读取：

```text
data/DFWB_sigma25_color/flists/train_gt.flist
data/DFWB_sigma25_color/flists/train_noisy.flist
data/DFWB_sigma25_color/flists/test_gt.flist
data/DFWB_sigma25_color/flists/test_noisy.flist
```

训练示例：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --sampling-timesteps 5 \
  --train-num-steps 120000 \
  --save-and-sample-every 1000 \
  --results-folder ./results/denoising_sigma25_color
```

续训示例：

```bash
CUDA_VISIBLE_DEVICES=3 python train.py \
  --sampling-timesteps 5 \
  --resume-milestone 40 \
  --results-folder ./results/denoising_sigma25_color
```

测试示例：

```bash
CUDA_VISIBLE_DEVICES=3 python test.py \
  --milestone 120 \
  --sampling-timesteps 5 \
  --results-folder ./results/denoising_sigma25_color \
  --output-folder ./results/denoising_sigma25_color_test
```
