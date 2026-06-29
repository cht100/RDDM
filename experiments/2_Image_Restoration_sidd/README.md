# RDDM SIDD 训练

这个目录用于 RDDM 的 SIDD 真实图像去噪训练。模型结构和采样逻辑仍然使用 RDDM 官方 restoration 代码：`UnetRes + ResidualDiffusion + Trainer`。

SIDD 是成对真实去噪任务，输入是 noisy，GT 是 clean，二者尺寸必须一致。数据准备脚本只负责转 RGB、复制成统一文件名并生成四个 flist，不生成合成噪声。

## 生成数据对和 flist

```bash
python scripts/sidd_denoising/prepare_rddm_sidd_pairs.py \
  --train-input-dir /data1/jiangtaoren/datasets/SIDD/train/input \
  --train-gt-dir /data1/jiangtaoren/datasets/SIDD/train/target \
  --test-input-dir /data1/jiangtaoren/datasets/SIDD/test/input \
  --test-gt-dir /data1/jiangtaoren/datasets/SIDD/test/target \
  --gt-template '{filename}' \
  --output-root /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sidd/data/SIDD_paired
```

如果 noisy 和 clean 文件名不同，例如 noisy 文件名多了 `_noisy` 后缀，可以使用：

```bash
--input-suffix _noisy --gt-template '{base}.png'
```

## 启动训练

```bash
cd /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sidd

CUDA_VISIBLE_DEVICES=3 python train.py \
  --sampling-timesteps 5 \
  --train-num-steps 120000 \
  --save-and-sample-every 10000 \
  --results-folder ./results/sidd
```

默认保存 `model-12.pt`，对应 120000 step / 每 10000 step 保存一次。

## 单独测试官方 Trainer 输出

```bash
CUDA_VISIBLE_DEVICES=3 python test.py \
  --milestone 12 \
  --sampling-timesteps 5 \
  --results-folder ./results/sidd \
  --output-folder ./results/sidd_test
```
