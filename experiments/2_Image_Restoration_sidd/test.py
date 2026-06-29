import argparse
import os
import sys
from pathlib import Path
from types import ModuleType


def parse_args():
    parser = argparse.ArgumentParser(description="RDDM SIDD paired denoising test.")
    parser.add_argument("--train-gt-flist", default="data/SIDD_paired/flists/train_gt.flist")
    parser.add_argument("--train-input-flist", default="data/SIDD_paired/flists/train_input.flist")
    parser.add_argument("--test-gt-flist", default="data/SIDD_paired/flists/test_gt.flist")
    parser.add_argument("--test-input-flist", default="data/SIDD_paired/flists/test_input.flist")
    parser.add_argument("--results-folder", default="./results/sidd")
    parser.add_argument("--output-folder", default="./results/sidd_test")
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--sampling-timesteps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda-devices", default=None)
    return parser.parse_args()


args = parse_args()
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

if args.cuda_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
dataset_root = script_dir / "datasets"
datasets_package = ModuleType("datasets")
datasets_package.__path__ = [str(dataset_root)]
datasets_package.__package__ = "datasets"
sys.modules["datasets"] = datasets_package

from src.residual_denoising_diffusion_pytorch import (  # noqa: E402
    ResidualDiffusion,
    Trainer,
    UnetRes,
    set_seed,
)


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((script_dir / path).resolve())


sys.stdout.flush()
set_seed(args.seed)

condition = True
input_condition = False
input_condition_mask = False
folder = [
    resolve_path(args.train_gt_flist),
    resolve_path(args.train_input_flist),
    resolve_path(args.test_gt_flist),
    resolve_path(args.test_input_flist),
]

model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    share_encoder=0,
    condition=condition,
    input_condition=input_condition,
)
diffusion = ResidualDiffusion(
    model,
    image_size=args.image_size,
    timesteps=1000,
    sampling_timesteps=args.sampling_timesteps,
    objective="pred_res_noise",
    loss_type="l1",
    condition=condition,
    sum_scale=1,
    input_condition=input_condition,
    input_condition_mask=input_condition_mask,
)

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=1,
    num_samples=1,
    train_lr=8e-5,
    train_num_steps=1,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=False,
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=1000,
    equalizeHist=False,
    crop_patch=True,
    generation=False,
    results_folder=args.results_folder,
)

if trainer.accelerator.is_local_main_process:
    trainer.load(args.milestone)
    trainer.set_results_folder(args.output_folder)
    trainer.test(last=True)
