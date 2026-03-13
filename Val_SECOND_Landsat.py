import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from skimage import io, exposure
from torch.utils.data import DataLoader
# 注意：请确保这两个导入路径正确，若报错需根据实际项目结构调整
from datasets.RS_ST import get_dataset_config
from datasets import RS_ST as RS
from models.HyFuSCD import HyFuSCD as Net
from utils.SCD_misc import ConfuseMatrixMeter


def index2color(pred, colormap):
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(colormap):
        pred_rgb[pred == idx] = color
    return pred_rgb


class PredEvalOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--dataname', required=False, type=str, default='SECOND',
                            help='SECOND, Landsat')
        parser.add_argument('--chkpt_path', required=False, type=str,
                            default='/root/***/***.pth',
                            help='Path to trained model weights (.pth)')
        parser.add_argument('--test_dir', required=False, type=str, default='/root/***/SECOND',
                            help='Root directory of test set')
        parser.add_argument('--out_root', required=False, default='/root/***',
                            help='Root directory for saving prediction maps and evaluation results')

        parser.add_argument('--pred_batch_size', required=False, default=1, type=int,
                            help='Batch size for prediction (default 1 to avoid OOM)')
        parser.add_argument('--image_format', required=False, default='png',
                            help='Image format (without dot), e.g., png/jpg/tif')
        parser.add_argument('--flip_tta', required=False, default=False, type=bool,
                            help='Whether to enable flip TTA augmentation (default False)')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        weight_name = os.path.basename(self.opt.chkpt_path).split('.')[0]
        self.opt.pred_dir = os.path.join(self.opt.out_root, self.opt.dataname, weight_name, 'out')
        self.opt.eval_dir = os.path.join(self.opt.out_root, self.opt.dataname, weight_name, 'eval')

        self.opt.im1_dir = os.path.join(self.opt.pred_dir, 'im1')
        self.opt.im2_dir = os.path.join(self.opt.pred_dir, 'im2')
        self.opt.im1_rgb_dir = os.path.join(self.opt.pred_dir, 'im1_rgb')  # 变化区域语义
        self.opt.im2_rgb_dir = os.path.join(self.opt.pred_dir, 'im2_rgb')  # 变化区域语义
        self.opt.im1_semantic_dir = os.path.join(self.opt.pred_dir, 'im1_semantic')  # 完整语义
        self.opt.im2_semantic_dir = os.path.join(self.opt.pred_dir, 'im2_semantic')  # 完整语义
        self.opt.change_dir = os.path.join(self.opt.pred_dir, 'change')

        for dir_path in [self.opt.pred_dir, self.opt.eval_dir, self.opt.im1_dir, self.opt.im2_dir,
                         self.opt.im1_rgb_dir, self.opt.im2_rgb_dir, self.opt.change_dir,
                         self.opt.im1_semantic_dir, self.opt.im2_semantic_dir]:
            os.makedirs(dir_path, exist_ok=True)
        return self.opt


def main():
    begin_time = time.time()
    opt = PredEvalOptions().parse()

    dataset_config = get_dataset_config(opt.dataname)
    num_class = dataset_config["num_classes"]
    ST_COLORMAP = dataset_config["ST_COLORMAP"]
    print(f"===== Loading {dataset_config['ST_CLASSES']} =====")
    print(f"Number of classes: {num_class} | Model weights: {opt.chkpt_path} | Prediction output: {opt.pred_dir}")
    print(f"Test set path: {opt.test_dir}")

    # Initialize model and load weights
    net = Net(num_classes=num_class).cuda()
    net.load_state_dict(torch.load(opt.chkpt_path, map_location='cuda'))
    net.eval()
    print("===== Model weights loaded successfully =====")

    # Load test dataset
    test_set = RS.Data(opt.test_dir, 'val')
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size, shuffle=False)
    print(f"===== Test set loaded: {len(test_set)} samples in total =====")

    # Initialize metric calculation tool
    tool4metric = ConfuseMatrixMeter(n_class=num_class)
    tool4metric.clear()

    # Start prediction and evaluation
    print("===== Starting prediction and evaluation =====")
    for vi, data in enumerate(tqdm(test_loader, desc='Predicting & Evaluating')):
        imgs_A, imgs_B, labels_A, labels_B, imgname = data
        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()
        mask_name = imgname[0]

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            change_mask = F.sigmoid(out_change).detach() > 0.5

        # ===================== 关键修改1：分离完整语义和mask后语义 =====================
        # 1. 完整的语义分割结果（不经过change mask过滤）
        preds_A_full = torch.argmax(outputs_A, dim=1)  # 完整语义
        preds_B_full = torch.argmax(outputs_B, dim=1)  # 完整语义

        # 2. 仅变化区域的语义结果（原逻辑，保留mask过滤）
        preds_A_masked = (preds_A_full * change_mask.squeeze().long())  # mask后语义
        preds_B_masked = (preds_B_full * change_mask.squeeze().long())  # mask后语义
        # ============================================================================

        # Update confusion matrix（评估仍用mask后的结果，保持原逻辑）
        pred_all = torch.cat([preds_A_masked, preds_B_masked], dim=0)
        label_all = torch.cat([labels_A, labels_B], dim=0)
        tool4metric.update_cm(pr=pred_all.cpu().numpy(), gt=label_all.cpu().numpy())

        # 转换为numpy数组
        change_mask_np = change_mask.cpu().squeeze().numpy().astype(np.uint8)
        preds_A_full_np = preds_A_full.cpu().squeeze().numpy().astype(np.uint8)  # 完整语义numpy
        preds_B_full_np = preds_B_full.cpu().squeeze().numpy().astype(np.uint8)  # 完整语义numpy
        preds_A_masked_np = preds_A_masked.cpu().squeeze().numpy().astype(np.uint8)  # mask后语义numpy
        preds_B_masked_np = preds_B_masked.cpu().squeeze().numpy().astype(np.uint8)  # mask后语义numpy

        # Save index maps（mask后的索引图，保留原逻辑）
        pred_A_idx = Image.fromarray(preds_A_masked_np)
        pred_B_idx = Image.fromarray(preds_B_masked_np)
        pred_A_idx.save(os.path.join(opt.im1_dir, mask_name))
        pred_B_idx.save(os.path.join(opt.im2_dir, mask_name))

        # ===================== 关键修改2：分别保存完整语义和mask后语义 =====================
        # 1. 保存完整语义分割可视化结果（im1_semantic/im2_semantic）
        io.imsave(os.path.join(opt.im1_semantic_dir, mask_name), index2color(preds_A_full_np, ST_COLORMAP))
        io.imsave(os.path.join(opt.im2_semantic_dir, mask_name), index2color(preds_B_full_np, ST_COLORMAP))

        # 2. 保存仅变化区域的语义可视化结果（im1_rgb/im2_rgb，原逻辑）
        im1_rgb = index2color(preds_A_masked_np, ST_COLORMAP)
        im2_rgb = index2color(preds_B_masked_np, ST_COLORMAP)
        io.imsave(os.path.join(opt.im1_rgb_dir, mask_name), im1_rgb)
        io.imsave(os.path.join(opt.im2_rgb_dir, mask_name), im2_rgb)
        # ============================================================================

        # Save change mask（保留原逻辑）
        change_map_ = exposure.rescale_intensity(~change_mask_np, out_range='uint8')
        io.imsave(os.path.join(opt.change_dir, mask_name), change_map_)

    # Calculate evaluation metrics
    scores_dictionary = tool4metric.get_scores()
    print("\n===== Test metrics =====")
    print(f'acc = {scores_dictionary["acc"]:.4f}')
    print(f'mIoU = {scores_dictionary["mIoU"]:.4f}')
    print(f'Sek = {scores_dictionary["Sek"]:.4f}')
    print(f'Fscd = {scores_dictionary["Fscd"]:.4f}')

    # Save evaluation results
    metric_file = os.path.join(opt.eval_dir, 'eval_metrics.txt')
    with open(metric_file, 'w', encoding='utf-8') as f:
        f.write(f"Model weights: {opt.chkpt_path}\n")
        f.write(f"Test set path: {opt.test_dir}\n")
        f.write(f"Prediction path: {opt.pred_dir}\n")
        f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Flip TTA enabled: {opt.flip_tta}\n")
        f.write(f"Number of classes: {num_class}\n")
        f.write("-" * 60 + "\n")
        f.write(f'acc = {scores_dictionary["acc"]:.4f}\n')
        f.write(f'mIoU = {scores_dictionary["mIoU"]:.4f}\n')
        f.write(f'Sek = {scores_dictionary["Sek"]:.4f}\n')
        f.write(f'Fscd = {scores_dictionary["Fscd"]:.4f}\n')

    # Calculate total time
    total_time = time.time() - begin_time
    print(f"\n===== All completed! Total time: {total_time:.2f}s =====")
    print(f"Prediction maps saved to: {opt.pred_dir}")
    print(f"Evaluation results saved to: {metric_file}")


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    main()