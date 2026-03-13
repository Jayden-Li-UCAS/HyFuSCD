import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import PIL.Image as Image

from datasets.change_detection import ChangeDetection, get_num_classes
from models.HyFuSCD import HyFuSCD
from models.HyFuSCD_Mutimodel import HyFuSCD_Mutimodel
from utils.EGMS.palette import get_color_map
from utils.EGMS.metric import IOUandSek

from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import argparse
import copy


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')

        parser.add_argument("--data_name", type=str, default=r"OpenMapCD", help="SCSCD7, OpenMapCD")
        parser.add_argument("--data_root", type=str, default=r"/root/***/OpenMapCD/")
        parser.add_argument("--load_from", type=str, default=r"/root/***.pth")
        parser.add_argument("--test_batch_size", type=int, default=1)
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args


def inference(args):
    working_path = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(working_path, 'pred_results')
    pred_save_path1 = os.path.join(pred_dir, 'pred1')  # mask后语义索引图
    pred_save_path2 = os.path.join(pred_dir, 'pred2')  # mask后语义索引图
    pred_save_path1_rgb = os.path.join(pred_dir, 'pred1_rgb')  # mask后语义可视化（变化区域）
    pred_save_path2_rgb = os.path.join(pred_dir, 'pred2_rgb')  # mask后语义可视化（变化区域）
    pred_save_path1_semantic = os.path.join(pred_dir, 'pred1_semantic')  # 完整语义可视化
    pred_save_path2_semantic = os.path.join(pred_dir, 'pred2_semantic')  # 完整语义可视化
    pred_save_path3 = os.path.join(pred_dir, 'pred_change')  # 变化mask图

    # 创建输出目录
    for path in [pred_save_path1, pred_save_path2, pred_save_path1_rgb, pred_save_path2_rgb,
                 pred_save_path1_semantic, pred_save_path2_semantic, pred_save_path3]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 加载数据集和模型
    testset = ChangeDetection(root=args.data_root, mode="val")
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                            pin_memory=True, num_workers=0, drop_last=False)
    if args.data_name == "OpenMapCD":
        Net = HyFuSCD_Mutimodel
    else:
        Net = HyFuSCD
    model = Net(num_classes=get_num_classes(args.data_name) - 1)

    if args.load_from:
        model.load_state_dict(torch.load(args.load_from), strict=True)

    model = model.cuda()
    model.eval()

    tbar = tqdm(testloader)
    metric = IOUandSek(num_classes=get_num_classes(args.data_name))
    begin_time = time.time()

    with torch.no_grad():
        for img1, img2, label1, label2, _, id in tbar:
            img1, img2 = img1.cuda(), img2.cuda()

            # 模型推理
            out_bn, out1, out2 = model(img1, img2)  # [b,6,512,512],[b,6,512,512],[b,512,512]
            out_bn = torch.sigmoid(out_bn)
            out_bn = out_bn.squeeze(1)

            # ===================== 核心修改1：分离完整语义和mask后语义 =====================
            # 1. 完整的语义分割结果（无mask过滤，+1是原代码的类别偏移）
            pred1_seg_full = torch.argmax(out1, dim=1).cpu().numpy() + 1  # 完整语义
            pred2_seg_full = torch.argmax(out2, dim=1).cpu().numpy() + 1  # 完整语义

            # 2. 变化mask（阈值0.5）
            change_mask = ((out_bn > 0.5).cpu().numpy()).astype(np.uint8)

            # 3. 仅变化区域的语义结果（mask过滤）
            pred1_seg_masked = copy.deepcopy(pred1_seg_full)
            pred2_seg_masked = copy.deepcopy(pred2_seg_full)
            pred1_seg_masked[change_mask == 0] = 0  # 非变化区域置0
            pred2_seg_masked[change_mask == 0] = 0  # 非变化区域置0
            # ============================================================================

            # 获取颜色映射表
            cmap = get_color_map(args.data_name)

            # 逐样本保存结果
            for i in range(pred1_seg_full.shape[0]):
                # 1. 保存mask后的语义索引图（pred1/pred2，原逻辑）
                mask1_idx = Image.fromarray(pred1_seg_masked[i].astype(np.uint8))
                mask1_idx.save(os.path.join(pred_save_path1, id[i]))
                mask2_idx = Image.fromarray(pred2_seg_masked[i].astype(np.uint8))
                mask2_idx.save(os.path.join(pred_save_path2, id[i]))

                # 2. 保存mask后的语义可视化图（pred1_rgb/pred2_rgb，变化区域）
                mask1_rgb = Image.fromarray(pred1_seg_masked[i].astype(np.uint8)).convert('P')
                mask1_rgb.putpalette(cmap)
                mask1_rgb.save(os.path.join(pred_save_path1_rgb, id[i]))
                mask2_rgb = Image.fromarray(pred2_seg_masked[i].astype(np.uint8)).convert('P')
                mask2_rgb.putpalette(cmap)
                mask2_rgb.save(os.path.join(pred_save_path2_rgb, id[i]))

                # 3. 保存完整语义可视化图（pred1_semantic/pred2_semantic，所有区域）
                mask1_sem = Image.fromarray(pred1_seg_full[i].astype(np.uint8)).convert('P')
                mask1_sem.putpalette(cmap)
                mask1_sem.save(os.path.join(pred_save_path1_semantic, id[i]))
                mask2_sem = Image.fromarray(pred2_seg_full[i].astype(np.uint8)).convert('P')
                mask2_sem.putpalette(cmap)
                mask2_sem.save(os.path.join(pred_save_path2_semantic, id[i]))

                # 4. 保存变化mask图（pred_change）
                mask_change = Image.fromarray(change_mask[i] * 255)
                mask_change.save(os.path.join(pred_save_path3, id[i]))

            # 更新评估指标（仍用mask后的结果，保持原逻辑）
            metric.add_batch(pred1_seg_masked, label1.numpy())
            metric.add_batch(pred2_seg_masked, label2.numpy())

        # 计算并打印评估指标
        change_ratio, OA, mIoU, Sek, Fscd, Score, Precision_scd, Recall_scd = metric.evaluate_inference()

        print('==>change_ratio', change_ratio)
        print('==>oa', OA)
        print('==>miou', mIoU)
        print('==>sek', Sek)
        print('==>Fscd', Fscd)
        print('==>score', Score)
        print('==>SC_Precision', Precision_scd)
        print('==>SC_Recall', Recall_scd)

        time_use = time.time() - begin_time
        print(f'==>infer time (s): {round(time_use, 2)}')


if __name__ == "__main__":
    args = Options().parse()
    inference(args)