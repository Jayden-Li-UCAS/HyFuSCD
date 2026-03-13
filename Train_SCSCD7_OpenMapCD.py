from datasets.change_detection import ChangeDetection, get_num_classes
from models.HyFuSCD import HyFuSCD
from models.HyFuSCD_Mutimodel import HyFuSCD_Mutimodel
from utils.EGMS.metric import IOUandSek
from utils.EGMS.loss import ChangeSimilarity, DiceLoss

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import os
import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, AdamW, SGD, lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import random
import numpy as np
import pandas as pd
from datetime import datetime

tqdm_config = {
    "ncols": 100,
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # 简洁格式
    "smoothing": 0.1,
    "dynamic_ncols": False
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_path = os.path.dirname(os.path.abspath(__file__))


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser('Semantic Change Detection')

        ## Training dataset
        parser.add_argument("--data_name", type=str, default=r"OpenMapCD", help="SCSCD7, OpenMapCD")
        parser.add_argument("--data_root", type=str, default=r"/root/autodl-tmp/OpenMapCD/")

        ## Training parameters
        parser.add_argument("--batch_size", type=int, default=6)
        parser.add_argument("--val_batch_size", type=int, default=8)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--lr", type=float, default=0.0003)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--warmup", dest="warmup", action="store_true", help='warm up')

        ## Resume training from intermediate epoch weights
        parser.add_argument("--model_name", type=str, default="HyFuSCD")
        parser.add_argument("--resume", type=str, default="",
                            help="Path to intermediate weights (e.g., weights of epoch43)")
        parser.add_argument("--start_epoch", type=int, default=0,
                            help="Starting epoch (must match resume weights, e.g., 43)")
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(f"[Config] Data: {args.data_name}, Batch Size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
        return args


class Trainer:
    def __init__(self, args):
        self.args = args
        self.num_classes = get_num_classes(args.data_name)

        self.log_dir = os.path.join(working_path, 'logs', self.args.data_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

        self.init_excel_log()

        trainset = ChangeDetection(root=args.data_root, mode="train")
        valset = ChangeDetection(root=args.data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, num_workers=8, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=8, drop_last=False)

        if args.data_name == "OpenMapCD":
            Net = HyFuSCD_Mutimodel
        else:
            Net = HyFuSCD
        self.model = Net(num_classes=self.num_classes-1)

        self.run_dir = self._get_next_run_dir()
        print(f"[Run] Current run folder: {self.run_dir}")

        if args.resume:
            if os.path.isfile(args.resume):
                print(f"[Resume] Loading weights from: {args.resume}")
                checkpoint = torch.load(args.resume)
                self.model.load_state_dict(checkpoint, strict=True)
            else:
                raise FileNotFoundError(f"Weights file not found: {args.resume}")

        self.criterion_seg = CrossEntropyLoss(ignore_index=-1)
        self.criterion_bn = BCELoss(reduction='none')
        self.criterion_bn_2 = DiceLoss()
        self.criterion_sc = ChangeSimilarity()

        self.optimizer = AdamW([
            {"params": [param for name, param in self.model.named_parameters() if "backbone" in name],
             "lr": args.lr},
            {"params": [param for name, param in self.model.named_parameters() if "backbone" not in name],
             "lr": args.lr * 1.0}
        ], lr=args.lr, weight_decay=args.weight_decay)

        self.model = self.model.cuda()
        self.iters = args.start_epoch * len(self.trainloader)
        self.total_iters = len(self.trainloader) * args.epochs
        self.previous_best = 0.0
        self.seg_best = 0.0
        self.change_best = 0.0
        self.epoch_losses = []

    def _get_next_run_dir(self):
        base_dir = os.path.join("checkpoints", self.args.data_name, self.args.model_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
        if not existing_runs:
            next_run_num = 1
        else:
            max_num = max(int(d.split("_")[1]) for d in existing_runs)
            next_run_num = max_num + 1
        run_dir = os.path.join(base_dir, f"run_{next_run_num:04d}")
        os.makedirs(run_dir)
        return run_dir

    def init_excel_log(self):
        self.excel_dir = os.path.join(working_path, 'excel_logs', self.args.data_name)
        if not os.path.exists(self.excel_dir):
            os.makedirs(self.excel_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_path = os.path.join(self.excel_dir, f'training_log_{timestamp}.xlsx')

        self.df_log = pd.DataFrame(columns=[
            'Epoch', 'Total_Loss', 'Loss_Seg', 'Loss_BN', 'Loss_Similarity',
            'Learning_Rate', 'Val_mIoU', 'Val_Sek', 'Val_Score', 'Val_Fscd', 'Val_OA'
        ])
        self.df_log.to_excel(self.excel_path, index=False)
        print(f"[Log] Excel log created: {self.excel_path}")

    def training(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.trainloader, **tqdm_config, desc=f"Train Epoch {curr_epoch}")
        self.model.train()
        total_loss = 0.0
        total_loss_seg = 0.0
        total_loss_bn = 0.0
        total_loss_similarity = 0.0

        curr_iter = curr_epoch * len(self.trainloader)

        for i, (img1, img2, mask1, mask2, mask_bn, id) in enumerate(tbar):
            running_iter = curr_iter + i + 1
            img1, img2 = img1.cuda(), img2.cuda()
            mask1, mask2, mask_bn = mask1.cuda(), mask2.cuda(), mask_bn.cuda()
            out_bn, out1, out2 = self.model(img1, img2)
            out_bn = torch.sigmoid(out_bn)
            out_bn = out_bn.squeeze(1)

            loss1 = self.criterion_seg(out1, mask1 - 1)
            loss2 = self.criterion_seg(out2, mask2 - 1)
            loss_seg = loss1 * 0.5 + loss2 * 0.5
            loss_similarity = self.criterion_sc(out1[:, 0:], out2[:, 0:], mask_bn)
            loss_bn_1 = self.criterion_bn(out_bn, mask_bn)
            loss_bn_1[mask_bn == 1] *= 2
            loss_bn_1 = loss_bn_1.mean()
            loss_bn_2 = self.criterion_bn_2(out_bn, mask_bn)
            loss_bn = loss_bn_1 + loss_bn_2
            loss = loss_bn + loss_seg + loss_similarity

            # 仅保留损失累计，删除实时平均计算
            total_loss_seg += loss_seg.item()
            total_loss_similarity += loss_similarity.item()
            total_loss_bn += loss_bn.item()
            total_loss += loss.item()

            self.iters += 1

            if self.args.warmup:
                warmup_steps = len(self.trainloader) * (self.args.epochs / 5)
                if warmup_steps and self.iters < warmup_steps:
                    warmup_percent_done = self.iters / warmup_steps
                    lr = self.args.lr * warmup_percent_done
                else:
                    lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5
            else:
                lr = self.args.lr * (1. - float(self.iters) / self.total_iters) ** 1.5

            self.optimizer.param_groups[0]["lr"] = lr
            if hasattr(self.args, 'pretrain_from') and self.args.pretrain_from:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0
            else:
                self.optimizer.param_groups[1]["lr"] = lr * 1.0

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 完全删除：实时平均损失计算、进度条postfix、tensorboard实时记录

        # Epoch结束后仅保留汇总计算
        loader_len = len(self.trainloader)
        avg_total_loss = total_loss / loader_len
        avg_loss_seg = total_loss_seg / loader_len
        avg_loss_bn = total_loss_bn / loader_len
        avg_loss_similarity = total_loss_similarity / loader_len
        current_lr = self.optimizer.param_groups[0]["lr"]

        self.current_epoch_info = {
            'Epoch': curr_epoch,
            'Total_Loss': avg_total_loss,
            'Loss_Seg': avg_loss_seg,
            'Loss_BN': avg_loss_bn,
            'Loss_Similarity': avg_loss_similarity,
            'Learning_Rate': current_lr
        }

        # 保留Epoch级汇总打印
        print(
            f"\n[Train] Epoch {curr_epoch} | Avg Loss: {avg_total_loss:.4f} | Seg: {avg_loss_seg:.4f} | BN: {avg_loss_bn:.4f} | Sim: {avg_loss_similarity:.4f}")

        # 可选保留：Epoch级tensorboard记录（按epoch维度）
        self.writer.add_scalar('train/epoch_total_loss', avg_total_loss, curr_epoch)
        self.writer.add_scalar('train/epoch_seg_loss', avg_loss_seg, curr_epoch)
        self.writer.add_scalar('train/epoch_bn_loss', avg_loss_bn, curr_epoch)
        self.writer.add_scalar('train/epoch_sim_loss', avg_loss_similarity, curr_epoch)
        self.writer.add_scalar('train/epoch_lr', current_lr, curr_epoch)

    def validation(self, epoch):
        curr_epoch = epoch
        tbar = tqdm(self.valloader, **tqdm_config, desc=f"Val Epoch {curr_epoch}")
        self.model.eval()
        metric = IOUandSek(num_classes=self.num_classes)

        val_miou = 0.0
        val_sek = 0.0
        val_score = 0.0
        val_fscd = 0.0
        val_oa = 0.0
        val_count = 0

        with torch.no_grad():
            for img1, img2, mask1, mask2, mask_bn, _ in tbar:
                img1, img2 = img1.cuda(), img2.cuda()

                out_bn, out1, out2 = self.model(img1, img2)
                out_bn = torch.sigmoid(out_bn)
                out_bn = out_bn.squeeze(1)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                out_bn = (out_bn > 0.5).cpu().numpy().astype(np.uint8)
                out1[out_bn == 0] = 0
                out2[out_bn == 0] = 0

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                score, miou, sek, Fscd, OA, SC_Precision, SC_Recall = metric.evaluate_SECOND()

                val_score = score
                val_miou = miou
                val_sek = sek
                val_fscd = Fscd
                val_oa = OA
                val_count += 1

        self.current_epoch_info.update({
            'Val_mIoU': val_miou,
            'Val_Sek': val_sek,
            'Val_Score': val_score,
            'Val_Fscd': val_fscd,
            'Val_OA': val_oa
        })

        self.df_log = pd.concat([self.df_log, pd.DataFrame([self.current_epoch_info])], ignore_index=True)
        self.df_log.to_excel(self.excel_path, index=False)

        if score >= self.previous_best:
            weight_filename = "epoch%i_Score%.2f_mIOU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth" % (
                curr_epoch, score * 100, miou * 100, sek * 100, Fscd * 100, OA * 100
            )
            weight_save_path = os.path.join(self.run_dir, weight_filename)

            torch.save(self.model.state_dict(), weight_save_path)
            print(f"[Save] Best model saved to: {weight_save_path}")
            self.previous_best = score

        # 保留Epoch级tensorboard验证指标记录
        self.writer.add_scalar('val_Score', score, curr_epoch)
        self.writer.add_scalar('val_mIOU', miou, curr_epoch)
        self.writer.add_scalar('val_Sek', sek, curr_epoch)
        self.writer.add_scalar('val_Fscd', Fscd, curr_epoch)
        self.writer.add_scalar('val_OA', OA, curr_epoch)


if __name__ == "__main__":
    args = Options().parse()
    trainer = Trainer(args)

    print(f"\n[Start Training] Total Epochs: {args.epochs}, Start from Epoch: {args.start_epoch}")
    print("=" * 80)

    for epoch in range(args.start_epoch, args.epochs):
        print(
            f"\nEpoch {epoch} | LR: {trainer.optimizer.param_groups[0]['lr']:.5f} | Best Score: {trainer.previous_best:.5f}")
        trainer.training(epoch)
        trainer.validation(epoch)