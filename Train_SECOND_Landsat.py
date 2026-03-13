import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.SCD_misc import ConfuseMatrixMeter, AverageMeter
from utils.loss import CrossEntropyLoss2d, BCDLoss, ChangeSimilarity
from datasets import RS_ST as RS
from datasets.RS_ST import get_dataset_config
from models.HyFuSCD import HyFuSCD as Net


def adjust_lr(optimizer, curr_iter, all_iter, init_lr):
    """
    Adjust learning rate with polynomial decay strategy
    """
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args.lr_decay_power)
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def main(args):
    # Load dataset configuration
    dataset_config = get_dataset_config(args.dataname)
    num_classes = dataset_config["num_classes"]

    # Initialize TensorBoard writer
    writer = SummaryWriter(args.chkpt_dir)

    # Initialize model
    net = Net(num_classes=num_classes).cuda()

    # Calculate total parameters
    parameters_tot = 0
    for _, param in net.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print(f"Number of model parameters: {parameters_tot}\n")

    # Load datasets
    train_set = RS.Data(args.datapath, 'train', augmentation=True)
    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, num_workers=4, shuffle=True)

    val_set = RS.Data(args.datapath, 'val')
    val_loader = DataLoader(val_set, batch_size=args.val_batchsize, num_workers=4, shuffle=False)

    # Initialize optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True
    )

    # Start training
    train(train_loader, val_loader, net, optimizer, writer, num_classes)
    writer.close()
    print('Training finished.')


def train(train_loader, val_loader, net, optimizer, writer, num_classes):
    """
    Main training loop with training and validation phases
    """
    # Initialize metric calculator
    tool4metric = ConfuseMatrixMeter(n_class=num_classes)
    best_sek = 0.0

    # Initialize loss functions
    criterion_seg = CrossEntropyLoss2d(ignore_index=0).cuda()
    criterion_sc = ChangeSimilarity().cuda()
    criterion_bn = BCDLoss().cuda()

    def training_phase(epoch):
        """Training phase for a single epoch"""
        torch.cuda.empty_cache()
        net.train()

        # Initialize loss meters
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()
        train_all_loss = AverageMeter()

        total_iters = float(len(train_loader) * args.epoch)
        curr_iter = epoch * len(train_loader)
        step = 0

        # Training loop
        loop = tqdm(train_loader, file=sys.stdout)
        for imgs_A, imgs_B, labels_A, labels_B, _ in loop:
            loop.set_description(f'Epoch:{epoch}')

            # Update learning rate
            running_iter = curr_iter + step + 1
            adjust_lr(optimizer, running_iter, total_iters, args.lr)
            step += 1

            # Move data to GPU
            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_bn = (labels_A > 0).cuda().long()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()

            # Forward pass
            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

            # Calculate losses
            loss_seg = criterion_seg(outputs_A, labels_A) + criterion_seg(outputs_B, labels_B)
            loss_bn = criterion_bn(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
            loss = loss_seg * 0.5 + loss_bn * 0.5 + loss_sc

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update loss meters
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())
            train_all_loss.update(loss.cpu().detach().numpy())

            # Update progress bar
            loop.set_postfix(loss=train_all_loss.val, lr=optimizer.param_groups[0]['lr'])

        # Print epoch loss
        print(
            f'LOSS {train_all_loss.val:.2f}: [seg {train_seg_loss.val:.4f} bn {train_bn_loss.val:.4f} sc {train_sc_loss.val:.4f}]')

        # Log losses to TensorBoard
        writer.add_scalar('train/seg_loss', train_seg_loss.val, epoch)
        writer.add_scalar('train/bn_loss', train_bn_loss.val, epoch)
        writer.add_scalar('train/total_loss', train_all_loss.val, epoch)

        # Clean up
        torch.cuda.empty_cache()
        del loss, outputs_A, outputs_B, out_change, loss_seg, loss_bn, loss_sc

    def validation_phase(epoch):
        """Validation phase for a single epoch"""
        tool4metric.clear()
        net.eval()
        torch.cuda.empty_cache()

        val_loss = AverageMeter()

        with torch.no_grad():
            loop = tqdm(val_loader, file=sys.stdout)
            for imgs_A, imgs_B, labels_A, labels_B, _ in loop:
                loop.set_description(f'Epoch:{epoch}')

                # Move data to GPU
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                # Forward pass
                out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

                # Calculate validation loss
                loss_A = nn.CrossEntropyLoss(ignore_index=0)(outputs_A, labels_A)
                loss_B = nn.CrossEntropyLoss(ignore_index=0)(outputs_B, labels_B)
                loss = loss_A * 0.5 + loss_B * 0.5
                val_loss.update(loss.cpu().detach().numpy())

                # Generate predictions
                change_mask = F.sigmoid(out_change).detach() > 0.5
                preds_A = torch.argmax(outputs_A, dim=1)
                preds_B = torch.argmax(outputs_B, dim=1)
                preds_A = (preds_A * change_mask.squeeze().long())
                preds_B = (preds_B * change_mask.squeeze().long())

                # Update confusion matrix
                pred_all = torch.cat([preds_A, preds_B], dim=0)
                label_all = torch.cat([labels_A, labels_B], dim=0)
                tool4metric.update_cm(pr=pred_all.cpu().numpy(), gt=label_all.cpu().numpy())

        # Calculate evaluation metrics
        scores = tool4metric.get_scores()

        # Print validation metrics
        print(f'Validation - acc = {scores["acc"] * 100:.4f}, mIoU = {scores["mIoU"] * 100:.4f}, Sek = {scores["Sek"] * 100:.4f}, Fscd = {scores["Fscd"] * 100:.4f}')

        # Log metrics to TensorBoard
        writer.add_scalar('val/loss', val_loss.average(), epoch)
        writer.add_scalar('val/mIoU', scores['mIoU'], epoch)

        # Clean up
        torch.cuda.empty_cache()
        del loss, outputs_A, outputs_B, out_change, preds_A, preds_B, loss_A, loss_B, pred_all, label_all

        return scores

    # Main epoch loop
    for epoch in range(args.epoch):
        training_phase(epoch)
        scores = validation_phase(epoch)

        # Save best model based on Sek score
        current_sek = scores['Sek']
        if current_sek > best_sek:
            best_sek = current_sek
            ckpt_filename = f"E{epoch}_iou{scores['mIoU'] * 100:.2f}_Sek{current_sek * 100:.2f}_acc{scores['acc'] * 100:.2f}_Fscd{scores['Fscd'] * 100:.2f}.pth"
            ckpt_path = os.path.join(args.chkpt_dir, ckpt_filename)
            torch.save(net.state_dict(), ckpt_path)


if __name__ == '__main__':
    # Get working directory
    working_path = os.path.dirname(os.path.abspath(__file__))

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Training script for HyFuSCD model on remote sensing change detection tasks")
    parser.add_argument("--dataname", default="SECOND", type=str, help="Dataset name: SECOND, Landsat")
    parser.add_argument("--datapath", default="/root/***/SECOND", type=str, help="Path to dataset directory")

    parser.add_argument("--modelname", default="HyFuSCD", type=str, help="Model name")
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate')
    parser.add_argument('--lr_decay_power', type=float, default=1.5,
                        help='Learning rate decay power for polynomial decay')
    parser.add_argument('--epoch', type=int, default=60, help='Maximum training epochs (SECOND:60, Landsat:100)')
    parser.add_argument('--train_batchsize', type=int, default=8, help='Training batch size')
    parser.add_argument('--val_batchsize', type=int, default=8, help='Validation batch size')
    parser.add_argument('--alpha', type=float, default=1.0, help='Loss weight coefficient')

    args = parser.parse_args()

    # Create checkpoint and result directories
    chkpt_dir = os.path.join(working_path, 'checkpoints', args.dataname, args.modelname)
    pred_dir = os.path.join(working_path, 'results', args.dataname)

    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # Create run directory
    run_dirs = sorted([f for f in os.listdir(chkpt_dir) if f.startswith("run_")])
    num_run = int(run_dirs[-1].split("_")[-1]) + 1 if run_dirs else 0
    args.chkpt_dir = os.path.join(chkpt_dir, f"run_{num_run:04d}/")
    args.pred_dir = pred_dir

    # Start training
    main(args)