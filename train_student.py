import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import sys

from cifar import build_dataset
from model import get_model
from loss_fn import DiVELoss, KDLoss, BKDLoss, WSLLoss

def parse_option():
    parser = argparse.ArgumentParser('Distillation')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    parser.add_argument('--imb_factor', type=int, default=100, choices=[1, 10, 50, 100])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr_decay_epochs', type=str, default='80,100')

    parser.add_argument('--teacher_arch', type=str, default='resnet32x4')
    parser.add_argument('--student_arch', type=str, default='resnet8x4')
    parser.add_argument('--teacher_ckpt_path', type=str)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=2e-4)

    parser.add_argument('--temperature', '--temp', type=float, default=3.0)

    parser.add_argument('--kd_type', type=str, default='kd', choices=['kd', 'ce', 'dive', 'bkd', 'wsl'])
    parser.add_argument('--alpha', type=float, default=0.5)

    # --- Options-for-DiVE ---
    parser.add_argument('--power', action='store_true',
                        help='use-power-normalization (p=0.5) for teacher probs')
    
    # --- Utilities ---
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=50)
    args = parser.parse_args()
    
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = [int(i) for i in iterations]

    args.n_classes = 10 if args.dataset == 'cifar10' else 100
    
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    if args.kd_type == 'dive':
        if epoch < 5:
            lr = args.learning_rate * (epoch + 1) / 5.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            return

    for e in args.lr_decay_epoch:
        if epoch == e:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_rate

def accuracy(output, target, topk=(1,)):
    """Tính-độ-chính-xác-top-k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(epoch, train_loader, student_model, teacher_model, criterion, optimizer, args):
    student_model.train()
    teacher_model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        with torch.no_grad():
            teacher_logits = teacher_model(images)
            
        student_logits = student_model(images)
        
        loss = criterion(student_logits, teacher_logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc1, _ = accuracy(student_logits, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        
        if (idx + 1) % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{idx + 1}/{len(train_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

    return losses.avg, top1.avg

def validate(val_loader, model, criterion, args):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
    print(f' * Val-Acc@1 {top1.avg:.3f} Val-Acc@5 {top5.avg:.3f} Val-Loss {losses.avg:.4f}')
    return losses.avg, top1.avg

def main():
    args = parse_option()
    setup_seed(args.seed)

    train_loader, valid_loader, test_loader, img_num_per_cls = build_dataset(args=args, get_img_num_per_cls=True)

    teacher_model = get_model(args.teacher_arch, num_classes=args.n_classes).cuda()
    student_model = get_model(args.student_arch, num_classes=args.n_classes).cuda()
    cudnn.benchmark = True
    
    ckpt = torch.load(args.teacher_ckpt_path)
    teacher_model.load_state_dict(ckpt['net'])
    teacher_model.eval()
    print(f"Acc@1 Teacher: {ckpt['acc@1']:.3f}")
    
    if args.kd_type == 'kd':
        criterion = KDLoss(args).cuda()
    elif args.kd_type == 'ce':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.kd_type == 'bkd':
        "Copy-from: https://github.com/EricZsy/BalancedKnowledgeDistillation/blob/main/train_student.py"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, img_num_per_cls)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_per_cls)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)    
        criterion = BKDLoss(cls_num_list=img_num_per_cls, args=args, weight=per_cls_weights).cuda()
    elif args.kd_type == 'dive':
        criterion = DiVELoss(img_num_per_cls,
                             alpha=args.alpha,
                             temperature=args.temperature,
                             power_norm=args.power).cuda()
    elif args.kd_type == 'wsl':
        criterion = WSLLoss(args).cuda()
    else:
        raise ValueError(f"Not valid kd_type: {args.kd_type}")
    
    val_criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = optim.SGD(student_model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)        

        train_loss, train_acc = train(epoch, train_loader, student_model, teacher_model, criterion, optimizer, args)
        val_loss, val_acc = validate(valid_loader, student_model, val_criterion, args)
                
        if val_acc > best_acc:
            best_acc = val_acc
            student_ckpt_path = f'student_{args.dataset}_imb{args.imb_factor}.pth'
            state = {
                'epoch': epoch,
                'model': student_model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, student_ckpt_path)
            
    print(f"Best acc: {best_acc:.3f}")

if __name__ == '__main__':
    main()