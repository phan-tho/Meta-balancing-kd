import argparse
import torch
import torch.nn.functional as F
import torch.optim

import model.resnet as resnet
import model.metaresnet as metaresnet

from cifar import build_dataset

parser = argparse.ArgumentParser(description='Meta-Balancing KD Training')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset (cifar10/cifar100)')
parser.add_argument('--num_valid', type=int, default=1000)   
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--lr_decay_epoch', default=[80, 100], nargs='+', type=int,
                    help='epochs to decay learning rate')
parser.add_argument('--temperature', default=4, type=float, help='temperature for softmax')
parser.add_argument('--print_freq', default=150, type=int)
parser.add_argument('--prefetch', type=int, default=0)

parser.add_argument('--teacher_ckpt', default='teacher_cifar100.pth', type=str)

parser.add_argument('--l_meta', default='hard', help='mix/hard/soft')
parser.add_argument('--imb_factor', default=1, type=float, help='imbalance factor, larger means more imbalance')
parser.add_argument('--hidden_wnet', default=[64, 64], nargs='+', type=int,
                    help='hidden layers for wnet')
parser.add_argument('--inner_accum_steps', default=1, type=int, help='accumulation steps for inner update of inner model')
parser.set_defaults(augment=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.seed = 12
args.n_classes = 10 if args.dataset == 'cifar10' else 100
torch.manual_seed(args.seed)

def load_teacher(args):
    model = resnet.resnet32x4(num_classes=10 if args.dataset == 'cifar10' else 100)
    ckt = torch.load(args.teacher_ckpt, map_location=args.device)
    model.load_state_dict(ckt['net']).to(args.device)
    print('load teacher acc', ckt['acc@1'])
    return model

def kd_loss_fn(student_logits, teacher_logits, target, args):
    hard_loss = F.cross_entropy(student_logits, target, reduction='none')
    log_student = F.log_softmax(student_logits / args.temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / args.temperature, dim=1)
    soft_loss = F.kl_div(log_student, soft_teacher, reduction='none').sum(1) * (args.temperature * args.temperature)
    # hard_loss shape: [batch_size]
    return hard_loss, soft_loss

class WNetWrapper():
    def __init__(self, wnet, optimizer_wnet, args):
        self.wnet = wnet
        self.optimizer_wnet = optimizer_wnet
        self.args = args

    def __call__(self, outputs_student, outputs_teacher, targets, no_grad=False):
        hard_loss, soft_loss = kd_loss_fn(outputs_student, outputs_teacher, targets, self.args)

        ce_teacher = torch.functional.F.cross_entropy(outputs_teacher, targets, reduction='none') # shape [batch]
        ce = torch.stack([hard_loss, ce_teacher], dim=1) # shape [batch, 2]
        weights = self.wnet(ce.data)

        if no_grad:
            weights = weights.detach()

        return weights, hard_loss, soft_loss
    
    def zero_grad(self):
        self.optimizer_wnet.zero_grad()
    def step(self):
        self.optimizer_wnet.step()

def train(train_loader, valid_loader, model, teacher, wnet_wrapper, optimizer_model, epoch, args):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    meta_loss = 0
    valid_loader_iter = iter(valid_loader)

    total_prec_train = 0
    total_prec_meta = 0

    inner_step_counter = 0
    wnet_wrapper.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        with torch.no_grad():
            outputs_teacher = teacher(inputs)

        # Step 1: Meta-model (for inner-update)
        meta_model = metaresnet.meta_resnet8x4(num_classes=args.n_classes).to(args.device)
        meta_model.load_state_dict(model.state_dict())
        meta_model.to(args.device)

        outputs_student = meta_model(inputs)

        weights, hard_loss, soft_loss = wnet_wrapper(outputs_student, outputs_teacher, targets)

        w_hard = weights[:, 0:1]  # (B,1)
        w_soft = weights[:, 1:2]  # (B,1)
        l_f_meta = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / w_hard.size(0)

        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_model.update_params(lr_inner=args.lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(valid_loader_iter)
        except StopIteration:
            valid_loader_iter = iter(valid_loader)
            inputs_val, targets_val = next(valid_loader_iter)

        inputs_val, targets_val = inputs_val.to(args.device), targets_val.to(args.device)
        with torch.no_grad():
            outputs_teacher_val = teacher(inputs_val)
        outputs_val_student = meta_model(inputs_val)

        hard_loss, soft_loss = kd_loss_fn(outputs_val_student, outputs_teacher_val, targets_val, args)
        if args.l_meta == 'mix':
            l_g_meta = torch.mean(hard_loss + soft_loss)
        elif args.l_meta == 'hard':
            l_g_meta = torch.mean(hard_loss)
        else:
            l_g_meta = torch.mean(soft_loss)

        inner_step_counter += 1
        (l_g_meta / args.inner_accum_steps).backward()       

        if (inner_step_counter % args.inner_accum_steps) == 0:
            wnet_wrapper.step()
            wnet_wrapper.zero_grad()

        outputs_student = model(inputs)
        weights, hard_loss, soft_loss = wnet_wrapper(outputs_student, outputs_teacher, targets, no_grad=True)
        w_hard = weights[:, 0:1]
        w_soft = weights[:, 1:2]
        loss = torch.sum(w_hard * hard_loss.unsqueeze(1) + w_soft * soft_loss.unsqueeze(1)) / w_hard.size(0)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        prec_meta = accuracy(outputs_val_student.data, targets_val.data, topk=(1,))[0]
        prec_train = accuracy(outputs_student.data, targets.data, topk=(1,))[0]

        train_loss += loss.item()
        meta_loss += l_g_meta.item()
        total_prec_train += prec_train.item()
        total_prec_meta += prec_meta.item()

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                    (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size,
                    train_loss / (batch_idx + 1), meta_loss / (batch_idx + 1), prec_train, prec_meta))

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, optimizer_wnet, args):
    for e in args.lr_decay_epoch:
        if epoch == e:
            args.lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            for param_group in optimizer_wnet.param_groups:
                param_group['lr'] *= 0.8

def test(model, test_loader, epoch, args):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def main():
    train_loader, valid_loader, test_loader = build_dataset(args=args)
    model = metaresnet.meta_resnet8x4(num_classes=args.n_classes).to(device)

    teacher = load_teacher(args)
    wnet = metaresnet.WNet(2, args.hidden_wnet, 2).to(device)

    optimizer_model = torch.optim.SGD(model.params(), args.lr,
                                      momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_wnet = torch.optim.Adam(wnet.params(), 1e-3, weight_decay=1e-4)
    wnet_wrapper = WNetWrapper(wnet, optimizer_wnet, args)

    best_acc = 0
    at_e = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_model, epoch, optimizer_wnet, args)
        train(train_loader, valid_loader, model, teacher, wnet_wrapper, optimizer_model, epoch, args)
        test_acc = test(model, test_loader, epoch, args)

        if test_acc > best_acc:
            best_acc = test_acc
            at_e = epoch
            ckpt = {
                'student': model.state_dict(),
                'wnet': wnet.state_dict(),
                'acc@1': best_acc,
                'epoch': epoch + 1,
                'config': vars(args)
            }
            torch.save(ckpt, f'8x4_{args.dataset}_imb{args.imb_factor}.pth')

    print(f'Best acc: {best_acc} at epoch {at_e + 1}')

if __name__ == '__main__':
    main()
