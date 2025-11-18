import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# criterion(student_logits, teacher_logits, labels)

class BSCELoss(nn.Module):
    """
    Balanced Softmax Cross-Entropy Loss (BSCE).
    """
    def __init__(self, img_num_per_cls):
        super(BSCELoss, self).__init__()
        img_num_per_cls = torch.tensor(img_num_per_cls, dtype=torch.float32)
        self.img_num_per_cls = img_num_per_cls
        self.log_priors = torch.log(img_num_per_cls / img_num_per_cls.sum())

    def forward(self, x_logits, target):
        log_priors_device = self.log_priors.to(x_logits.device)
        x_logits_adjusted = x_logits + log_priors_device.unsqueeze(0)
        return F.cross_entropy(x_logits_adjusted, target)

class DiVELoss(nn.Module):
    """
        L_DiVE = (1 - alpha) * L_BSCE(y, s_BSCE) + alpha * tau^2 * L_KL(t_tau, s_tau)
        From: https://arxiv.org/pdf/2103.15042
    """
    def __init__(self, img_num_per_cls, alpha, temperature, power_norm=False):
        super(DiVELoss, self).__init__()
        self.alpha = alpha
        self.temp = temperature
        self.power_norm = power_norm
        
        self.bsce_loss = BSCELoss(img_num_per_cls)
        self.kl_div = nn.KLDivLoss(reduction='batchmean').cuda()

    def forward(self, student_logits, teacher_logits, target):
        loss_bsce = self.bsce_loss(student_logits, target)
        student_logits_soft = student_logits / self.temp
        teacher_logits_soft = teacher_logits / self.temp
        
        p_s = F.log_softmax(student_logits_soft, dim=1)
        if self.power_norm:
            p_t = F.softmax(teacher_logits_soft, dim=1)
            p_t = torch.pow(p_t, 0.5)
            p_t = p_t / p_t.sum(1, keepdim=True)
        else:
            p_t = F.softmax(teacher_logits_soft, dim=1)
            
        loss_kd = self.kl_div(p_s, p_t.detach()) 
        loss_kd = loss_kd * (self.alpha * (self.temp ** 2))

        loss_bsce_weighted = (1.0 - self.alpha) * loss_bsce
        total_loss = loss_bsce_weighted + loss_kd
        return total_loss
    
class KDLoss(nn.Module):
    """
    L = alpha * L_ce + (1 - alpha) * tau^2 * L_KL
    """
    def __init__(self, args):
        super(KDLoss, self).__init__()
        self.temp = args.temperature
        self.alpha = args.alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, target):
        loss_ce = self.ce_loss(student_logits, target)
        student_logits_soft = student_logits / self.temp
        teacher_logits_soft = teacher_logits / self.temp
        
        p_s = F.log_softmax(student_logits_soft, dim=1)
        p_t = F.softmax(teacher_logits_soft, dim=1)
            
        loss_kd = self.kl_div(p_s, p_t.detach()) 
        loss_kd = loss_kd * (self.temp ** 2)

        total_loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
        return total_loss
    

class BKDLoss(nn.Module):
    "Copy from: https://github.com/EricZsy/BalancedKnowledgeDistillation/blob/main/losses.py"
    def __init__(self, cls_num_list, args, weight=None):
        super(BKDLoss, self).__init__()
        self.args = args
        self.T = args.temperature
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def forward(self, student_logits, teacher_logits, target):
        pred_t = F.softmax(teacher_logits/self.T, dim=1)
        if self.weight is not None:
            pred_t = pred_t * self.weight
            pred_t = pred_t / pred_t.sum(1)[:, None]
        kd = F.kl_div(F.log_softmax(student_logits/self.T, dim=1),
                        pred_t,
                        reduction='none').mean(dim=0)
        kd_loss = kd.sum() * self.T * self.T
        ce_loss = self.CELoss(student_logits, target)
        loss = self.args.alpha * kd_loss + ce_loss

        return loss, kd
    

class WSLLoss(nn.Module):
    """
    Weighted Soft Label Loss (WSL).
    Adapted from: https://github.com/bellymonster/Weighted-Soft-Label-Distillation/blob/master/knowledge_distiller.py
    """
    def __init__(self, args):
        super(WSLLoss, self).__init__()
        self.args = args
        self.T = args.temperature
        self.alpha = args.alpha

        # self.T = 4
        # self.alpha = 2.5

        self.hard_loss = nn.CrossEntropyLoss().cuda()
        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, student_logits, teacher_logits, target):
        s_input_for_softmax = student_logits / self.T
        t_input_for_softmax = teacher_logits / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)

        student_logits_auto = student_logits.detach()
        teacher_logits_auto = teacher_logits.detach()
        log_softmax_s = self.logsoftmax(student_logits_auto)
        log_softmax_t = self.logsoftmax(teacher_logits_auto)

        one_hot_label = F.one_hot(target, num_classes=self.args.num_classes).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)
        hard_loss = self.hard_loss(student_logits, target)

        loss = hard_loss + self.alpha * soft_loss
        return loss