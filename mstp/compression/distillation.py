from utils.utils import *
from utils.dataset import DVS_Lip
from utils.test import test
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from utils.net import Net


def train_with_distillation(args, device, student_net: Net, teacher_net: Net, d=0.02):
    dataset = DVS_Lip("train", args)
    student_net.logger.info(f"Start Training with Distillation, Data Length: {len(dataset)}")
    
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    tot_iter = 0
    best_acc, best_acc_p1, best_acc_p2 = 0.0, 0.0, 0.0
    best_epoch = 0
    scaler = GradScaler(device.type)
    teacher_net.net.eval()

    with torch.no_grad():
        acc_teacher, _, _, _ = test(args, device, teacher_net)
    acc_stop_threshold = acc_teacher - 0.02  # critério de parada: 2% ou 5% abaixo
    student_net.logger.info(f"Acurácia professor: {acc_teacher:.4f} | Limite mínimo aluno: {acc_stop_threshold:.4f}")

    for epoch in range(args.max_epoch):
        for (i_iter, input) in enumerate(loader):
            tic = time.time()
            student_net.net.train()
            event_low = input.get("event_low").to(device, non_blocking=True)
            event_high = input.get("event_high").to(device, non_blocking=True)
            label = input.get("label").to(device, non_blocking=True).long()

            student_net.optimizer.zero_grad()
            with torch.no_grad():
                with autocast(device.type):
                    teacher_logits = teacher_net.net(event_low, event_high)

            with autocast(device.type):
                student_logits = student_net.net(event_low, event_high)
                loss = distillation_loss(student_logits, teacher_logits, label, T=2.0, alpha=0.5)

            scaler.scale(loss).backward()
            scaler.step(student_net.optimizer)
            scaler.update()

            toc = time.time()
            if i_iter % 20 == 0:
                msg = f"epoch={epoch}, train_iter={tot_iter}, eta={(toc-tic)*(len(loader)-i_iter)/3600.0:.5f}, loss={loss.item():.5f}"
                student_net.logger.info(msg)

            student_net.writer.add_scalar("loss", loss.item(), tot_iter)

            if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):
                acc, acc_p1, acc_p2, msg = test(args, device, student_net)
                student_net.logger.info(msg)
                student_net.writer.add_scalar("test_acc", acc, tot_iter)
                student_net.writer.add_scalar("test_acc/part1", acc_p1, tot_iter)
                student_net.writer.add_scalar("test_acc/part2", acc_p2, tot_iter)

                if acc > best_acc:
                    best_acc, best_acc_p1, best_acc_p2, best_epoch = acc, acc_p1, acc_p2, epoch
                    savename = student_net.log_dir + "/model_best.pth"
                    os.makedirs(os.path.split(savename)[0], exist_ok=True)
                    torch.save(student_net.net.module.state_dict(), savename)
                
                if acc >= acc_stop_threshold:
                    student_net.logger.info(f"Criterio de parada atendido: {acc:.4f} >= {acc_stop_threshold:.4f}")
                    return student_net

            tot_iter += 1

        student_net.scheduler.step()

    student_net.logger.info(f"best_acc={best_acc:.5f}, part1={best_acc_p1:.5f}, part2={best_acc_p2:.5f}, epoch={best_epoch}")

    return student_net


def distillation_loss(student_logits, teacher_logits, true_labels, T=2.0, alpha=0.5):
    """Compute knowledge distillation loss"""
    # Distilling the Knowledge in a Neural Network (2015) - Geoffrey Hinton
    log_student = nn.functional.log_softmax(student_logits / T, dim=1)
    soft_teacher = nn.functional.softmax(teacher_logits / T, dim=1)
    loss_kd = nn.functional.kl_div(log_student, soft_teacher, reduction="batchmean") * (T * T)
    loss_ce = nn.functional.cross_entropy(student_logits, true_labels)
    return alpha * loss_ce + (1 - alpha) * loss_kd
