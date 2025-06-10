from utils.utils import *
from utils.dataset import DVS_Lip
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from utils.net import Net
from utils.test import test

def train(args, device, net: Net):
    dataset = DVS_Lip("train", args)
    net.logger.info(f"Start Training, Data Length: {len(dataset)}")
    
    loader = dataset2dataloader(dataset, args.batch_size, args.num_workers)
    loss_fn = nn.CrossEntropyLoss()

    tot_iter = 0
    best_acc, best_acc_p1, best_acc_p2 = 0.0, 0.0, 0.0
    best_epoch = 0
    alpha = 0.2
    scaler = GradScaler(device.type)
    for epoch in range(args.max_epoch):
        for (i_iter, input) in enumerate(loader):
            tic = time.time()

            net.net.train()
            event_low = input.get("event_low").to(device, non_blocking=True)
            event_high = input.get("event_high").to(device, non_blocking=True)
            label = input.get("label").to(device, non_blocking=True).long()

            loss = {}
            with autocast(device.type):
                logit = net.net(event_low, event_high)
                loss_bp = loss_fn(logit, label)

            loss["Total"] = loss_bp
            net.optimizer.zero_grad()
            scaler.scale(loss_bp).backward()  
            scaler.step(net.optimizer)
            scaler.update()
            
            toc = time.time()
            if i_iter % 20 == 0:
                msg = "epoch={},train_iter={},eta={:.5f}".format(epoch, tot_iter, (toc-tic)*(len(loader)-i_iter)/3600.0)
                for k, v in loss.items():
                    msg += ",{}={:.5f}".format(k, v)
                msg = msg + str(",lr=" + str(showLR(net.optimizer)))
                msg = msg + str(",best_acc={:2f}".format(best_acc))
                net.logger.info(msg)
            net.writer.add_scalar("lr", float(showLR(net.optimizer)), tot_iter)
            net.writer.add_scalar("loss", loss_bp.item(), tot_iter)
            
            if i_iter == len(loader) - 1 or (epoch == 0 and i_iter == 0):
                acc, acc_p1, acc_p2, msg = test(args, device, net)
                net.logger.info(msg)
                net.writer.add_scalar("test_acc", acc, tot_iter)
                net.writer.add_scalar("test_acc/part1", acc_p1, tot_iter)
                net.writer.add_scalar("test_acc/part2", acc_p2, tot_iter)

                if acc > best_acc:
                    best_acc, best_acc_p1, best_acc_p2, best_epoch = acc, acc_p1, acc_p2, epoch
                    savename = net.log_dir + "/model_best.pth"
                    temp = os.path.split(savename)[0]
                    if not os.path.exists(temp):
                        os.makedirs(temp)
                    torch.save(net.net.module.state_dict(), savename)
                    
            tot_iter += 1        
            
        net.scheduler.step()

    net.logger.info("best_acc={:2f}".format(best_acc))
    net.logger.info("best_acc_part1={:2f}".format(best_acc_p1))
    net.logger.info("best_acc_part2={:2f}".format(best_acc_p2))
    net.logger.info("best_epoch={:2f}".format(best_epoch))

    return net