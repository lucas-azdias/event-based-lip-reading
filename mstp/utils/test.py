from utils.utils import *
from utils.dataset import DVS_Lip
import time
import numpy as np
import torch
from tqdm import tqdm
from torch.amp import autocast
import torch.profiler
from utils.net import Net

def test(args, device, net: Net, use_profiler=False):
    if not use_profiler:
        return test_normal(args, device, net)
    else:
        return test_with_profiler(args, device, net)


def test_normal(args, device, net: Net):
    with torch.no_grad():
        dataset = DVS_Lip("test", args)
        if net.logger:
            net.logger.info(f"Start Testing, Data Length: {len(dataset)}")
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
        if net.logger:
            net.logger.info("start testing")
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}

        print(f"Total iterations: {len(loader)}")

        for (i_iter, input) in tqdm(enumerate(loader)):
            net.net.eval()
            tic = time.time()
            event_low = input.get("event_low").to(device, non_blocking=True)
            event_high = input.get("event_high").to(device, non_blocking=True)
            label = input.get("label").to(device, non_blocking=True)
            with autocast(device.type):
                logit = net.net(event_low, event_high)

            v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())
            toc = time.time()
            label_list = label.cpu().numpy().tolist()
            pred_list = logit.argmax(-1).cpu().numpy().tolist()
            for i in range(len(label_list)):
                label_pred[label_list[i]].append(pred_list[i])

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())
        msg = "test acc: {:.5f}, acc part1: {:.5f}, acc part2: {:.5f}".format(acc, acc_p1, acc_p2)
        return acc, acc_p1, acc_p2, msg
    

def test_with_profiler(args, device, net: Net):
    with torch.no_grad():
        dataset = DVS_Lip("test", args)
        if net.logger:
            net.logger.info(f"Start Testing, Data Length: {len(dataset)}")
        loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, shuffle=False)        
        
        if net.logger:
            net.logger.info("start testing")
        v_acc = []
        label_pred = {i: [] for i in range(args.n_class)}

        print(f"Total iterations: {len(loader)}")

        # Params
        total_params = sum(p.numel() for p in net.net.parameters())
        trainable_params = sum(p.numel() for p in net.net.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Reset GPU peak memory tracking
        torch.cuda.reset_peak_memory_stats(device)

        # Use profiler through ALL iterations!
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            with_flops=True,
            profile_memory=True
        ) as prof:

            start_time = time.time()  # Start timing full test

            for (i_iter, input) in tqdm(enumerate(loader)):
                net.net.eval()

                event_low = input.get("event_low").to(device, non_blocking=True)
                event_high = input.get("event_high").to(device, non_blocking=True)
                label = input.get("label").to(device, non_blocking=True)

                with autocast(device.type):
                    logit = net.net(event_low, event_high)

                v_acc.extend((logit.argmax(-1) == label).cpu().numpy().tolist())

                label_list = label.cpu().numpy().tolist()
                pred_list = logit.argmax(-1).cpu().numpy().tolist()
                for i in range(len(label_list)):
                    label_pred[label_list[i]].append(pred_list[i])

                # Step profiler
                prof.step()

            end_time = time.time()  # End timing
            total_time = end_time - start_time

        # Profiler summary
        # print("===== Torch Profiler Summary =====")
        # print(prof.key_averages().table(
        #     sort_by="flops",
        #     row_limit=20
        # ))

        # Total FLOPs per forward pass
        total_flops = sum([item.flops for item in prof.key_averages() if item.flops is not None])

        # Peak GPU RAM used
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)

        acc_p1, acc_p2 = compute_each_part_acc(label_pred)
        acc = float(np.array(v_acc).reshape(-1).mean())

        msg = (
            f"test acc: {acc:.5f}, acc part1: {acc_p1:.5f}, acc part2: {acc_p2:.5f}, "
            f"Total Params: {total_params:,}, Trainable Params: {trainable_params:,}, "
            f"FLOPs per pass: {total_flops:.2e}, Time spent: {total_time:.2f}s, "
            f"Peak GPU RAM: {peak_mem_mb:.2f} MB"
        )

        return acc, acc_p1, acc_p2, msg