import os
from model.model import MSTP
from model.simpler_model import SimpleMSTP
import torch.optim as optim
import torch
import torch.nn as nn
from utils.utils import *

class Net():
    def __init__(self, args, device, weigths_path=None, log_dir=None, is_simple=False):
        net = MSTP(args).to(device) if not is_simple else SimpleMSTP(args).to(device)

        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=5e-6)
        if log_dir:
            logger, writer, log_dir = build_log(args)
            logger.info("Network Arch: ")
            logger.info(net)
        else:
            logger, writer, log_dir = (None, None, None)

        if weigths_path is not None:
            if log_dir:
                logger.info("load weights")
            # Load weights to the appropriate device
            weight = torch.load(os.path.join("log", weigths_path, "model_best.pth"), map_location=device)
            load_missing(net, weight)
        
        net = nn.DataParallel(net)
        
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.writer = writer
        self.log_dir = log_dir