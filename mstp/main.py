from utils.utils import *
import torch
from compression.prune import apply_pruning
from compression.distillation import train_with_distillation
from utils.net import Net
from utils.test import test
from utils.train import train


if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, required=False)
    parser.add_argument("--lr", type=float, required=False, default=3e-4)
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--n_class", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--num_workers", type=int, required=False, default=12)
    parser.add_argument("--max_epoch", type=int, required=False, default=80)
    parser.add_argument("--num_bins", type=str2list, required=True, default="1+4")
    parser.add_argument("--test", type=str2bool, required=False, default="false")
    parser.add_argument("--log_dir", type=str, required=False, default=None)
    parser.add_argument("--weights", type=str, required=False, default=None)

    # dataset
    parser.add_argument("--event_root", type=str, default="./data/DVS-Lip")

    # model
    parser.add_argument("--se", type=str2bool, default=False)
    parser.add_argument("--base_channel", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--beta", type=int, default=5)
    parser.add_argument("--t2s_mul", type=int, default=2)

    # compression
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--distillation", action="store_true")
    parser.add_argument("--teacher_weights", type=str, required=False, default=None)
    parser.add_argument("--use_simple", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--d", type=int, default=0.02)
    parser.add_argument("--p", type=int, default=0.02)

    # tests
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--use_profiler", action="store_true")

    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Optional: If using CUDA, print additional info
    if device.type == "cuda":
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())

    mstp_net = Net(args, device, weigths_path = args.weights, log_dir = args.log_dir, is_simple = args.use_simple)

    if args.test:
        print("Testing...")
        for i in range(args.reps):
            print(f"Starting test {i + 1}...")
            acc, acc_p1, acc_p2, msg = test(args, device, mstp_net, use_profiler=args.use_profiler)
            print(msg)
            print(f"Ended test {i + 1}")
        print("Tested successfully.")
    else:
        print("Training...")
        if not args.skip_train:
            if not args.distillation:
                mstp_net = train(args, device, mstp_net)
            elif args.distillation:
                if args.teacher_weights is None:
                    raise Exception("No weights for teacher net")
                teacher_net = Net(args, device, weigths_path = args.teacher_weights, log_dir = None)
                mstp_net = train_with_distillation(args, device, mstp_net, teacher_net, args.d)
        if args.prune:
            mstp_net = apply_pruning(args, device, mstp_net, args.p)
        print("Trained successfully.")
