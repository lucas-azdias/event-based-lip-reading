import os
import torch
import torch.nn.utils.prune as prune
from utils.net import Net
from utils.test import test


def adjust_state_dict(state_dict, model_is_dataparallel):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if model_is_dataparallel and not key.startswith("module."):
            new_key = "module." + key
        elif not model_is_dataparallel and key.startswith("module."):
            new_key = key[len("module."):]
        new_state_dict[new_key] = value
    return new_state_dict

def apply_pruning(args, device, net:Net, step=0.02):
    amount = step
    best_amount = 0
    model_is_dataparallel = isinstance(net.net, torch.nn.DataParallel)
    best_state_dict = adjust_state_dict(getattr(net.net, 'module', net.net).state_dict(), model_is_dataparallel)

    with torch.no_grad():
        acc, _, _, _ = test(args, device, net)
    acc_stop_threshold = acc - 0.02  # critério de parada: 2% ou 5% abaixo
    net.logger.info(f"Acurácia inicial antes da poda: {acc:.4f} | Limite mínimo aluno: {acc_stop_threshold:.4f}")

    while amount < 1:
        net.net.load_state_dict(adjust_state_dict(best_state_dict, model_is_dataparallel))

        for name, module in net.net.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                # Aplica pruning L1
                prune.l1_unstructured(module, name="weight", amount=amount)
                # Remove o reparametrization hook (permite que o peso seja realmente modificado)
                prune.remove(module, "weight")
        
        pruned_acc, _, _, _ = test(args, device, net)
        
        net.logger.info(f"Prunned with amount={amount} got acc={pruned_acc}")
        
        if pruned_acc >= acc_stop_threshold:
            best_amount = amount
            best_state_dict = adjust_state_dict(getattr(net.net, 'module', net.net).state_dict(), model_is_dataparallel)
            amount += step
        else:
            net.net.load_state_dict(adjust_state_dict(best_state_dict, model_is_dataparallel))
            break

        best_state_dict = getattr(net.net, 'module', net.net).state_dict()

    savename = net.log_dir + "/model_best.pth"
    os.makedirs(os.path.split(savename)[0], exist_ok=True)
    torch.save(getattr(net.net, 'module', net.net).state_dict(), savename)
    print(f"Pruning applied with amount={best_amount}")