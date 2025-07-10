import argparse
import os
import warnings

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import transforms
from source.data.augs import augmentation_strong

from source.models.classification.knet import AKOrN
from source.training_utils import save_checkpoint, save_model
from source.utils import load_state_dict_ignore_size_mismatch, str2bool

from source.evals.classification.adv_attacks import (
    fgsm_attack,
    pgd_linf_attack,
    autoattack,
    random_attack,
)
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def evaluate_model(net, testloader, criterion, device="cuda", eps=0.0, attack_method="fgsm"):
    """Evaluate model accuracy with optional adversarial attacks"""
    correct = 0
    total = 0
    net.eval()
    
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        if eps > 0:
            if attack_method == "fgsm":
                inputs = fgsm_attack(net, inputs, labels, eps, criterion=criterion)
            elif attack_method == "random":
                inputs = random_attack(inputs, eps)
            elif attack_method == "pgd":
                inputs = pgd_linf_attack(
                    net,
                    inputs,
                    labels,
                    eps,
                    alpha=eps / 3,
                    num_iter=20,
                    criterion=criterion,
                )
            elif attack_method == "autoattack":
                inputs = autoattack(net, inputs, labels, eps)
            else:
                raise NotImplementedError(f"Attack method {attack_method} not implemented")
        
        with torch.no_grad():
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    if eps > 0:
        if attack_method == "fgsm":
            print(f"FGSM Adversarial Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "random":
            print(f"Random Noise Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "pgd":
            print(f"PGD Adversarial Accuracy: {acc:.2f}%, eps:{255*eps:.1f}/255")
        elif attack_method == "autoattack":
            print(f"Autoattack Adversarial Accuracy: {acc:.2f}%, eps:{eps:.1f}/255")
    else:
        print(f"Accuracy of the network on the test images: {acc:.2f}%")
    return acc


def train_epoch(net, epoch, ema, trainloader, optimizer, criterion, device="cuda"):
    """Train for one epoch"""
    net.train()
    running_loss = 0.0
    n = 0
    
    for i, data in tqdm(enumerate(trainloader, 0)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        running_loss += loss.item() * inputs.shape[0]
        n += inputs.shape[0]
    
    avg_loss = running_loss / n
    print(f"Epoch: {epoch} Training loss: {avg_loss:.3f}")
    return avg_loss


def adversarial_evaluation(writer, epoch, net, testloader, criterion, eps=8/255, eval_pgd=False, prefix="model/", device="cuda"):
    """Comprehensive adversarial evaluation"""
    # Clean accuracy
    writer.add_scalar(prefix+"test accuracy", evaluate_model(net, testloader, criterion, device), epoch)
    
    # Random noise
    writer.add_scalar(
        prefix+"Random noise test accuracy",
        evaluate_model(net, testloader, criterion, device, 64/255, attack_method="random"),
        epoch,
    )
    
    # FGSM attack
    writer.add_scalar(
        prefix+"FGSM test accuracy", 
        evaluate_model(net, testloader, criterion, device, eps, attack_method="fgsm"), 
        epoch
    )
    
    # PGD attack (less frequent)
    if eval_pgd:
        writer.add_scalar(
            prefix+"PGD test accuracy", 
            evaluate_model(net, testloader, criterion, device, eps, attack_method="pgd"), 
            epoch
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training")
    
    parser.add_argument("exp_name", type=str, help="Experiment name")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every specified epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--criterion", type=str, default="ce", help="Loss criterion")
    parser.add_argument("--beta", type=float, default=0.99, help="EMA decay rate")
    parser.add_argument("--adveval_freq", type=int, default=10, help="Frequency of adversarial evaluation")
    parser.add_argument("--pgdeval_freq", type=int, default=50, help="Frequency of PGD evaluation")
    
    # Data arguments
    parser.add_argument("--data", type=str, default="cifar10", help="Dataset")
    parser.add_argument("--batchsize", type=int, default=128, help="Batch size")
    
    # Model arguments
    parser.add_argument("--n", type=int, default=2, help="Oscillator dimension")
    parser.add_argument("--L", type=int, default=3, help="Number of layers")
    parser.add_argument("--C", type=int, default=10, help="Number of classes")
    parser.add_argument("--ch", type=int, default=64, help="Number of channels")
    parser.add_argument("--T", type=int, default=3, help="Timesteps")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter")
    parser.add_argument("--J", type=str, default="conv", help="Connectivity type")
    parser.add_argument("--ksizes", nargs="+", type=int, default=[9, 7, 5], help="Kernel sizes")
    parser.add_argument("--ro_ksize", type=int, default=3, help="Readout kernel size")
    parser.add_argument("--ro_N", type=int, default=2, help="Readout N")
    parser.add_argument("--norm", type=str, default="bn", help="Normalization")
    parser.add_argument("--c_norm", type=str, default="gn", help="C normalization")
    parser.add_argument("--use_omega", type=str2bool, default=True, help="Use omega")
    parser.add_argument("--init_omg", type=float, default=1.0, help="Initial omega")
    parser.add_argument("--global_omg", type=str2bool, default=True, help="Global omega")
    parser.add_argument("--learn_omg", type=str2bool, default=True, help="Learn omega")
    parser.add_argument("--ensemble", type=int, default=1, help="Ensemble size")
    
    # Fine-tuning
    parser.add_argument("--finetune", type=str, default=None, help="Path to checkpoint for fine-tuning")
    parser.add_argument("--ignore_size_mismatch", action="store_true", help="Ignore size mismatch when loading")
    
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform_aug = augmentation_strong(imsize=32)
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Dataset loading
    if args.data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_aug
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {args.data} not implemented")
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batchsize, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4
    )
    
    # Loss criterion
    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Criterion {args.criterion} not implemented")
    
    # Model
    net = AKOrN(
        n=args.n,
        ch=args.ch,
        out_classes=args.C,
        L=args.L,
        T=args.T,
        J=args.J,
        ksizes=args.ksizes,
        ro_ksize=args.ro_ksize,
        ro_N=args.ro_N,
        norm=args.norm,
        c_norm=args.c_norm,
        gamma=args.gamma,
        use_omega=args.use_omega,
        init_omg=args.init_omg,
        global_omg=args.global_omg,
        learn_omg=args.learn_omg,
        ensemble=args.ensemble,
    ).to(device)
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    
    # Load checkpoint if specified
    if args.finetune:
        if args.ignore_size_mismatch:
            load_state_dict_ignore_size_mismatch(
                net, torch.load(args.finetune)["model_state_dict"]
            )
        else:
            net.load_state_dict(
                torch.load(args.finetune)["model_state_dict"], strict=False
            )
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    
    if args.finetune:
        try:
            optimizer.load_state_dict(torch.load(args.finetune)["optimizer_state_dict"])
        except:
            warnings.warn("Optimizer state dict could not be loaded")
        optimizer.param_groups[0]["lr"] = args.lr
    
    # EMA setup
    from ema_pytorch import EMA
    ema = EMA(net, beta=args.beta, update_every=10, update_after_step=200)
    
    # Load EMA checkpoint if fine-tuning
    if args.finetune:
        dir_name, file_name = os.path.split(args.finetune)
        file_name = file_name.replace("checkpoint", "ema")
        ema_path = os.path.join(dir_name, file_name)
        if os.path.exists(ema_path):
            ema.load_state_dict(torch.load(ema_path)["model_state_dict"])
            print(f"Loaded EMA model from {ema_path}")
    
    # Logging
    jobdir = f"runs/{args.exp_name}/"
    writer = SummaryWriter(jobdir)
    writer_ema = SummaryWriter(os.path.join(jobdir, "ema"))
    
    print("Start training...")
    
    # Training loop
    for epoch in range(args.epochs):
        total_loss = train_epoch(net, epoch, ema, trainloader, optimizer, criterion, device)
        
        # Log training loss
        writer.add_scalar("training loss", total_loss, epoch)
        
        # Adversarial evaluation
        if ((epoch + 1) % args.adveval_freq) == 0:
            # Evaluate original model
            print(f"Evaluating original model at epoch {epoch}")
            adversarial_evaluation(
                writer,
                epoch,
                net,
                testloader,
                criterion,
                8/255,
                eval_pgd=True if ((epoch + 1) % args.pgdeval_freq) == 0 else False,
                device=device,
            )
            print(f"Evaluating EMA model at epoch {epoch}")
            adversarial_evaluation(
                writer_ema,
                epoch,
                ema.ema_model,
                testloader,
                criterion,
                8/255,
                eval_pgd=True if ((epoch + 1) % args.pgdeval_freq) == 0 else False,
                device=device,
            )
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(net, optimizer, epoch, total_loss, checkpoint_dir=jobdir)
            save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")
    
    # Save final models
    torch.save(net.state_dict(), os.path.join(jobdir, "model.pth"))
    torch.save(ema.state_dict(), os.path.join(jobdir, "ema_model.pth"))
    print("Training completed!")