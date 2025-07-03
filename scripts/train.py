import argparse
import os
import warnings

import torch
import torch.nn as nn
from torch import optim
from source.sophia import SophiaG
import torchvision
from torchvision import transforms
from source.evals.adv_attacks import (
    fgsm_attack,
    pgd_linf_attack,
    autoattack,
    random_attack,
)
from source.models.knetwork import KNetworkCIFAR10
from source.models.deepknetwork import DeepKuramotoNetwork
from source.models.resnet import ResNet18V2, ResNet18V2k5, ResNet18V2dr05
from source.models.vit import VisionTransformer
from source.training_utils import save_checkpoint, save_model, LinearWarmupScheduler
from source.utils import load_state_dict_ignore_size_mismatch
from source.data.augs import my_aug, augmix, random_Linf_noise
from torch.utils.tensorboard import SummaryWriter
from source.data.datasets.imagenet.imagenet32 import ImageNet32Dataset

from source.utils import compare_optimizer_state_dicts, str2bool

# Visualization

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from IPython.display import HTML

from tqdm import tqdm


def add_gradient_histograms(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(name + "/grad", param.grad, epoch)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="OccilatorMachine")

    parser.add_argument("exp_name", type=str, help="expname")
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="end core")
    parser.add_argument("--epochs", type=int, default=200, help="num of epochs")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="save checkpoint every specified epochs",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="lr")
    parser.add_argument(
        "--lr_k",
        type=float,
        default=2e-4,
        help="lr for k-layers, should be higher than lr, since currently the k-weights are bit harder to be optimized than standard layers",
    )
    parser.add_argument(
        "--train_only_klayers", action="store_true", help="K-layers are only optimized"
    )
    parser.add_argument("--criterion", type=str, default="ce")
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_start", type=int, default=50)
    parser.add_argument("--swa_freq", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.99, help="ema beta")
    parser.add_argument("--cos_lr_T", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--adveval_freq", type=int, default=1, help="freqadv eval")
    parser.add_argument("--pgdeval_freq", type=int, default=20, help="freqadv eval")

    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--augtype", type=str, default="my_aug")

    parser.add_argument("--model", type=str, default="knet", help="model")
    parser.add_argument("--N", nargs="+", type=int, help="occilator dimensions")
    parser.add_argument("--L", type=int, default=3, help="num of layers")
    parser.add_argument("--R", type=int, default=1, help="num of repetitions")
    parser.add_argument("--C", type=int, default=10, help="num of classes")
    parser.add_argument("--ch", type=int, default=64, help="num of channels")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size ")
    parser.add_argument("--st", type=str, default=None)
    parser.add_argument("--ksizes", nargs="+", type=int, help="Kernel sizes")
    parser.add_argument("--T", nargs="+", type=int, help="Timesteps")
    parser.add_argument("--J", nargs="+", type=str, default="conv", help="connectivity")
    parser.add_argument("--use_omega", action="store_true")
    parser.add_argument("--c_as_omega", action="store_true")
    parser.add_argument("--skip_klayer", action="store_true")
    parser.add_argument("--angle", action="store_true")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--knorm", type=str, default="gn")
    parser.add_argument("--gaussinit", action="store_true")
    parser.add_argument("--ro_ksize", type=int, default=3)
    parser.add_argument("--psize", type=int, default=4)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--mix_different_groups", action="store_true")
    parser.add_argument("--learned_init_x", action="store_true")
    parser.add_argument("--init_x_from_c", action="store_true")
    parser.add_argument("--random_phase", action="store_true")
    parser.add_argument("--last_softmax", action="store_true")
    parser.add_argument("--del_t", type=float, default=1.0, help="time step")
    parser.add_argument("--aa", action="store_true", help="use antialias conv")
    parser.add_argument("--kmodel_type", type=str, default="type3")
    parser.add_argument("--no_ro", action="store_true")
    parser.add_argument("--omg_c", type=str2bool, default=False)
    parser.add_argument("--like_a_transformer", type=str2bool, default=False)
    parser.add_argument("--gcn", type=str2bool, default=False)
    parser.add_argument("--global_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=1.0)
    parser.add_argument("--multi_res", type=str2bool, default=False)
    parser.add_argument("--ro_x", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=True)
    parser.add_argument("--nl", type=str2bool, default=True)
    parser.add_argument("--maxpool", type=str2bool, default=False)
    parser.add_argument("--noise", type=float, default=0.)
    parser.add_argument("--roN", type=int, default=None)
    parser.add_argument("--speed_test", action="store_true")

    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="path to the checkpoint. Training starts from that checkpoint with a new optimizer",
    )
    parser.add_argument(
        "--finetune_ema",
        type=str,
        default=None,
        help="path to ema model",
    )

    parser.add_argument("--ignore_size_mismatch", action="store_true")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    mean = ()
    if args.augtype == "my_aug":
        transform_aug = my_aug()
    elif args.augtype == "my_aug_strong":
        from source.data.augs import my_aug_strong
        transform_aug = my_aug_strong(args.noise)
    elif args.augtype == "simclr":
        from source.data.augs import simclr_aug 
        transform_aug = simclr_aug()
    elif args.augtype == "my_aug_noise":
        from source.data.augs import my_aug_noise
        transform_aug = my_aug_noise(args.noise)
    elif args.augtype == "augmix":
        transform_aug = augmix()
    elif args.augtype == "plus_random_noise":
        transform_aug = random_Linf_noise(my_aug(), epsilon=64 / 255)
    elif args.augtype == "none":
        transform_aug = transforms.Compose([transforms.ToTensor()])
    else:
        raise NotImplementedError

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    transform = transforms.Compose([transforms.ToTensor()])

    if args.data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_aug
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.data == "imagenet32":
        trainset = ImageNet32Dataset(
            "./data/Imagenet32", transform=transform_aug, train=True
        )
        testset = ImageNet32Dataset(
            "./data/Imagenet32", transform=transform, train=False
        )
    elif args.data == "mnist":
        from source.data.datasets.mnist.mnist import MNISTDataset
        from torchvision.transforms import InterpolationMode

        transform_train = transforms.Compose(
            [
                transforms.RandomRotation(30, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        trainset = MNISTDataset("./data/", train=True, transform=transform_train)
        transform_test = transforms.Compose(
            [
                transforms.RandomRotation(90, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        testset = MNISTDataset("./data/", train=False, transform=transform_test)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
    )

    def add_energy_histograms(writer, net, epoch):

        def get_energy_tensors(net):
            img, _ = next(iter(testloader))
            _, _, _, energy_tensors = net.feature(img.to("cuda"))
            return energy_tensors

        energy_tensors = get_energy_tensors(net)

        sum_energy = 0
        for i, tensor in enumerate(energy_tensors):
            energies = tensor[-1].detach().cpu().numpy()  # take last
            writer.add_histogram(f"energy_tensor_l{i}_hist", energies, epoch)
            writer.add_scalar(f"energy_tensor_l{i}_mean", energies.sum(), epoch)
            sum_energy += energies.sum()
        writer.add_scalar(f"mean_all_energy", sum_energy, epoch)

    if args.criterion == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "mse":
        _crterion = nn.MSELoss()

        def _mse(x, t):
            # t is assumed to be a discrete label
            t = nn.functional.one_hot(t, num_classes=args.C)
            return (
                _crterion(x, t.to(torch.float32)) * args.C
            )  # scaling loss by num of classes

        criterion = _mse
    else:
        raise NotImplementedError

    def _eval(net, eps=0.0, attack_method="fgsm"):
        correct = 0
        total = 0
        net.eval()
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
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
                    raise NotImplementedError
            # with torch.no_grad():
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        if eps > 0:
            if attack_method == "fgsm":
                print(
                    f"FGSM Adversarial Accuracy of the network on the 10000 test images: {acc:.2f}%, eps:{255*eps:.1f}/255"
                )
            elif attack_method == "random":
                print(
                    f"Random Noise Accuracy of the network on the 10000 test images: {acc:.2f}%, eps:{255*eps:.1f}/255"
                )
            elif attack_method == "pgd":
                print(
                    f"PGD Adversarial Accuracy of the network on the 10000 test images: {acc:.2f}%, eps:{255*eps:.1f}/255"
                )
            elif attack_method == "autoattack":
                print(
                    f"Autoattack Adversarial Accuracy of the network on the 10000 test images: {acc:.2f}%, eps:{eps:.1f}/255"
                )
        else:
            print(f"Accuracy of the network on the 10000 test images: {acc:.2f}%")
        return acc

    jobdir = f"runs/{args.exp_name}/"
    writer = SummaryWriter(jobdir)
    writer_ema = SummaryWriter(os.path.join(jobdir, "ema"))

    if args.swa:
        writer_swa = SummaryWriter(os.path.join(jobdir, "swa"))

    def evaluation(writer, epoch, net, eps, eval_pgd=False, prefix="model/"):
        writer.add_scalar(prefix+"test accuracy", _eval(net), epoch)
        writer.add_scalar(
            prefix+"Random noise test accuracy",
            _eval(net, 64 / 255, attack_method="random"),
            epoch,
        )
        if isinstance(net, KNetworkCIFAR10):
            net.test_time_st = "full"
        writer.add_scalar(
            prefix+"FGSM test accuracy", _eval(net, eps, attack_method="fgsm"), epoch
        )
        if isinstance(net, KNetworkCIFAR10):
            net.test_time_st = net.training_time_st
        writer.add_scalar(
            prefix+"FGSM test accuracy wo test time ST",
            _eval(net, eps, attack_method="fgsm"),
            epoch,
        )
        if eval_pgd:
            writer.add_scalar(
                prefix+"PGD test accuracy", _eval(net, eps, attack_method="pgd"), epoch
            )



    def train(net, opt, epoch):
        losses = []

        initial_params = {name: param.clone() for name, param in net.named_parameters()}
        running_loss = 0.0
        n = 0
        k = 50
        iter_num = 0
        if args.speed_test:
            it_sp = 0
            time_per_iter = []
            import numpy as np
        for i, data in tqdm(enumerate(trainloader, 0)):
            net.train()   
            inputs, labels = data
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            opt.zero_grad()

            if args.speed_test:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16 if args.bf16 else torch.float32
            ):
                outputs = net(inputs)
            outputs = outputs.to(torch.float32)

            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            
            if args.speed_test:
                end.record()
                torch.cuda.synchronize()
                time_elapsed_per_iter = start.elapsed_time(end)
                time_per_iter.append(time_elapsed_per_iter)
                print(time_elapsed_per_iter)
                it_sp = it_sp + 1
                if it_sp == 100:
                    np.save(os.path.join(jobdir, 'time.npy'), np.array(time_per_iter))
                    exit(0)
                


            running_loss += loss.item() * inputs.shape[0]
            n += inputs.shape[0]

            iter_num += 1
            # if iter_num > 500:
            #    break
            if isinstance(optimizer, SophiaG):
                if (iter_num + 1) % k != k - 1:
                    continue
                else:
                    X = inputs
                    # update hessian EMAX
                    out = net(X)
                    if args.criterion == "mse":
                        samp_dist = torch.distributions.Categorical(
                            probs=torch.clamp(out, 0, 1)
                        )
                    else:
                        samp_dist = torch.distributions.Categorical(logits=out)
                    y_sample = samp_dist.sample()
                    if args.criterion == "mse":
                        loss_sampled = criterion(out, y_sample)
                    else:
                        loss_sampled = criterion(
                            out.view(-1, out.size(-1)), y_sample.view(-1)
                        )

                    optimizer.zero_grad(set_to_none=True)
                    net.zero_grad()

                    loss_sampled.backward()
                    optimizer.update_hessian()

        if isinstance(net, KNetworkCIFAR10):
            add_energy_histograms(writer, net, epoch)

        add_gradient_histograms(writer, net, epoch)
        for name, param in net.named_parameters():
            diff = param - initial_params[name]
            writer.add_histogram(f"{name}_diff", diff, epoch)

        print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss/n:.3f}")

        total_loss = running_loss / n
        writer.add_scalar("training loss", total_loss, epoch)

        return total_loss

    if args.model == "knet":
        lr = args.lr
        warmup_iter = 0
        net = KNetworkCIFAR10(
            args.N,
            ch=args.ch,
            C=args.C,
            L=args.L,
            T=args.T,
            gamma=args.gamma,
            J=args.J,
            angle=args.angle,
            use_omega=args.use_omega,
            c_as_omega=args.c_as_omega,
            skip_klayer=args.skip_klayer,
            tau=args.tau,
            ksizes=args.ksizes,
            del_t=args.del_t,
            norm=args.norm,
            ro_ksize=args.ro_ksize,
            groups=args.groups,
            mix_different_groups=args.mix_different_groups,
            aa=args.aa,
            weighted_avg=False,
            training_time_st=args.st,
            learned_init_x=args.learned_init_x,
            init_x_from_c=args.init_x_from_c,
            last_softmax=args.last_softmax,
            random_phase=args.random_phase,
            gaussinit=args.gaussinit,
            use_ro=not args.no_ro,
            kmodel_type=args.kmodel_type,
            knorm=args.knorm,
            omg_c=args.omg_c,
            like_a_transformer=args.like_a_transformer,
            R=args.R,
            psize=args.psize,
            gcn=args.gcn,
            global_omg=args.global_omg,
            init_omg=args.init_omg,
            multi_res=args.multi_res,
            learn_omg=args.learn_omg,
            nl=args.nl,
            ro_x=args.ro_x,
            maxpool=args.maxpool,
            roN=args.roN, 
        ).to("cuda")

    elif args.model == "resnet":
        net = ResNet18V2(num_classes=args.C).cuda()
    elif args.model == "resnet_k5":
        net = ResNet18V2k5(num_classes=args.C).cuda()
    elif args.model == "resnet_dr05":
        net = ResNet18V2dr05(num_classes=args.C).cuda()
    elif args.model == "vgg":
        from source.models.vgg import VGG11
        net = VGG11().cuda()
    elif args.model == "vit":
        net = VisionTransformer(32, patch_size=args.psize, in_channels=3, num_classes=args.C, embed_dim=args.ch, mlp_dim=args.ch*2, num_heads=8).cuda()
    elif args.model == "itrconv":
        from source.models.itrconv import ItrConv
        net = ItrConv(ch=args.ch, num_classes=args.C, L=args.L, T=args.T, ksizes=args.ksizes, input_dim=3).cuda()
    else:
        raise NotImplementedError
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    if args.finetune:
        if args.ignore_size_mismatch:
            load_state_dict_ignore_size_mismatch(
                net, torch.load(args.finetune)["model_state_dict"]
            )
        else:
            net.load_state_dict(
                torch.load(args.finetune)["model_state_dict"], strict=False
            )

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    # optimizer = SophiaG(net.parameters(), lr=args.lr, rho=0.05, weight_decay=0.1)

    if args.finetune:
        try:
            optimizer.load_state_dict(torch.load(args.finetune)["optimizer_state_dict"])
        except:
            warnings.warn("optimizer state dict could not be loaded")
        optimizer.param_groups[0]["lr"] = args.lr

    if args.swa:
        from torch.optim.swa_utils import AveragedModel

        swa_model = AveragedModel(net)
        if args.cos_lr_T > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, args.cos_lr_T
            )
        else:
            scheduler = None
    else:
        scheduler = None

    print("Start training...")

    from ema_pytorch import EMA
    
    ema = EMA(net, beta=args.beta, update_every=1, update_after_step=1)
    if args.finetune_ema is not None:
        print(f"Loading EMA model from {args.finetune_ema}")
        ema.load_state_dict(torch.load(args.finetune_ema)["model_state_dict"])

    for epoch in range(0, args.epochs):
        total_loss = train(net, optimizer, epoch)
        scheduler.step() if scheduler is not None else None
        ema.update()
        if ((epoch + 1) % args.adveval_freq) == 0:
            evaluation(
                writer,
                epoch,
                net,
                8 / 255,
                eval_pgd=True if ((epoch + 1) % args.pgdeval_freq) == 0 else False,
            )
            evaluation(
                writer_ema,
                epoch,
                ema.ema_model,
                8 / 255,
                eval_pgd=True if ((epoch + 1) % args.pgdeval_freq) == 0 else False,
            )

        if (
            args.swa
            and (epoch + 1) >= args.swa_start
            and ((epoch + 1) % args.swa_freq) == 0
        ):
            print("SWA update...")
            swa_model.update_parameters(net)
            torch.optim.swa_utils.update_bn(trainloader, swa_model, device="cuda")
            evaluation(
                writer_swa,
                epoch,
                swa_model.module,
                8 / 255,
                eval_pgd=True if ((epoch + 1) % 20) == 0 else False,
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(net, optimizer, epoch, total_loss, checkpoint_dir=jobdir)
            save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")
            if args.swa:
                save_model(swa_model, epoch, checkpoint_dir=jobdir, prefix="swa")

    torch.save(net.state_dict(), os.path.join(jobdir, f"model.pth"))
    torch.save(ema.state_dict(), os.path.join(jobdir, f"ema_model.pth"))
