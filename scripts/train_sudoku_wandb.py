import torch
import sys, os
import tqdm
import argparse
import wandb

from source.models.sudoku.transformer import SudokuTransformer

from source.training_utils import save_checkpoint, save_model
from source.data.datasets.sudoku.sudoku import SudokuDataset, HardSudokuDataset
#from source.models.sudoku.knet import SudokuAKOrN
from source.models.sudoku.my_knet import MySudokuAKOrN as SudokuAKOrN
from source.utils import str2bool
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ema_pytorch import EMA
import datetime
import random
from pathlib import Path
import json

def apply_threshold(model, threshold):
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.where(
                param.abs() < threshold, torch.tensor(0.0), param.data
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, help="expname")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--epochs", type=int, default=100, help="num of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--beta", type=float, default=0.995, help="ema decay")
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="clip grad norm"
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="save checkpoint every specified epochs",
    )
    parser.add_argument("--eval_freq", type=int, default=10, help="freqadv eval")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="sudoku_akorn", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    
    # Save directory
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results. Defaults to [experiment_name]_[timestamp]')

    # Data loading
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="end core")
    parser.add_argument("--data_root", type=str, default=None,
        help="Optional. Specify the root dir of the dataset. If None, use a default path set for each dataset",
    )
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)

    # General model options
    parser.add_argument("--model", type=str, default="akorn", help="model")
    parser.add_argument("--L", type=int, default=1, help="num of layers")
    parser.add_argument("--T", type=int, default=16, help="Timesteps")
    parser.add_argument("--ch", type=int, default=512, help="num of channels")
    parser.add_argument("--heads", type=int, default=8)

    # AKOrN options
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity")
    parser.add_argument("--J_bias", type=str2bool, default=False, help="use bias in connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)
    parser.add_argument("--ksize", type=int, default=9, help="kernel size for KLayer")
    parser.add_argument("--bp_steps", type=int, default=None, help="number of back propagation steps in KLayer. If loss is itp, automatically set to None")
    parser.add_argument("--speed_test", action="store_true")

    # loss choices
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "align_energy", "ipt"], help="how to compute loss, cross_entropy or align_energy")

    # --- Align (Koyama-Hayashi-Takashiro loss) ---
    parser.add_argument("--align_energy_steps", type=int, default=None, help="how to compute loss, cross_entropy or align_energy")
    parser.add_argument("--align_energy_tau", type=float, default=1.0,  help="temperature for energy weights (more uniform when large)")
    
    # --- Incremental Progress Training (Algorithm 1 in Bansal et al, 2022) ---
    parser.add_argument("--ipt_alpha", type=float, default=0.1, help="Weight α for L = (1-α)*L_full + α*L_prog")
    parser.add_argument("--ipt_k_min", type=int, default=1, help="Minimum extra steps k for progressive branch")
    parser.add_argument("--ipt_k_max", type=int, default=None, help="Maximum extra steps k (defaults to T if None)")

    args = parser.parse_args()

    print("Exp name: ", args.exp_name)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    
    if args.seed is not None:
        import random
        import numpy as np

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    def worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    if args.data_root is not None:
        rootdir = args.data_root
    else:
        rootdir = "./data/sudoku"
        
    trainloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=True),
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    testloader = torch.utils.data.DataLoader(
        SudokuDataset(rootdir, train=False),
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    # Initialize wandb
    if not args.no_wandb:
        wandb_config = {
            "exp_name": args.exp_name,
            "save_dir": args.save_dir,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "beta": args.beta,
            "clip_grad_norm": args.clip_grad_norm,
            "eval_freq": args.eval_freq,
            "batchsize": args.batchsize,
            "model": args.model,
            "L": args.L,
            "T": args.T,
            "ch": args.ch,
            "heads": args.heads,
            "N": args.N,
            "gamma": args.gamma,
            "J": args.J,
            "use_omega": args.use_omega,
            "global_omg": args.global_omg,
            "learn_omg": args.learn_omg,
            "init_omg": args.init_omg,
            "nl": args.nl,
            "J_bias": args.J_bias,
            "ksize": args.ksize,
            "bp_steps": args.bp_steps,
        }

        job_id = os.environ.get("PBS_JOBID") or os.environ.get("PJM_JOBID") or "local"
        if args.wandb_run_name is not None:
            wandb_run_name = args.wandb_run_name.format(**vars(args), job_id=job_id)
        else:
            wandb_run_name = f"{wandb_config['exp_name']}_job{job_id}"
        #run_name = args.wandb_run_name if args.wandb_run_name else f"{args.exp_name}_T{args.T}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
        )
    
    # wandb.init(...) の直後に
    run = wandb.run if not args.no_wandb else None

    # save_dirを決める（Run ID 基準）
    if not args.no_wandb:
        if args.save_dir is None:
            args.save_dir = f"./results/{args.exp_name}/{run.id}"
        # W&B上のconfigにも反映（後から見返せるように）
        wandb.config.update({"save_dir": args.save_dir}, allow_val_change=True)
    else:
        if args.save_dir is None:
            current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            args.save_dir = f"./results/{args.exp_name}_{current_time}"
    # if args.save_dir is None:
    #     current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    #     args.save_dir = f"./results/{args.exp_name}_{current_time}"
    jobdir = args.save_dir
    #jobdir = f"runs/{args.exp_name}/"
    
    # Create job directory
    os.makedirs(jobdir, exist_ok=True)

    # jobdir 作成直後が良い
    if not args.no_wandb:
        rd = Path(jobdir)
        # 1) 人間が見つけやすいテキスト
        rd.joinpath("wandb_run.txt").write_text(
            f"id={run.id}\nname={run.name}\nproject={run.project}\n"
            f"entity={run.entity}\nurl={run.url}\n"
        )
        # 2) config 丸ごと保存（yaml優先、yaml無ければjson）
        try:
            import yaml as _yaml
            rd.joinpath("config.yaml").write_text(
                _yaml.safe_dump(wandb.config.as_dict(), sort_keys=True)
            )
        except Exception:
            rd.joinpath("config.json").write_text(
                json.dumps(wandb.config.as_dict(), indent=2)
            )
        # 3) 相互symlink
        try:
            rd.joinpath("wandb_run").symlink_to(Path(run.dir).resolve())
            Path(run.dir).joinpath("results_dir").symlink_to(rd.resolve())
        except Exception as e:
            print(f"[warn] symlink failed: {e}")

    # only compute digit-wise accuracy
    from source.evals.sudoku.evals import compute_board_accuracy
    def compute_acc(net, loader):
        net.eval()
        correct = 0
        total = 0
        correct_input = 0
        total_input = 0
        for X, Y, is_input in loader:
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()

            with torch.no_grad():
                out = net(X, is_input)
            
            _, _, board_accuracy = compute_board_accuracy(out, Y, is_input)
            correct += board_accuracy.sum().item()
            total += board_accuracy.shape[0]
           
            # digit wise input accuracy
            out = out.argmax(dim=-1)
            Y = Y.argmax(dim=-1)
            mask = (1 - is_input).view(out.shape)
            correct_input += ((1 - mask) * (out == Y)).sum().item()
            total_input += (1 - mask).sum().item()

        acc = correct / total
        input_acc = correct_input / total_input
        return acc, input_acc, (total, correct), (total_input, correct_input)
    
    def compute_train_acc(net, loader, max_batches=10):
        """Training accuracy computation (limited batches for efficiency)"""
        net.eval()
        correct, total = 0, 0
        batch_count = 0

        with torch.no_grad():
            for X, Y, is_input in loader:
                if batch_count >= max_batches:
                    break

                X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()
                out = net(X, is_input)

                _, _, board_accuracy = compute_board_accuracy(out, Y, is_input)
                correct += board_accuracy.sum().item()
                total += board_accuracy.shape[0]
                batch_count += 1

        return correct / total if total > 0 else 0.0

    if args.model == "akorn":
        print(
            f"n: {args.N}, ch: {args.ch}, L: {args.L}, T: {args.T}, type of J: {args.J}, J_bias: {args.J_bias}, ksize: {args.ksize}"
        )
        net = SudokuAKOrN(
            n=args.N,
            ch=args.ch,
            L=args.L,
            T=args.T,
            gamma=args.gamma,
            J=args.J,
            use_omega=args.use_omega,
            global_omg=args.global_omg,
            init_omg=args.init_omg,
            learn_omg=args.learn_omg,
            nl=args.nl,
            heads=args.heads, # below the next line, only used for MySudokuAKOrN
            ksize=args.ksize,
            bp_steps=args.bp_steps,
            J_bias=args.J_bias,
        )
    elif args.model == "itrsa":
        net = SudokuTransformer(
            ch=args.ch,
            blocks=args.L,
            heads=args.heads,
            mlp_dim=args.ch * 2,
            T=args.T,
            gta=False,
        )
    else:
        raise NotImplementedError

    net.cuda()

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    # Log model info to wandb
    if not args.no_wandb:
        wandb.log({
            "model/total_params": total_params,
            "model/model_size_mb": total_params * 4 / 1e6  # Assuming float32
        })

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ema = EMA(net, beta=args.beta, update_every=10, update_after_step=100)

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    # Measure speed
    if args.speed_test:
        it_sp = 0
        time_per_iter = []
        import numpy as np
        
    best_acc = 0.0
    best_ema_acc = 0.0
        
    for epoch in range(args.epochs):
        total_loss = 0
        step = 0

        for X, Y, is_input in tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            net.train()
            ema.train()
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()
            # 変数名	   形状（例）       dtype                    内容
            # X 	     [B, 9, 9]	    torch.int32	    整数盤面：空白 = 0, 手がかりの数字 = 1–9
            # Y	         [B, 9, 9, 9]	torch.float32	1-hot 正解：9 クラス（数字 1–9）の one-hot
            # is_input	 [B, 9, 9]	    torch.float32	手がかりマスク：そのマスに与えられた数字があるか（1 or 0）

            if args.speed_test:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            
            if args.loss == "align_energy" and args.L == 1: # 一旦L=1のときのみ考慮
                ret = net(X, is_input, return_xs=True, return_es=True)
                out, xs, es = ret[:3]
                E_all  = torch.stack(es[0])[1:,:] # <- (T+1) * B tensor

                steps = args.T if args.align_energy_steps is None else args.align_energy_steps
                steps = min(steps, E_all.shape[0])
                E_sel = E_all[-steps:, :]                 # [steps, B]
                
                def logits_from_state(net, x_t):
                    # readout: x_t -> c_t,  out: c_t -> [B,9,H,W] を想定
                    readout = net.layers[0][1]           # ← もし違えば適宜変更（例: [0][3] など）
                    c_t = readout(x_t)
                    return net.out(c_t).permute(0, 2, 3, 1)  # [B,H,W,9]

                # readout = net.layers[0][1]
                # c_to_output = lambda c: net.out(c).permute(0, 2, 3, 1) <- lambda にすると勾配流れないんだったかも

                xs_sel = xs[0][-steps:]                   # list of length=steps, each [B,C,H,W]
                interim_outs = [logits_from_state(net, x_t) for x_t in xs_sel]

                Y = Y.argmax(dim=-1).reshape(-1)   # [B, 9, 9, 9] -> [B*9*9] (vector)

                interim_losses = [
                    criterion(out.reshape(-1,9), Y).reshape(X.shape[0],-1).mean(dim=1)
                    for out in interim_outs
                ]
                interim_losses_ten = torch.stack(interim_losses, dim=0) # [steps, B]

                E_center = E_sel - E_sel.mean(dim=0, keepdim=True)
                meas = torch.softmax(-E_center / max(1e-6, args.align_energy_tau), dim=0)  # [steps, B]
                #meas = nn.Softmax(dim=0)(-es_align_mz) # measure, or wights to evaluate energy with, align_energy_steps * B

                loss = (interim_losses_ten * meas).sum(dim=0).mean()
            elif args.loss == "ipt" and args.L == 1: # 一旦L=1のときのみ考慮
                # -------- Incremental Progress Training (Algorithm 1 in Bansal et al, 2022) --------
                # Where T/bp_steps live:
                #   - T は net.T
                #   - bp_steps は 各 KLayer (net.layers[?][0]).bp_steps
                T_total = getattr(net, "T")
                k_max   = args.ipt_k_max or T_total
                k_min   = min(args.ipt_k_min, k_max)
                k       = random.randint(k_min, k_max)                # extra steps
                n_max   = max(0, T_total - k)
                n       = random.randint(0, n_max) if n_max > 0 else 0  # start index
                
                # Progressive branch: run n+k steps, but only last k backprop
                net.T = n + k
                net.layers[0][0].bp_steps = k
                #setattr(net, "bp_steps", k)
                out_prog = net(X, is_input)
                out_prog = out_prog.reshape(-1, 9)
                Y_flat   = Y.argmax(dim=-1).reshape(-1)
                loss_prog = criterion(out_prog, Y_flat).mean()

                # Full branch: run full T with user/default bp_steps
                net.T = T_total
                net.layers[0][0].bp_steps = None
                out_full = net(X, is_input)
                out_full = out_full.reshape(-1, 9)
                loss_full = criterion(out_full, Y_flat).mean()

                # Blend losses
                alpha = args.ipt_alpha
                loss  = (1.0 - alpha) * loss_full + alpha * loss_prog

                # Maybe unnecessary, restore attrs
                net.T = T_total
                net.layers[0][0].bp_steps = None
                #setattr(net, "T", old_T)
                #setattr(net, "bp_steps", None) #old_bp if old_bp is not None else args.bp_steps)

                if not args.no_wandb:
                    wandb.log({
                        "train/loss_full": loss_full.item(),
                        "train/loss_prog": loss_prog.item(),
                        "train/k": k, "train/n": n, "train/T_prog": n + k
                    })


            else:
                out = net(X, is_input)
                out = out.reshape(-1, 9)           # [B, 9, 9, 9] -> [B*9*9, 9]
                Y = Y.argmax(dim=-1).reshape(-1)   # [B, 9, 9, 9] -> [B*9*9] (vector)
                
                loss = criterion(out, Y).mean()    #  -> shape: [B*9*9] (vector) -> scalar
            
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
            optimizer.step()

            if args.speed_test:
                end.record()
                torch.cuda.synchronize()
                time_elapsed_per_iter = start.elapsed_time(end)
                time_per_iter.append(time_elapsed_per_iter)
                print(time_elapsed_per_iter)
                it_sp = it_sp + 1
                if it_sp == 100:
                    np.save(os.path.join(jobdir, "time.npy"), np.array(time_per_iter))
                    exit(0)

            total_loss += loss.item()
            ema.update()
            
            # Log batch-level metrics to wandb
            if not args.no_wandb:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/step": epoch * len(trainloader) + step,
                    "train/epoch": epoch,
                })
            
            step += 1

        total_loss = total_loss / len(trainloader)

        # Log epoch-level training metrics
        log_dict = {"train/epoch_loss": total_loss, "epoch": epoch}
        
        if not args.no_wandb:
            wandb.log(log_dict)
            
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {total_loss:.4f}")

        if (epoch + 1) % args.eval_freq == 0:  # 同じ頻度で評価
            # Training accuracy computation
            train_acc = compute_train_acc(net, trainloader, max_batches=10)
            log_dict.update({"train/accuracy": train_acc})
            print(f"[Train]: Accuracy: {train_acc:.4f}")

            # Test accuracy computation
            acc, input_acc, stats, stats_input = compute_acc(net, testloader)

            # Update best accuracy
            if acc > best_acc:
                best_acc = acc
            
            log_dict.update({
                "test/accuracy": acc,
                "test/input_accuracy": input_acc,
                "test/total_blanks": stats[0],
                "test/correct_blanks": stats[1],
                "test/total_given": stats_input[0],
                "test/correct_given": stats_input[1],
                "test/best_accuracy": best_acc,
            })
            
            print(f"[Test]: Total blanks:{stats[0]}, Accuracy: {acc:.4f}")
            print(f"[Test]: Total given squares:{stats_input[0]}, Accuracy on given digits: {input_acc:.4f}")

            # EMA evals
            ema_acc, ema_input_acc, ema_stats, ema_stats_input = compute_acc(ema.ema_model, testloader)
            
            # Update best EMA accuracy
            if ema_acc > best_ema_acc:
                best_ema_acc = ema_acc
            
            log_dict.update({
                "ema_test/accuracy": ema_acc,
                "ema_test/input_accuracy": ema_input_acc,
                "ema_test/total_blanks": ema_stats[0],
                "ema_test/correct_blanks": ema_stats[1],
                "ema_test/total_given": ema_stats_input[0],
                "ema_test/correct_given": ema_stats_input[1],
                "ema_test/best_accuracy": best_ema_acc,
            })
            
            print(f"[EMA Test]: Total blanks:{ema_stats[0]}, Accuracy: {ema_acc:.4f}")
            print(f"[EMA Test]: Total given squares:{ema_stats_input[0]}, Accuracy on given digits: {ema_input_acc:.4f}")
            
            # Log all evaluation metrics to wandb
            if not args.no_wandb:
                wandb.log(log_dict)

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(net, optimizer, epoch, total_loss, checkpoint_dir=jobdir)
            save_model(ema, epoch, checkpoint_dir=jobdir, prefix="ema")

    # Save final models
    torch.save(net.state_dict(), os.path.join(jobdir, f"model.pth"))
    torch.save(ema.state_dict(), os.path.join(jobdir, f"ema_model.pth"))
    
    # Log final results
    if not args.no_wandb:
        wandb.log({
            "final/best_accuracy": best_acc,
            "final/best_ema_accuracy": best_ema_acc,
            "final/total_params": total_params,
        })
        
        # Save model artifacts to wandb
        model_artifact = wandb.Artifact(f"model_{args.exp_name}", type="model")
        model_artifact.add_file(os.path.join(jobdir, "model.pth"))
        model_artifact.add_file(os.path.join(jobdir, "ema_model.pth"))
        wandb.log_artifact(model_artifact)
        
        wandb.finish()
    
    print(f"Training completed!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Best EMA accuracy: {best_ema_acc:.4f}")
    print(f"Models saved in: {jobdir}")