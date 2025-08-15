import sys, os
import torch
import torch.nn
import torch.optim
import tqdm
import torchvision
from torchvision import transforms
import numpy as np
from torch.optim.swa_utils import AveragedModel
import matplotlib.pyplot as plt
import wandb

from source.data.datasets.sudoku.sudoku import SudokuDataset, HardSudokuDataset
from source.models.sudoku.knet import SudokuAKOrN
from source.models.sudoku.transformer import SudokuTransformer
from source.evals.sudoku.evals import compute_board_accuracy
from source.utils import str2bool
from ema_pytorch import EMA
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--exp_name", type=str, default="sudoku_eval", help="experiment name for wandb")

    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="sudoku_akorn_eval", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--no_wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")

    # Data loading
    parser.add_argument("--data", type=str, default="id", help="data")
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int, default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int, default=16, help="end core")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
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
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="num of random oscillator samples for each input",
    )
    parser.add_argument("--minimum_chunk", type=int, default=None)
    parser.add_argument("--evote_type", type=str, default="last", help="last or sum")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=True)
    parser.add_argument("--global_omg", type=str2bool, default=True)
    parser.add_argument("--learn_omg", type=str2bool, default=False)
    parser.add_argument("--init_omg", type=float, default=0.1)
    parser.add_argument("--nl", type=str2bool, default=True)

    parser.add_argument("--speed_test", action="store_true")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    # Initialize wandb
    if not args.no_wandb:
        wandb_config = {
            "model_path": args.model_path,
            "exp_name": args.exp_name,
            "data": args.data,
            "batchsize": args.batchsize,
            "model": args.model,
            "L": args.L,
            "T": args.T,
            "ch": args.ch,
            "heads": args.heads,
            "N": args.N,
            "K": args.K,
            "minimum_chunk": args.minimum_chunk,
            "evote_type": args.evote_type,
            "gamma": args.gamma,
            "J": args.J,
            "use_omega": args.use_omega,
            "global_omg": args.global_omg,
            "learn_omg": args.learn_omg,
            "init_omg": args.init_omg,
            "nl": args.nl,
        }
        
        run_name = args.wandb_run_name if args.wandb_run_name else f"{args.exp_name}_{args.data}_K{args.K}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=wandb_config,
        )

    if args.limit_cores_used:

        def worker_init_fn(worker_id):
            os.sched_setaffinity(0, range(args.cpu_core_start, args.cpu_core_end))

    else:
        worker_init_fn = None

    if args.data == "id":
        loader = torch.utils.data.DataLoader(
            SudokuDataset(
                args.data_root if args.data_root is not None else "./data/sudoku",
                train=False,
            ),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        dataset_name = "Standard Sudoku (In-Distribution)"
    elif args.data == "ood":
        loader = torch.utils.data.DataLoader(
            HardSudokuDataset(
                args.data_root if args.data_root is not None else "./data/sudoku-rrn",
                split="test",
            ),
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=worker_init_fn,
        )
        dataset_name = "Hard Sudoku (Out-of-Distribution)"
    else:
        raise NotImplementedError

    if args.model == "akorn":
        print(
            f"n: {args.N}, ch: {args.ch}, L: {args.L}, T: {args.T}, type of J: {args.J}"
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
            heads=args.heads,
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

    model = EMA(net).cuda()
    
    # Load model state dict - handle both checkpoint format and direct state dict format
    checkpoint = torch.load(args.model_path, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Checkpoint format (from save_model function)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Direct state dict format (from final model save)
        model.load_state_dict(checkpoint)
    
    model = model.ema_model
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    print(f"Evaluating on: {dataset_name}")
    print(f"Using K={args.K} samples per puzzle")

    K = args.K

    corrects_vote = 0
    corrects_avg = 0
    totals = 0
    
    # Track additional metrics
    total_blanks = 0
    correct_blanks = 0
    total_given = 0 
    correct_given = 0
    
    # For energy-based voting analysis
    energy_stats = []
    batch_accuracies = []

    minimum_chunk = args.minimum_chunk if args.minimum_chunk is not None else K

    print(f"Starting evaluation...")
    
    for i, (X, Y, is_input) in tqdm.tqdm(enumerate(loader), desc="Evaluating"):
        B = X.shape[0]
        batch_correct_vote = 0
        batch_total = 0
        
        if args.model == 'akorn' and K > 1:  # Energy-based voting
            for j in range(B):
                preds = []
                es_list = []
                for k in range(K//minimum_chunk):
                    
                    _X = X[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _Y = Y[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _is_input = is_input[j : j + 1].repeat(minimum_chunk, 1, 1, 1)
                    _X, _Y, _is_input = (
                        _X.to(torch.int32).cuda(),
                        _Y.cuda(),
                        _is_input.cuda(),
                    )

                    with torch.no_grad():
                        pred, es = model(_X, _is_input, return_es=True)
                        preds.append(pred.detach())
                        if args.evote_type =='sum':
                            # the sum of energy values over timesteps as board correctness indicator 
                            es = torch.stack(es[-1], 0).sum(0).detach()
                        elif args.evote_type == 'last':
                            es = es[-1][-1].detach()
                        es_list.append(es)
                        
                pred = torch.cat(preds, 0)
                es = torch.cat(es_list, 0)
                
                # Store energy statistics
                energy_stats.extend(es.cpu().numpy().tolist())
                
                idxes = torch.argsort(es)  # minimum energy first
                pred_vote = pred[idxes[:1]].mean(0, keepdim=True)
                pred_avg = pred.mean(0, keepdim=True)

                num_blanks, num_corrects, board_correct_vote = compute_board_accuracy(
                    pred_vote, _Y[:1], _is_input[:1]
                )
                
                # Track detailed metrics
                total_blanks += num_blanks.item()
                correct_blanks += num_corrects.item()
                
                # Track given digits accuracy
                pred_digits = pred_vote.argmax(dim=-1)
                true_digits = _Y[:1].argmax(dim=-1)
                given_mask = _is_input[:1].squeeze(-1)
                total_given += given_mask.sum().item()
                correct_given += (given_mask * (pred_digits == true_digits)).sum().item()

                corrects_vote += board_correct_vote.sum().item()
                totals += board_correct_vote.numel()
                
                batch_correct_vote += board_correct_vote.sum().item()
                batch_total += board_correct_vote.numel()
            
        else:
            X, Y, is_input = X.to(torch.int32).cuda(), Y.cuda(), is_input.cuda()
            with torch.no_grad():
                pred = model(X, is_input)
            
            num_blanks, num_corrects, board_correct = compute_board_accuracy(pred, Y, is_input)
            
            # Track detailed metrics
            total_blanks += num_blanks.sum().item()
            correct_blanks += num_corrects.sum().item()
            
            # Track given digits accuracy
            pred_digits = pred.argmax(dim=-1)
            true_digits = Y.argmax(dim=-1)
            given_mask = is_input.squeeze(-1)
            total_given += given_mask.sum().item()
            correct_given += (given_mask * (pred_digits == true_digits)).sum().item()
            
            corrects_vote += board_correct.sum().item()
            totals += board_correct.numel()
            
            batch_correct_vote += board_correct.sum().item()
            batch_total += board_correct.numel()
        
        # Track batch accuracy
        batch_acc = batch_correct_vote / batch_total if batch_total > 0 else 0
        batch_accuracies.append(batch_acc)
        
        # Log batch-level metrics to wandb
        if not args.no_wandb and i % 10 == 0:  # Log every 10 batches
            log_dict = {
                "eval/batch_accuracy": batch_acc,
                "eval/batch_idx": i,
                "eval/running_accuracy": corrects_vote / totals if totals > 0 else 0,
            }
            if energy_stats:
                log_dict["eval/mean_energy"] = np.mean(energy_stats[-minimum_chunk:]) if len(energy_stats) >= minimum_chunk else np.mean(energy_stats)
            wandb.log(log_dict)

    # Compute final metrics
    accuracy_vote = corrects_vote / totals
    blank_accuracy = correct_blanks / total_blanks if total_blanks > 0 else 0
    given_accuracy = correct_given / total_given if total_given > 0 else 0
    
    # Compute statistics
    batch_acc_mean = np.mean(batch_accuracies)
    batch_acc_std = np.std(batch_accuracies)
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model}")
    print(f"K (samples per puzzle): {K}")
    if args.model == 'akorn' and K > 1:
        print(f"Energy voting type: {args.evote_type}")
    print(f"Total puzzles evaluated: {totals}")
    print(f"Board-level accuracy: {accuracy_vote:.4f}")
    print(f"Blank digits accuracy: {blank_accuracy:.4f} ({correct_blanks}/{total_blanks})")
    print(f"Given digits accuracy: {given_accuracy:.4f} ({correct_given}/{total_given})")
    print(f"Batch accuracy mean±std: {batch_acc_mean:.4f}±{batch_acc_std:.4f}")
    
    if energy_stats:
        energy_mean = np.mean(energy_stats)
        energy_std = np.std(energy_stats)
        energy_min = np.min(energy_stats)
        energy_max = np.max(energy_stats)
        print(f"Energy statistics: mean={energy_mean:.4f}, std={energy_std:.4f}, min={energy_min:.4f}, max={energy_max:.4f}")

    # Log final results to wandb
    if not args.no_wandb:
        final_results = {
            "eval/final_accuracy": accuracy_vote,
            "eval/blank_accuracy": blank_accuracy, 
            "eval/given_accuracy": given_accuracy,
            "eval/total_puzzles": totals,
            "eval/correct_puzzles": corrects_vote,
            "eval/total_blanks": total_blanks,
            "eval/correct_blanks": correct_blanks,
            "eval/total_given": total_given,
            "eval/correct_given": correct_given,
            "eval/batch_acc_mean": batch_acc_mean,
            "eval/batch_acc_std": batch_acc_std,
            "model/total_params": total_params,
            "config/dataset": args.data,
            "config/K_samples": K,
            "config/T_eval": args.T,
        }
        
        if energy_stats:
            final_results.update({
                "eval/energy_mean": energy_mean,
                "eval/energy_std": energy_std, 
                "eval/energy_min": energy_min,
                "eval/energy_max": energy_max,
            })
            
            # Create energy distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(energy_stats, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Energy Values')
            plt.ylabel('Frequency')
            plt.title(f'Energy Distribution - {dataset_name}')
            plt.grid(True, alpha=0.3)
            
            # Log plot to wandb
            wandb.log({"eval/energy_distribution": wandb.Image(plt)})
            plt.close()
            
        # Create accuracy distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(batch_accuracies, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Batch Accuracy')
        plt.ylabel('Frequency')
        plt.title(f'Batch Accuracy Distribution - {dataset_name}')
        plt.axvline(batch_acc_mean, color='red', linestyle='--', label=f'Mean: {batch_acc_mean:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log plot to wandb
        wandb.log({"eval/batch_accuracy_distribution": wandb.Image(plt)})
        plt.close()
        
        wandb.log(final_results)

        # Create summary table
        # summary_table = wandb.Table(
        #     columns=["Metric", "Value"],
        #     data=[
        #         ["Board Accuracy", f"{accuracy_vote:.4f}"],
        #         ["Blank Digits Accuracy", f"{blank_accuracy:.4f}"],
        #         ["Given Digits Accuracy", f"{given_accuracy:.4f}"],
        #         ["Total Puzzles", totals],
        #         ["Correct Puzzles", corrects_vote],
        #         ["Dataset", dataset_name],
        #         ["K Samples", K],
        #         ["Model Parameters", f"{total_params:,}"],
        #     ]
        # )
        # wandb.log({"eval/summary_table": summary_table})
        
        wandb.finish()

    print(f"{'='*50}")
    print(f"Evaluation completed!")