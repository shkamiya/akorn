#!/usr/bin/env python3
import os, sys, glob, subprocess, shlex
from pathlib import Path
import itertools, re
import wandb

try:
    import yaml
except ImportError:
    print("pip install pyyaml を実行してください", file=sys.stderr); sys.exit(1)

def parse_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def basename_id_from_results_dir(d: Path):
    # ..._YYYYMMDDHHMMSS_<runid> の <runid> を取得（無い場合は None）
    name = d.name
    if "_" in name:
        rid = name.split("_")[-1]
        if re.fullmatch(r"[a-z0-9]{8}", rid):  # wandb run id 形式
            return rid
    return None

def find_wandb_dir(run_id: str, wandb_root="wandb"):
    hits = list(Path(wandb_root).glob(f"run-*-{run_id}"))
    if len(hits) == 1:
        return hits[0]
    elif len(hits) == 0:
        return None
    else:
        # 万一複数ヒットした場合は最も新しいもの
        return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]

def load_train_config(wandb_dir: Path):
    cfg = {}
    cfg_yaml = wandb_dir / "files" / "config.yaml"
    if cfg_yaml.exists():
        y = yaml.safe_load(cfg_yaml.read_text())
        # W&Bの {key:{value:...}} をフラット化
        for k, v in (y or {}).items():
            cfg[k] = v.get("value", v) if isinstance(v, dict) else v
    return cfg

def canon_bool(x):
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    return s in ("1","true","t","yes","y")

def compute_T_eval_list(T_train, specs):
    out = []
    for s in specs:
        if isinstance(s, int):
            out.append(s)
        else:
            m = re.fullmatch(r"x(\d+)", str(s).lower())
            if m: out.append(int(m.group(1)) * int(T_train))
            else:
                try: out.append(int(s))
                except: raise ValueError(f"T_eval spec not understood: {s}")
    # 重複除去＆整列
    return sorted(set(out))

def run_eval(model_path, arch, evalp, wandb_opt, run_name_base, extra_flags):
    cmd = [
        "python", "eval_sudoku_wandb.py",
        "--model_path", str(model_path),
        "--model", "akorn",
        "--L", str(arch["L"]), "--T", str(evalp["T_eval"]), "--ch", str(arch["ch"]), "--N", str(arch["N"]),
        "--gamma", str(arch.get("gamma", 0.01)), "--J", str(arch.get("J","attn")),
        "--use_omega", str(canon_bool(arch.get("use_omega", True))).lower(),
        "--global_omg", str(canon_bool(arch.get("global_omg", True))).lower(),
        "--learn_omg", str(canon_bool(arch.get("learn_omg", False))).lower(),
        "--data", str(evalp.get("data","ood")),
        "--K", str(evalp["K"]),
        "--evote_type", str(evalp.get("evote_type","last")),
    ]
    if wandb_opt["use_wandb"]:
        cmd += ["--wandb_project", wandb_opt["wandb_project"], "--wandb_entity", wandb_opt["wandb_entity"],
                "--wandb_run_name", f"{run_name_base}_Teval{evalp['T_eval']}_K{evalp['K']}_{evalp['evote_type']}"]
    else:
        cmd += ["--no_wandb"]
    cmd += extra_flags
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd).returncode


def extract_meta(train_cfg):
    # 学習時だけ意味があるものはここへ
    meta = {}
    meta["loss"] = train_cfg.get("loss", None)
    # 表記ゆれに対応
    meta["ipt_alpha"] = (train_cfg.get("ipt_alpha")
                         or train_cfg.get("alpha")
                         or train_cfg.get("ipt.alpha"))
    return meta

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML file for eval sweep")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = parse_yaml(args.config)
    results_glob = cfg["results_glob"]
    ckpts = cfg.get("ckpts", ["ema_99.pth","ema_model.pth"])
    use_wandb = bool(cfg.get("use_wandb", True))
    wandb_project = cfg.get("wandb_project","sudoku_eval_sweep")
    wandb_entity  = cfg.get("wandb_entity","shunsuke-kamiya-the-university-of-tokyo")
    extra_flags = list(map(str, cfg.get("extra_flags", [])))

    E = cfg["eval_params"]
    grid = {
        "T_eval": E.get("T_eval", ["x2"]),
        "K": E.get("K", [1]),
        "evote_type": E.get("evote_type", ["last"]),
        "data": E.get("data", ["ood"]),
    }

    # 結果ディレクトリ列挙
    results_dirs = sorted(Path(p) for p in glob.glob(results_glob) if Path(p).is_dir())
    if not results_dirs:
        print(f"no results matched: {results_glob}", file=sys.stderr); sys.exit(2)

    for rdir in results_dirs:
        run_id = basename_id_from_results_dir(rdir)
        if not run_id:
            print(f"[skip] no run_id in name: {rdir}", file=sys.stderr); continue
        wdir = find_wandb_dir(run_id)
        if not wdir:
            print(f"[skip] wandb dir not found for run_id={run_id}", file=sys.stderr); continue

        train_cfg = load_train_config(wdir)
        # 学習時Tなどのアーキパラ（不足は既定で補完）
        arch = dict(
            L   = train_cfg.get("L"),#, 1),
            N   = train_cfg.get("N"),# 4),
            ch  = train_cfg.get("ch"),# 512),
            gamma = train_cfg.get("gamma"),
            J   = train_cfg.get("J"),#, "attn"),
            use_omega = train_cfg.get("use_omega", True),
            global_omg = train_cfg.get("global_omg", True),
            learn_omg = train_cfg.get("learn_omg", False),
        )
        T_train = int(train_cfg.get("T", 8))
        T_evals = compute_T_eval_list(T_train, grid["T_eval"])

        meta = extract_meta(train_cfg)

        # run名に付ける（存在する時だけ）
        suffix = ""
        if meta["loss"] is not None: suffix += f"_loss{meta['loss']}"
        if meta["ipt_alpha"] is not None: suffix += f"_a{meta['ipt_alpha']}"

        run_name_base = f"eval_{rdir.name}{suffix}"

        # W&Bのconfig/summaryにも刻む
        # if use_wandb:
        #     wandb.config.update(
        #         {"train_loss": meta["loss"], "ipt_alpha": meta["ipt_alpha"]},
        #         allow_val_change=True
        #     )

        # ckpt × (T_eval, K, evote_type, data)
        for ck in ckpts:
            model_path = rdir / ck
            if not model_path.exists():
                print(f"[skip] ckpt not found: {model_path}", file=sys.stderr); continue

            for Te, K, ev, dd in itertools.product(T_evals, grid["K"], grid["evote_type"], grid["data"]):
                evalp = {"T_eval": Te, "K": K, "evote_type": ev, "data": dd}
                if args.dry_run:
                    print(f"DRY: {model_path.name} Teval={Te} K={K} evote={ev} data={dd}")
                    continue
                rc = run_eval(model_path, arch, evalp,
                              {"use_wandb": use_wandb, "wandb_project": wandb_project, "wandb_entity": wandb_entity},
                              run_name_base, extra_flags)
                if rc != 0:
                    print(f"[warn] eval failed: {model_path} Teval={Te} K={K}", file=sys.stderr)

if __name__ == "__main__":
    main()
