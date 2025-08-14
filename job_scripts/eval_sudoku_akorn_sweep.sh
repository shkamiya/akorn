#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N sudoku_exp
#PBS -o logs/
#PBS -e logs/
#PBS -j oe
#PBS -W group_list=gj26

module purge
module load singularity

# --- ログは $PBS_O_WORKDIR に出る ---
cd $PBS_O_WORKDIR

export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65
export WANDB_PROJECT=sudoku_exp

TODAY=$(date '+%Y%m%d')

# --- sweep結果の取得と評価 ---
# results/sudoku_sweep_altered_loss_* ディレクトリを検索
for result_dir in results/sudoku_sweep_altered_loss_*; do
    if [ -d "$result_dir" ]; then
        echo "Processing: $result_dir"
        
        # 対応するwandbディレクトリを探す
        timestamp=$(basename "$result_dir" | sed 's/sudoku_sweep_altered_loss_//')
        wandb_dir=$(find wandb -name "run-${timestamp:0:8}_*" -type d | head -1)
        
        if [ -n "$wandb_dir" ] && [ -d "$wandb_dir" ]; then
            echo "Found wandb config: $wandb_dir"
            
            # config.yamlからパラメータを抽出
            config_file="$wandb_dir/files/config.yaml"
            if [ -f "$config_file" ]; then
                L=$(grep "^L:" "$config_file" | awk '{print $2}')
                N=$(grep "^\"N\":" "$config_file" | awk '{print $2}')
                T=$(grep "^T:" "$config_file" | awk '{print $2}')
                ch=$(grep "^ch:" "$config_file" | awk '{print $2}')
                loss=$(grep "^loss:" "$config_file" | awk '{print $2}')
                
                echo "Parameters: L=$L, N=$N, T=$T, ch=$ch, loss=$loss"
                
                # 各モデルファイルについて評価を実行
                for model_file in "$result_dir"/ema_*.pth; do
                    if [ -f "$model_file" ]; then
                        model_name=$(basename "$model_file" .pth)
                        echo "Evaluating $model_name with parameters L=$L, N=$N, T=$T, ch=$ch"
                        
                        singularity exec --nv \
                          --bind $(pwd):/workspace \
                          --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/pki/tls/certs/ca-bundle.crt \
                          ~/singularity/kamiya_miyabi.sif \
                          python scripts/eval_sudoku.py \
                              --model_path "$model_file" \
                              --model akorn \
                              --L "$L" --T "$T" --ch "$ch" --N "$N" \
                              --data ood --K 1 \
                              --exp_name "eval_${model_name}_T${T}_loss${loss}" 2>&1 | tee "logs/eval_${model_name}_T${T}_loss${loss}.log"
                        
                        eval_status=$?
                        if [ $eval_status -ne 0 ]; then
                            echo "Evaluation failed for $model_file"
                        else
                            echo "Evaluation completed for $model_file"
                        fi
                    fi
                done
            else
                echo "Config file not found: $config_file"
            fi
        else
            echo "Corresponding wandb directory not found for $result_dir"
        fi
        echo "---"
    fi
done

STATUS=$?   # 0=正常, それ以外=異常

# ---- Slack 通知 ----

JOB_NAME=$PBS_JOB_NAME
JOB_ID=$PBS_JOBID
NODE_NAME=$(hostname)

send_slack() {         # 小さなヘルパー関数
  curl -s -X POST -H 'Content-type: application/json' \
       --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK"
}

if [ "$STATUS" -eq 0 ]; then
    MESSAGE="✅ *Job Finished Successfully*\n> Job Name: \`$JOB_NAME\`\n> Job ID: \`$JOB_ID\`\n> Node: \`$NODE_NAME\`"
    send_slack "$MESSAGE"
else
    MESSAGE="❌ *Job Failed*\n> Job Name: \`$JOB_NAME\`\n> Job ID: \`$JOB_ID\`\n> Node: \`$NODE_NAME\`\n> Exit Code: \`$STATUS\`"
    send_slack "$MESSAGE"
fi