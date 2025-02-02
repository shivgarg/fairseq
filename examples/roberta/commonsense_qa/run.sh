MAX_UPDATES=15000      # Number of training steps.
WARMUP_UPDATES=150    # Linearly increase LR over this many steps.
LR=1e-05              # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8      # Batch size.
SEED=1                # Random seed.
ROBERTA_PATH=roberta.base/model.pt
DATA_DIR=data/CommonsenseQA/gpt_10ep_greedy
CKPT_DIR=gpt_10ep_greedy

# we use the --user-dir option to load the task from

# the examples/roberta/commonsense_qa directory:

FAIRSEQ_PATH=/scratch/cluster/shivgarg/nlp_project/roberta-cosqa/fairseq
FAIRSEQ_USER_DIR=${FAIRSEQ_PATH}/examples/roberta/commonsense_qa


CUDA_VISIBLE_DEVICES=0 fairseq-train --ddp-backend=no_c10d \
	$DATA_DIR \
	--save-dir $CKPT_DIR \
	--user-dir $FAIRSEQ_USER_DIR \
	--restore-file $ROBERTA_PATH \
	--reset-optimizer --reset-dataloader --reset-meters \
	--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
	--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
	--task commonsense_qa --init-token 0 --bpe gpt2 \
	--arch roberta_base --max-positions 512 \
	--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
	--criterion sentence_ranking --num-classes 5 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
	--lr-scheduler polynomial_decay --lr $LR \
	--warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
	--max-sentences $MAX_SENTENCES \
	--max-update $MAX_UPDATES \
	--log-format simple --log-interval 25 \
	--seed $SEED
