import subprocess
import itertools

batch_sizes = [8, 16]
n_rollouts = [2, 3, 4]

models = [
    "./outputs/sft/checkpoint-200",
    "./outputs/sft/checkpoint-400",
    "./outputs/sft/checkpoint-600",
    "./outputs/sft/checkpoint-800",
    "./outputs/sft/checkpoint-1000"
]

for bs, roll, model in itertools.product(batch_sizes, n_rollouts, models):

    cmd = [
        "python",
        "rl/grpo/train.py",
        "--batch_size", str(bs),
        "--n_rollouts", str(roll),
        "--model_name", model,
    ]

    print("Running:", " ".join(cmd))

    subprocess.run(cmd)