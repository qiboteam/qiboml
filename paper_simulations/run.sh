#!/usr/bin/env bash

#SBATCH --job-name=qiboml
#SBATCH --output=qiboml_torch.out

python execute.py   --backend "qiboml" \
                    --n_runs 10 \
                    --nepochs 100 \
                    --seed 42 \
                    --with_noise false \
                    --with_mitigation false \
                    --differentiation_rule "Jax" \
                    --platform "jax" \
                    # --nshots 1024 \


