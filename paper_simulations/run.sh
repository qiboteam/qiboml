#!/usr/bin/env bash

python execute.py   --backend "qiboml" \
                    --platform "pytorch" \
                    --n_runs 3 \
                    --nepochs 50 \
                    --seed 42 \
                    --with_noise true \
                    --with_mitigation true \
                    --nshots 1024 \
