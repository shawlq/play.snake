"""串行训练三套 neuro player，然后写出对比报告。"""
from __future__ import annotations

import os
import time

from player.neuro.fly_player_fixed_brain import FlyFixedBrainPlayer
from player.neuro.fly_player_plastic_brain import FlyPlasticBrainPlayer
from player.neuro.fly_player_train_encoder import FlyTrainEncoderPlayer


def main():
    os.makedirs("logs/neuro", exist_ok=True)
    jobs = [
        ("flyfixed", FlyFixedBrainPlayer),
        ("flyplastic", FlyPlasticBrainPlayer),
        ("flyenc", FlyTrainEncoderPlayer),
    ]
    timings = {}
    for name, cls in jobs:
        print(f"\n===== TRAIN {name} =====")
        t0 = time.time()
        cls.train(new_model=True)
        timings[name] = time.time() - t0
        print(f"===== DONE {name} in {timings[name]:.1f}s =====")

    print("\n===== COMPARE EVAL =====")
    from player.neuro import compare_eval

    compare_eval.main()
    print("timings:", timings)


if __name__ == "__main__":
    main()
