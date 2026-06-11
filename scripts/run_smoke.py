from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residual_rl0611.data import sine_reference
from residual_rl0611.evaluate import evaluate_zero_residual
from residual_rl0611.train import train_short


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    reference = sine_reference(args.steps + 1, amplitude=1.0, frequency_hz=3.0)
    baseline = evaluate_zero_residual(reference, disturbance_mode="none", steps=args.steps)
    disturbed = evaluate_zero_residual(reference, disturbance_mode="impulse", steps=args.steps)
    disturbance_obs = evaluate_zero_residual(
        reference,
        use_disturbance_obs=True,
        disturbance_mode="impulse",
        steps=args.steps,
    )
    print("PID/no_disturbance", baseline)
    print("PID/impulse", disturbed)
    print("PID/impulse/disturbance_obs_shape_path", disturbance_obs)
    if args.train:
        print("short_train", train_short(use_disturbance_obs=True, disturbance_mode="impulse", steps=args.steps))


if __name__ == "__main__":
    main()
