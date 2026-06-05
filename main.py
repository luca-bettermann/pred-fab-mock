"""Full PFAB journey on a simulated robotic extrusion printer.

The agent calibrates two process parameters (water ratio, print speed) for a
fixed design intent, learning to print on-target from synthetic sensor data:

    Baseline → Initial Training → Exploration → Inference

All complexity lives in `PrintingShowcase`; this file is just the storyline.
"""

from showcase import PrintingShowcase


def main() -> None:
    showcase = PrintingShowcase(
        design_intent={"design": "B", "material": "reinforced"},
        bounds={"water_ratio": (0.30, 0.50), "print_speed": (20.0, 60.0)},
        performance_weights={"path_accuracy": 0.75, "energy_efficiency": 0.25},
    )

    showcase.baseline(n=10)   # space-filling experiments to seed the dataset
    showcase.train()          # fit the prediction model
    showcase.explore(rounds=8)  # information-driven sampling, re-training each round
    showcase.infer(rounds=3)    # exploit the model for optimal parameters
    showcase.plots()          # render the figures


if __name__ == "__main__":
    main()
