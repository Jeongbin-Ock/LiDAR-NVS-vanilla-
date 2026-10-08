"""Build a KITTI-360 raw split manifest without importing datasets or CUDA."""
import argparse
import json
from pathlib import Path

from .splits import build_training_split


def main():
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/kitti360_train_split.yaml")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    manifest = build_training_split(config)
    output = Path(config["manifest_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Manifest: {output.resolve()}")


if __name__ == "__main__":
    main()
