
# !/usr/bin/env python3
# automates the creation of new experiment dirs with base templates
import argparse
from pathlib import Path
from datetime import datetime
import shutil

EXPERIMENT_BASE = Path("experiments/base")
EXPERIMENT_ROOT = Path("experiments")

SKY_TEMPLATE = """\
resources:
  accelerators: A100:1
  cloud: lambda
  disk_size: 100

envs:
  WANDB_API_KEY: ${{WANDB_API_KEY}}
  AWS_ACCESS_KEY_ID: ${{AWS_ACCESS_KEY_ID}}  
  AWS_SECRET_ACCESS_KEY: ${{AWS_SECRET_ACCESS_KEY}}
  AWS_DEFAULT_REGION: ${{AWS_DEFAULT_REGION}}
  S3_BUCKET: ${{S3_BUCKET}}

setup: |
  
  # Install AWS CLI for S3 syncing
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install

  # Clone repo
  git clone https://${{GITHUB_TOKEN}}@github.com/grad-ient/PoorMansNanoGPT.git
  cd PoorMansNanoGPT
  git checkout add_tiny_stories

  # Install requirements
  pip install -r requirements.txt

  # Prepare data if needed
  python data/tinystories/prepare.py

run: |
  python train.py config/train_tinystories_char.py --out_dir=out

  aws s3 sync out s3://${{S3_BUCKET}}/experiments/001-tinystories-a100-spot
"""

README_TEMPLATE = """\
# Experiment: {name}

**Date Created**: {date}

## Hypothesis
<!-- What are we testing with this experiment? -->

## Configuration Changes
- Primary parameter overrides:
  - batch_size: 64
  - learning_rate: 0.001

## Results
<!-- Add conclusions after experiment completes -->
"""

TEMPLATES = {
    "sky.yaml": SKY_TEMPLATE,
    "README.md": README_TEMPLATE
}

def create_new_experiment(name: str):
    """Create new experiment directory with template files"""
    exp_path = EXPERIMENT_ROOT / name
    
    try:
        exp_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(f"Experiment {name} already exists!")

    # Create template files
    for fname, content in TEMPLATES.items():
        with open(exp_path / fname, "w") as f:
            formatted = content.format(
                name=name,
                date=datetime.now().strftime("%Y-%m-%d")
            )
            f.write(formatted)

    print(f"Created new experiment: {exp_path}")
    print(f"Next steps:")
    print(f"1. Edit {exp_path/'config_overrides.yaml'}")
    print(f"2. Configure {exp_path/'sky.yaml'} resources")
    print(f"3. Launch with: sky spot launch -n {name} --workdir {exp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new experiment template")
    parser.add_argument(
        "--name", 
        required=True,
        help="Experiment name (e.g. 003-shakespeare-a100-spot)"
    )
    args = parser.parse_args()
    
    create_new_experiment(args.name)