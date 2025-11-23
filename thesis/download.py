import wandb
import os

# Initialize the W&B API
api = wandb.Api()

# Specify the run path
run_path = "fluffpoff-technical-university-of-berlin/maml-trpo_ML10/runs/vtmni7hb"
run = api.run(run_path)

# Create a directory to download the files to
output_dir = f"./{run.name}_files"
os.makedirs(output_dir, exist_ok=True)

# Iterate through the files and download them
for file in run.files():
    file.download(root=output_dir)

print(f"Files downloaded to {output_dir}")