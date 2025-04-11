import os
import subprocess
import sys

def clone_repositories():
    # Define the paths and repositories
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
    repositories = {
        "MiDaS": "https://github.com/intel-isl/MiDaS.git",
        # Add other depth estimation repositories here
    }
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Clone each repository
    for repo_name, repo_url in repositories.items():
        repo_path = os.path.join(models_dir, repo_name)
        
        if os.path.exists(repo_path):
            print(f"{repo_name} already exists at {repo_path}")
        else:
            print(f"Cloning {repo_name} from {repo_url}...")
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            print(f"Successfully cloned {repo_name}")

if __name__ == "__main__":
    clone_repositories()