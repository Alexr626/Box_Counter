import subprocess
import os

def run_midas_depth_estimation(image_path, output_path):
    """Run MiDaS depth estimation in its dedicated environment"""
    script_path = "src/depth_map_estimation/midas.py"
    
    # Use full path to conda executable
    conda_path = "/Users/alex/opt/anaconda3/bin/conda"

    # Get current directory for PYTHONPATH
    current_dir = os.getcwd()
    
    # Set up environment with PYTHONPATH
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{current_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = current_dir
    
    cmd = [
        conda_path, "run", "-n", "CSCI677", "python",
        script_path,
        "--input", image_path,
        "--output", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running MiDaS: {result.stderr}")
    return output_path

if __name__ == "__main__":
    # Example usage
    image_path = "data/images/examples/many_boxes/04-0207-02/1738952451425_898d63ef-1635-4427-bba8-762ffd30d1c1.png"
    output_path = "data/images/depth_maps/04-0207-02/1738952451425_898d63ef-1635-4427-bba8-762ffd30d1c1.png"
    run_midas_depth_estimation(image_path, output_path)