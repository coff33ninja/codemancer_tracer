import subprocess
import sys

def install_dependencies():
    """Installs all dependencies listed in requirements.txt."""
    try:
        with open("requirements.txt", "r") as f:
            dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("[!] requirements.txt not found. Please ensure it's in the root directory.")
        sys.exit(1)

    for dep in dependencies:
        try:
            # Attempt to import to check if installed
            __import__(dep.split('==')[0].split('<')[0].split('>')[0].split('~')[0])
        except ImportError:
            print(f"[*] Installing {dep}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
                print(f"[*] Successfully installed {dep}.")
            except subprocess.CalledProcessError as e:
                print(f"[!] Failed to install {dep}: {e.stderr.decode()}")
                sys.exit(1)