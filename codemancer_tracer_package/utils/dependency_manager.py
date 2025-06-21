import subprocess
import sys
import os
import json
from typing import Optional

def detect_cuda():
    """Detects if a CUDA-capable GPU is available and returns (cuda_available, device_name)."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
        return cuda_available, device_name
    except Exception as e:
        print(f"[!] Could not import torch to check CUDA: {e}")
        return False, None

def install_dependencies(gpu: bool = False, auto_cuda: bool = True, gpu_series: Optional[str] = None):
    """Installs all dependencies listed in requirements.txt. Installs GPU-enabled torch if gpu=True, CODEMANCER_GPU=1, or CUDA is detected. gpu_series can be '10' or '20+'."""
    # Auto-detect CUDA if requested
    cuda_available, device_name = (False, None)
    if auto_cuda:
        try:
            import importlib.util
            if importlib.util.find_spec('torch'):
                import torch
                cuda_available = torch.cuda.is_available()
                device_name = torch.cuda.get_device_name(0) if cuda_available else None
        except Exception:
            pass
    # Determine series
    series = None
    if gpu_series:
        series = gpu_series
    elif device_name:
        if any(x in device_name for x in ['RTX 20', 'RTX 30', 'RTX 40', 'A', 'H', 'L40', 'L4', 'L20', 'L40S', 'L4S']):
            series = '20+'
        else:
            series = '10'
    else:
        series = '10'
    # Map series to torch/torchvision/torchaudio and index-url
    cuda_versions = {
        '10': {
            'torch': 'torch==2.5.1+cu121',
            'torchvision': 'torchvision==0.20.1+cu121',
            'torchaudio': 'torchaudio==2.5.1+cu121',
            'index_url': 'https://download.pytorch.org/whl/cu121',
            'label': 'Nvidia 10 series (CUDA 12.1)',
            'extra': [
                'numpy<2',
                'transformers==4.52.4'
            ],
            'wheels': {
                'torch': 'https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl',
                'torchvision': 'https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl',
                'torchaudio': 'https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl'
            }
        },
        '20+': {
            'torch': 'torch==2.5.1+cu122',
            'torchvision': 'torchvision==0.20.1+cu122',
            'torchaudio': 'torchaudio==2.5.1+cu122',
            'index_url': 'https://download.pytorch.org/whl/cu122',
            'label': 'Nvidia 20/30/40 series (CUDA 12.2)',
            'extra': [
                'numpy<2',
                'transformers==4.52.4'
            ],
            'wheels': {
                'torch': 'https://download.pytorch.org/whl/cu122/torch-2.5.1%2Bcu122-cp311-cp311-win_amd64.whl',
                'torchvision': 'https://download.pytorch.org/whl/cu122/torchvision-0.20.1%2Bcu122-cp311-cp311-win_amd64.whl',
                'torchaudio': 'https://download.pytorch.org/whl/cu122/torchaudio-2.5.1%2Bcu122-cp311-cp311-win_amd64.whl'
            }
        }
    }
    gpu_mode = gpu or os.environ.get("CODEMANCER_GPU", "0") == "1" or cuda_available
    if gpu_mode:
        config = cuda_versions[series]
        print(f"[*] Uninstalling conflicting torch/vision/audio and extras before installing {config['label']}...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio", "numpy", "transformers"])
        print(f"[*] Installing {config['label']} torch/torchvision/torchaudio...")
        try:
            wheels = config.get('wheels', None)
            if wheels:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    wheels['torch'], wheels['torchvision'], wheels['torchaudio']
                ], check=True, capture_output=True)
                print(result.stdout.decode())
                print(result.stderr.decode())
            else:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install",
                    config['torch'], config['torchvision'], config['torchaudio'],
                    "--index-url", config['index_url']
                ], check=True, capture_output=True)
                print(result.stdout.decode())
                print(result.stderr.decode())
            if 'extra' in config:
                result = subprocess.run([sys.executable, "-m", "pip", "install"] + config['extra'], check=True, capture_output=True)
                print(result.stdout.decode())
                print(result.stderr.decode())
            print(f"[*] Successfully installed {config['label']} torch/vision/audio and extras.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to install CUDA-enabled torch packages or extras: {e}")
            print(e.stdout.decode() if e.stdout else "")
            print(e.stderr.decode() if e.stderr else "")
            sys.exit(1)
        # Double-check torch is importable after install
        try:
            import importlib
            importlib.import_module('torch')
        except ImportError:
            print("[!] torch is still not importable after install! Exiting.")
            sys.exit(1)
    try:
        with open("requirements.txt", "r") as f:
            dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("[!] requirements.txt not found. Please ensure it's in the root directory.")
        sys.exit(1)
    for dep in dependencies:
        # Skip torch if in GPU mode (already installed above)
        if gpu_mode and dep.startswith("torch"):
            continue
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
    # After installing dependencies, write CUDA info for double validation
    write_cuda_info_json()

def write_cuda_info_json(path="cuda_info.json"):
    """Write torch and CUDA info to a JSON file for double validation."""
    try:
        import torch
        info = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except Exception as e:
        info = {
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "device_name": None,
            "error": str(e)
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"[*] CUDA info written to {path}")

def cycle_cuda_install(max_attempts=3, series_hint=None):
    """Attempt to uninstall and reinstall CUDA-enabled torch for the detected Nvidia series until a CUDA device is detected or max_attempts is reached.
    series_hint: '10' for 10-series (GTX 1070, etc), '20+' for RTX 20/30/40 series, or None for auto.
    """
    # Map series to torch/torchvision/torchaudio and index-url
    cuda_versions = {
        '10': {
            'torch': 'torch==2.5.1+cu121',
            'torchvision': 'torchvision==0.20.1+cu121',
            'torchaudio': 'torchaudio==2.5.1+cu121',
            'index_url': 'https://download.pytorch.org/whl/cu121',
            'label': 'Nvidia 10 series (CUDA 12.1)'
        },
        '20+': {
            'torch': 'torch==2.5.1+cu122',
            'torchvision': 'torchvision==0.20.1+cu122',
            'torchaudio': 'torchaudio==2.5.1+cu122',
            'index_url': 'https://download.pytorch.org/whl/cu122',
            'label': 'Nvidia 20/30/40 series (CUDA 12.2)'
        }
    }
    # Auto-detect series if not provided
    detected_series = '10'
    device_name = None
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if any(x in device_name for x in ['RTX 20', 'RTX 30', 'RTX 40', 'A', 'H', 'L40', 'L4', 'L20', 'L40S', 'L4S']):
                detected_series = '20+'
    except Exception:
        pass
    series = series_hint or detected_series
    config = cuda_versions[series]
    print(f"[*] Using {config['label']} install config for CUDA-enabled torch.")
    for attempt in range(1, max_attempts + 1):
        print(f"[*] CUDA install attempt {attempt}/{max_attempts}")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            config['torch'], config['torchvision'], config['torchaudio'],
            "--index-url", config['index_url']
        ])
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[*] CUDA device detected: {torch.cuda.get_device_name(0)}")
                return True
            else:
                print("[!] CUDA not detected after install. Retrying...")
        except Exception as e:
            print(f"[!] Error importing torch after install: {e}")
    print("[!] Failed to detect CUDA device after multiple attempts.")
    return False