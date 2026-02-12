import os
import sys
from pathlib import Path
import httpx
import tarfile

# Set env vars for testing / downloading
# User specified paths
os.environ["AF2_CACHE_DIR"] = "/scratch/rd34/share/af2"
os.environ["PROTENIX_CACHE_DIR"] = "/scratch/rd34/share/protenix"
os.environ["BOLTZ_CACHE_DIR"] = "/scratch/rd34/share/boltz"

def download_af2(data_dir_str: str):
    print(f"Checking AF2 in {data_dir_str}...")
    data_dir = Path(data_dir_str).expanduser()
    if (data_dir / "params" / "alphafold_params_2022-12-06.tar").exists():
         print("AF2 params already exist (tar file found). Skipping.")
         return
    
    # Check if extracted
    # A bit loose check, but matches af2.py logic roughly
    if (data_dir / "params" / "params_model_1.npz").exists():
         print("AF2 params already extracted. Skipping.")
         return

    print(f"Downloading AF2 parameters to {data_dir}/params...")
    params_dir = data_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    tar_path = params_dir / "alphafold_params_2022-12-06.tar"

    try:
        with httpx.stream("GET", url, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                total = 0
                for chunk in r.iter_bytes():
                    f.write(chunk)
                    total += len(chunk)
                    if total % (100 * 1024 * 1024) == 0:
                        print(f"Downloaded {total / 1024 / 1024:.0f} MB...")
        
        print("Extracting AF2 params...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=params_dir)
        # We don't unlink the tar path in this script to be safe/idempotent-ish
        # or we follow af2.py which unlinks it. 
        # af2.py unlinks it. I'll unlink it to save space.
        tar_path.unlink()
        print("AF2 download complete.")
    except Exception as e:
        print(f"Error downloading AF2: {e}")

def download_boltz(cache_dir_str: str):
    print(f"Checking Boltz in {cache_dir_str}...")
    cache_path = Path(cache_dir_str).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Boltz1
        from boltz.main import download_boltz1
        print("Targeting Boltz1 download...")
        # download_boltz1 takes the *cache directory*
        download_boltz1(cache_path)
    except ImportError:
        print("Could not import boltz.main.download_boltz1. Is boltz installed?")
    except Exception as e:
        print(f"Error downloading Boltz1: {e}")

    try:
        # Boltz2
        from boltz.main import download_boltz2
        print("Targeting Boltz2 download...")
        download_boltz2(cache_path)
    except ImportError:
        pass # Optional?
    except Exception as e:
        print(f"Error downloading Boltz2: {e}")

def download_file(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        print(f"File {path.name} already exists. Skipping.")
        return
    
    print(f"Downloading {url} to {path}...")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                total = 0
                for chunk in r.iter_bytes():
                    f.write(chunk)
                    total += len(chunk)
                    if total % (50 * 1024 * 1024) == 0:
                        print(f"Downloaded {total / 1024 / 1024:.0f} MB...")
        print(f"Finished {path.name}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_protenix(cache_dir_str: str):
    print(f"Checking Protenix in {cache_dir_str}...")
    cache_path = Path(cache_dir_str).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    # Manual download list
    base_data_url = "https://af3-dev.tos-cn-beijing.volces.com/release_data"
    base_model_url = "https://af3-dev.tos-cn-beijing.volces.com/release_model"

    data_files = [
        "components.v20240608.cif",
        "components.v20240608.cif.rdkit_mol.pkl",
        "clusters-by-entity-40.txt"
    ]

    models_to_download = [
        "protenix_mini_default_v0.5.0",
        "protenix_tiny_default_v0.5.0",
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0"
    ]

    print("Downloading Protenix data files...")
    for filename in data_files:
        url = f"{base_data_url}/{filename}"
        download_file(url, cache_path / filename)

    print("Downloading Protenix models...")
    for model_name in models_to_download:
        filename = f"{model_name}.pt"
        url = f"{base_model_url}/{filename}"
        download_file(url, cache_path / filename)

    # Attempt to override defaults in memory just in case, but rely on manual download
    os.environ["PROTENIX_DATA_ROOT_DIR"] = str(cache_path)

if __name__ == "__main__":
    download_af2(os.environ["AF2_CACHE_DIR"])
    download_boltz(os.environ["BOLTZ_CACHE_DIR"])
    download_protenix(os.environ["PROTENIX_CACHE_DIR"])
