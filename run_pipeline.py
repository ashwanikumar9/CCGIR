"""
run_pipeline.py - CCGIR Complete Setup & Execution Pipeline
============================================================

This script automates the entire CCGIR setup and execution process:
  1. Install Python dependencies from requirements.txt
  2. Install Node.js Solidity parser (@solidity-parser/parser)
  3. Download NLTK data (punkt, wordnet, averaged_perceptron_tagger)
  4. Download CodeBERT model from HuggingFace (if not present)
  5. Extract vector store from RAR archive (if not already extracted)
  6. Run the main CCGIR retrieval pipeline

Usage:
  python run_pipeline.py              # Full pipeline: setup + run CCGIR.py
  python run_pipeline.py --setup-only # Only setup, don't run CCGIR
  python run_pipeline.py --run-only   # Skip setup, only run CCGIR.py
  python run_pipeline.py --app        # Setup + run Flask web app (app.py)
"""

import os
import sys
import subprocess
import argparse
import time


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
CODEBERT_DIR = os.path.join(MODEL_DIR, "codebert")
VECTOR_STORE_PKL = os.path.join(MODEL_DIR, "code_vector_whitening.pkl")
VECTOR_STORE_RAR = os.path.join(MODEL_DIR, "code_vector_whitening.rar")
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def banner(msg):
    width = 70
    print(f"\n{Colors.CYAN}{'=' * width}")
    print(f"  {Colors.BOLD}{msg}{Colors.RESET}{Colors.CYAN}")
    print(f"{'=' * width}{Colors.RESET}\n")


def step(number, total, msg):
    print(f"{Colors.BOLD}[{number}/{total}]{Colors.RESET} {msg}")


def success(msg):
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {msg}")


def warn(msg):
    print(f"  {Colors.YELLOW}[!!]{Colors.RESET} {msg}")


def fail(msg):
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")


def run_cmd(cmd, cwd=None, check=True, capture=False):
    """Run a shell command with optional error handling."""
    if cwd is None:
        cwd = PROJECT_ROOT
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=check,
            capture_output=capture,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture:
            fail(f"Command failed: {cmd}")
            if e.stdout:
                print(f"    stdout: {e.stdout.strip()}")
            if e.stderr:
                print(f"    stderr: {e.stderr.strip()}")
        return e


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Install Python dependencies
# ──────────────────────────────────────────────────────────────────────────────

def install_python_deps():
    step(1, 6, "Installing Python dependencies from requirements.txt...")

    if not os.path.isfile(REQUIREMENTS_FILE):
        fail(f"requirements.txt not found at {REQUIREMENTS_FILE}")
        return False

    result = run_cmd(
        f'"{sys.executable}" -m pip install -r "{REQUIREMENTS_FILE}" --quiet',
        check=False,
        capture=True,
    )
    if isinstance(result, subprocess.CalledProcessError) or result.returncode != 0:
        fail("Failed to install some Python dependencies.")
        warn("Trying to install packages individually...")
        with open(REQUIREMENTS_FILE, "r") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        failed_pkgs = []
        for pkg in packages:
            r = run_cmd(
                f'"{sys.executable}" -m pip install {pkg} --quiet',
                check=False,
                capture=True,
            )
            if isinstance(r, subprocess.CalledProcessError) or r.returncode != 0:
                failed_pkgs.append(pkg)
                warn(f"  Could not install: {pkg}")
            else:
                success(f"  Installed: {pkg}")
        if failed_pkgs:
            fail(f"Failed packages: {', '.join(failed_pkgs)}")
            warn("The pipeline will continue, but some features may not work.")
        return True
    else:
        success("All Python dependencies installed.")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Install Node.js Solidity parser
# ──────────────────────────────────────────────────────────────────────────────

def install_node_deps():
    step(2, 6, "Installing Node.js Solidity parser (@solidity-parser/parser)...")

    # Check if node_modules already has it
    parser_path = os.path.join(PROJECT_ROOT, "node_modules", "@solidity-parser", "parser")
    if os.path.isdir(parser_path):
        success("@solidity-parser/parser already installed.")
        return True

    # Check if Node.js and npm are available
    node_check = run_cmd("node --version", check=False, capture=True)
    if isinstance(node_check, subprocess.CalledProcessError) or node_check.returncode != 0:
        fail("Node.js is not installed or not in PATH.")
        warn("The Solidity AST parser (util.py) requires Node.js.")
        warn("Please install Node.js locally using these commands:")
        warn("  wget https://nodejs.org/dist/v20.11.1/node-v20.11.1-linux-x64.tar.xz")
        warn("  tar -xf node-v20.11.1-linux-x64.tar.xz")
        warn("  export PATH=\"$PWD/node-v20.11.1-linux-x64/bin:$PATH\"")
        return False

    npm_check = run_cmd("npm --version", check=False, capture=True)
    if isinstance(npm_check, subprocess.CalledProcessError) or npm_check.returncode != 0:
        fail("npm is not available.")
        return False

    result = run_cmd("npm install @solidity-parser/parser", check=False, capture=True)
    if isinstance(result, subprocess.CalledProcessError) or result.returncode != 0:
        fail("Failed to install @solidity-parser/parser.")
        return False

    success("@solidity-parser/parser installed successfully.")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Download NLTK data
# ──────────────────────────────────────────────────────────────────────────────

def download_nltk_data():
    step(3, 6, "Downloading NLTK data (punkt, wordnet, tagger)...")

    try:
        import nltk

        nltk_packages = [
            "punkt",
            "punkt_tab",
            "wordnet",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "omw-1.4",
        ]
        for pkg in nltk_packages:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                warn(f"  Could not download NLTK package: {pkg}")

        success("NLTK data downloaded.")
        return True

    except ImportError:
        fail("nltk is not installed. Skipping NLTK data download.")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Download CodeBERT model
# ──────────────────────────────────────────────────────────────────────────────

def download_codebert():
    step(4, 6, "Checking CodeBERT model...")

    # Check if model already exists
    required_files = ["config.json"]
    if os.path.isdir(CODEBERT_DIR):
        existing = os.listdir(CODEBERT_DIR)
        if any(f in existing for f in required_files):
            success(f"CodeBERT model already exists at '{CODEBERT_DIR}'.")
            return True

    print(f"  Downloading CodeBERT from HuggingFace (microsoft/codebert-base)...")
    print(f"  Destination: {CODEBERT_DIR}")
    print(f"  This may take several minutes on first run...\n")

    try:
        from transformers import RobertaTokenizer, RobertaModel

        # Download tokenizer
        print("  Downloading tokenizer...")
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        tokenizer.save_pretrained(CODEBERT_DIR)
        success("Tokenizer downloaded and saved.")

        # Download model weights
        print("  Downloading model weights (~500 MB)...")
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        model.save_pretrained(CODEBERT_DIR)
        success("Model weights downloaded and saved.")

        success(f"CodeBERT model ready at '{CODEBERT_DIR}'.")
        return True

    except ImportError:
        fail("'transformers' package not installed. Cannot download CodeBERT.")
        return False
    except Exception as e:
        fail(f"Error downloading CodeBERT: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Extract vector store from RAR
# ──────────────────────────────────────────────────────────────────────────────

def extract_vector_store():
    step(5, 6, "Checking vector store (code_vector_whitening.pkl)...")

    if os.path.isfile(VECTOR_STORE_PKL):
        size_mb = os.path.getsize(VECTOR_STORE_PKL) / (1024 * 1024)
        success(f"Vector store already extracted ({size_mb:.1f} MB).")
        return True

    if not os.path.isfile(VECTOR_STORE_RAR):
        fail(f"RAR archive not found: {VECTOR_STORE_RAR}")
        warn("You need to obtain the code_vector_whitening.rar or .pkl file.")
        return False

    print(f"  Extracting {VECTOR_STORE_RAR}...")

    # Try method 1: patoolib (uses system archivers like WinRAR)
    try:
        import patoolib
        patoolib.extract_archive(VECTOR_STORE_RAR, outdir=MODEL_DIR)
        if os.path.isfile(VECTOR_STORE_PKL):
            size_mb = os.path.getsize(VECTOR_STORE_PKL) / (1024 * 1024)
            success(f"Vector store extracted successfully ({size_mb:.1f} MB).")
            return True
    except Exception as e:
        warn(f"  patoolib extraction failed: {e}")

    # Try method 2: rarfile with system unrar
    try:
        import rarfile
        rf = rarfile.RarFile(VECTOR_STORE_RAR)
        rf.extractall(MODEL_DIR)
        rf.close()
        if os.path.isfile(VECTOR_STORE_PKL):
            size_mb = os.path.getsize(VECTOR_STORE_PKL) / (1024 * 1024)
            success(f"Vector store extracted successfully ({size_mb:.1f} MB).")
            return True
    except Exception as e:
        warn(f"  rarfile extraction failed: {e}")

    # Try method 3: WinRAR command line directly
    winrar_paths = [
        r"C:\Program Files\WinRAR\rar.exe",
        r"C:\Program Files\WinRAR\unrar.exe",
        r"C:\Program Files (x86)\WinRAR\rar.exe",
    ]
    for rar_exe in winrar_paths:
        if os.path.isfile(rar_exe):
            result = run_cmd(
                f'"{rar_exe}" x -o+ "{VECTOR_STORE_RAR}" "{MODEL_DIR}\\"',
                check=False,
                capture=True,
            )
            if not isinstance(result, subprocess.CalledProcessError) and result.returncode == 0:
                if os.path.isfile(VECTOR_STORE_PKL):
                    size_mb = os.path.getsize(VECTOR_STORE_PKL) / (1024 * 1024)
                    success(f"Vector store extracted successfully ({size_mb:.1f} MB).")
                    return True

    fail("Could not extract RAR file. Please extract it manually:")
    warn(f"  Extract '{VECTOR_STORE_RAR}' to '{MODEL_DIR}'")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Run main program
# ──────────────────────────────────────────────────────────────────────────────

def run_ccgir():
    step(6, 6, "Running CCGIR retrieval pipeline (CCGIR.py)...")

    ccgir_path = os.path.join(PROJECT_ROOT, "CCGIR.py")
    if not os.path.isfile(ccgir_path):
        fail(f"CCGIR.py not found at {ccgir_path}")
        return False

    print(f"  Executing: python CCGIR.py\n")
    result = run_cmd(f'"{sys.executable}" "{ccgir_path}"', check=False)
    if isinstance(result, subprocess.CalledProcessError) or result.returncode != 0:
        fail("CCGIR.py exited with errors.")
        return False

    success("CCGIR pipeline completed successfully.")
    return True


def run_app():
    step(6, 6, "Starting Flask web application (app.py)...")

    app_path = os.path.join(PROJECT_ROOT, "app.py")
    if not os.path.isfile(app_path):
        fail(f"app.py not found at {app_path}")
        return False

    print(f"  Executing: python app.py")
    print(f"  Web UI will be available at http://127.0.0.1:5000/index\n")
    result = run_cmd(f'"{sys.executable}" "{app_path}"', check=False)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_setup():
    """Run all setup steps (1-5)."""
    results = {}

    results["python_deps"] = install_python_deps()
    print()
    results["node_deps"] = install_node_deps()
    print()
    results["nltk_data"] = download_nltk_data()
    print()
    results["codebert"] = download_codebert()
    print()
    results["vector_store"] = extract_vector_store()
    print()

    # Summary
    banner("Setup Summary")
    for name, status in results.items():
        icon = f"{Colors.GREEN}OK{Colors.RESET}" if status else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{icon}] {name}")

    all_ok = all(results.values())
    critical_ok = results.get("codebert", False) and results.get("vector_store", False)

    if all_ok:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}All setup steps completed successfully!{Colors.RESET}")
    elif critical_ok:
        print(f"\n  {Colors.YELLOW}{Colors.BOLD}Setup completed with warnings. Critical components are ready.{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}{Colors.BOLD}Setup incomplete. Some critical components are missing.{Colors.RESET}")

    return all_ok, critical_ok


def main():
    parser = argparse.ArgumentParser(
        description="CCGIR - Complete Setup & Execution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py              # Full pipeline: setup + run CCGIR.py
  python run_pipeline.py --setup-only # Only setup dependencies and models
  python run_pipeline.py --run-only   # Skip setup, directly run CCGIR.py
  python run_pipeline.py --app        # Setup + launch Flask web app
        """,
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only run setup steps (install deps, download models), don't execute CCGIR",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Skip setup, directly run CCGIR.py (assumes everything is already set up)",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Run the Flask web app (app.py) instead of CCGIR.py",
    )

    args = parser.parse_args()

    banner("CCGIR - Setup & Execution Pipeline")
    start_time = time.time()

    os.chdir(PROJECT_ROOT)

    if args.run_only:
        # Skip setup, go straight to execution
        if args.app:
            run_app()
        else:
            run_ccgir()
    elif args.setup_only:
        # Only setup
        run_setup()
    else:
        # Full pipeline: setup + run
        all_ok, critical_ok = run_setup()

        if critical_ok:
            print()
            if args.app:
                run_app()
            else:
                run_ccgir()
        else:
            fail("Cannot run CCGIR: critical setup steps failed.")
            fail("Fix the issues above and re-run the pipeline.")
            sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{Colors.CYAN}Pipeline finished in {elapsed:.1f} seconds.{Colors.RESET}\n")


if __name__ == "__main__":
    main()
