"""
download_codebert.py
Downloads the microsoft/codebert-base model from HuggingFace
and saves it locally to model/codebert/ if not already present.
"""

import os
import sys


def download_codebert(model_dir="model/codebert"):
    """Download CodeBERT model from HuggingFace if not already present."""

    # Check if model already exists by looking for key files
    required_files = ["config.json", "tokenizer.json"]
    if os.path.isdir(model_dir):
        existing = os.listdir(model_dir)
        if all(f in existing for f in required_files):
            print(f"[OK] CodeBERT model already exists at '{model_dir}'. Skipping download.")
            return True

    print(f"[>>] Downloading CodeBERT model to '{model_dir}'...")
    print("    Source: https://huggingface.co/microsoft/codebert-base")
    print("    This may take a few minutes depending on your internet speed.\n")

    try:
        from transformers import AutoTokenizer, RobertaModel

        # Download and save tokenizer
        print("    Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tokenizer.save_pretrained(model_dir)
        print("    [OK] Tokenizer saved.")

        # Download and save model
        print("    Downloading model weights...")
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        model.save_pretrained(model_dir)
        print("    [OK] Model weights saved.")

        print(f"\n[OK] CodeBERT model successfully downloaded to '{model_dir}'.")
        return True

    except ImportError:
        print("[FAIL] Error: 'transformers' package not installed.")
        print("    Run: pip install transformers torch")
        return False
    except Exception as e:
        print(f"[FAIL] Error downloading CodeBERT: {e}")
        return False


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/codebert"
    success = download_codebert(model_dir)
    sys.exit(0 if success else 1)
