# safetensor_converter_universal.py
import torch
from safetensors.torch import save_file
import sys
import os

def convert_to_safetensors(filepath, unsafe_load=False):
    """
    Loads any .ckpt, .pth, or .pt file and saves it in .safetensors format.
    - Handles safe and unsafe loading.
    - Handles nested 'state_dict' or 'model' keys.
    - Skips non-tensor data.
    """
    try:
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return

        print(f"Loading model from: {filepath}")

        # --- Unsafe Loading Logic ---
        if unsafe_load:
            print("\n!!! WARNING: Running in UNSAFE mode. !!!")
            weights = torch.load(filepath, map_location="cpu", weights_only=False)
        else:
            print("\nRunning in SAFE mode (default).")
            weights = torch.load(filepath, map_location="cpu")

        # --- UNIVERSAL WEIGHTS EXTRACTION ---
        # This is our new, smart logic.
        if 'model' in weights and hasattr(weights['model'], 'state_dict'):
            print("Found a 'model' object. Extracting its state_dict.")
            weights = weights['model'].state_dict()
        elif 'state_dict' in weights:
            print("Found a 'state_dict' key. Using its contents.")
            weights = weights['state_dict']
        elif 'params_ema' in weights:
            print("Found 'params_ema'. Using its contents.")
            weights = weights['params_ema']
        # ------------------------------------

        final_weights = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                final_weights[k] = v

        if not final_weights:
            print("\nError: No valid tensor weights were found after parsing.")
            return

        directory, filename = os.path.split(filepath)
        base_filename, _ = os.path.splitext(filename)
        output_filepath = os.path.join(directory, f"{base_filename}.safetensors")

        print(f"Saving to: {output_filepath}")
        save_file(final_weights, output_filepath, metadata={'format': 'pt'})

        print("\nConversion successful!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nIf this is a trusted file, you may need to run this script with the --unsafe flag.")

# --- Script Execution ---
if __name__ == "__main__":
    unsafe_flag = '--unsafe' in sys.argv
    args = [arg for arg in sys.argv if arg != '--unsafe']

    if len(args) < 2:
        print("Usage: python safetensor_converter_universal.py [--unsafe] <path_to_model_file>")
    else:
        model_path = args[1]
        convert_to_safetensors(model_path, unsafe_load=unsafe_flag)
