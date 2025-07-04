# safetensor_converter_v4_universal_inspector.py
import torch
import sys
import os

def inspect_model_structure(filepath, unsafe_load=False):
    """
    Loads a model file and prints its top-level structure.
    Includes an 'unsafe' mode to handle models with pickled Python code.
    """
    try:
        print(f"--- Inspecting Model Structure ---")
        print(f"File: {filepath}\n")

        if not os.path.exists(filepath):
            print("Error: File not found.")
            return

        # --- NEW Unsafe Loading Logic for Inspector ---
        if unsafe_load:
            print("!!! WARNING: Running inspection in UNSAFE mode. !!!")
            weights = torch.load(filepath, map_location="cpu", weights_only=False)
        else:
            print("Running inspection in SAFE mode (default).")
            weights = torch.load(filepath, map_location="cpu")
        # ----------------------------------------------

        if 'state_dict' in weights:
            print("Found a 'state_dict' key. Inspecting contents of 'state_dict'.")
            weights = weights['state_dict']
        else:
            print("No 'state_dict' key found. Inspecting top-level keys.")

        print("\n--- Top-Level Keys and Types ---")
        for k, v in weights.items():
            print(f"Key: '{k}',  Type: {type(v)}")
        print("--------------------------------\n")
        print("Inspection complete.")

    except Exception as e:
        print(f"An error occurred during inspection: {e}")
        print("\nIf you trust this file, you may need to run this script with the --unsafe flag.")
        print("Example: python your_script_name.py --inspect --unsafe \"path\\to\\file.pt\"")

# --- Script Execution ---
if __name__ == "__main__":
    inspect_flag = '--inspect' in sys.argv
    unsafe_flag = '--unsafe' in sys.argv

    args = [arg for arg in sys.argv if arg not in ['--inspect', '--unsafe']]

    if inspect_flag:
        if len(args) > 1:
            model_path = args[1]
            inspect_model_structure(model_path, unsafe_load=unsafe_flag)
        else:
            print("Usage for inspection: python your_script_name.py --inspect [--unsafe] <path_to_model>")
    else:
        print("This is an inspector script. Please run with the --inspect flag.")
