import torch
from safetensors.torch import load_file
import argparse
import os

def convert_safetensors_to_pth(input_path, output_path):
    """
    Loads a model from a .safetensors file and saves it as a .pth file.
    """
    try:
        print(f"Loading safetensors model from: {input_path}")
        # Load the state dictionary from the safetensors file
        model_weights = load_file(input_path)

        print(f"Saving as .pth to: {output_path}")
        # Save the state dictionary using torch.save(), which uses pickle
        torch.save(model_weights, output_path)

        print(f"Successfully converted '{input_path}' to '{output_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a .safetensors model file to a .pth model file."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="The full path to the input .safetensors file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="(Optional) The full path for the output .pth file. "
             "If not provided, it will save in the same directory with a .pth extension."
    )

    args = parser.parse_args()

    # Determine the output path if not provided
    if not args.output_path:
        # Replaces the file extension with .pth
        base_name = os.path.splitext(args.input_path)[0]
        args.output_path = base_name + ".pth"

    convert_safetensors_to_pth(args.input_path, args.output_path)
