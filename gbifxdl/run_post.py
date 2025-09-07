# Example of use of POST from the GBIF Occurrence API.
# See https://techdocs.gbif.org/en/openapi/v1/occurrence for more details.
# You need to create a account on GBIF to use this example.
# To use this example, add a file named .env next to this python file.
# In .env add the following:
#   GBIF_PWD=your_gbif_password
# Replace `your_gbif_password` with your GBIF password.

import argparse
from gbifxdl import post
from dotenv import dotenv_values
from os.path import join, dirname, realpath

def addcwd(path):
    """Add current Python file workdir to path."""
    return join(dirname(realpath(__file__)), path)

def run(payload_path, download_key_path, pwd=None):
    # Load password either from arg or from .env
    if pwd is None:
        env_path = addcwd(".env")
        env = dotenv_values(env_path)
        if "GBIF_PWD" not in env:
            raise ValueError("GBIF_PWD not found in .env and no password provided via CLI.")
        pwd = env["GBIF_PWD"]

    # Send a post request to GBIF
    download_key = post(payload_path, pwd=pwd, wait=False)

    # Save the download key to a file
    with open(download_key_path, "w") as file:
        file.write(download_key)

    print(f"Download key saved to {download_key_path}: {download_key}")

def cli():
    parser = argparse.ArgumentParser(
        description="Submit a GBIF occurrence download request."
    )
    parser.add_argument(
        "--payload",
        "-p",
        help="Path to the payload JSON file",
    )
    parser.add_argument(
        "--keyfile",
        "-k",
        default="download_key.txt",
        help="Path to save the download key (default: download_key.txt)",
    )
    parser.add_argument(
        "--password",
        "-w",
        help="GBIF password (if not provided, will be read from .env)",
    )

    args = parser.parse_args()
    run(args.payload, args.output, pwd=args.password)

if __name__ == "__main__":
    cli()

