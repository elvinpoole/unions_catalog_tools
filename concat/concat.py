import argparse
import sys
import yaml
import cattools
from dask.distributed import Client, LocalCluster

required_config_var = []

def main(bands, maxn, config):
    if config.get('parallel'): 
        #if running in parallel, start a dask client
        n_workers = config.get("n_workers", 1)
        print(f"Starting Dask with {n_workers} workers...")
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}")
    else:
        client = None

    try:
        c = cattools.ConCat(bands, maxn, config, verbose=True)
        c.run()
    finally:
        if client is not None:
            client.close()

def parse_maxn(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("maxn must be an integer or 'auto'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate file into 1 hdf5 files. Select only objects that have a magnitude in all bands specified")
    parser.add_argument(
        "--bands",
        type=str,
        choices=["ugri", "ugriz"],
        required=True,
        help="Band combination to use: 'ugri' or 'ugriz'"
    )
    parser.add_argument(
        "--maxn",
        type=parse_maxn,
        default="auto",
        help="Maximum number of items to process (integer or 'auto', if auto will pull from the file headers)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file containing file paths"
    )
    
    args = parser.parse_args()

    # Load config file
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            for req_var_name in required_config_var:
                var = config.get(req_var_name)
                if var is None:
                    print(f"Error: '{req_var_name}' not found in config file.", file=sys.stderr)
                    sys.exit(1)
    except Exception as e:
        print(f"Error reading config file: {e}", file=sys.stderr)
        sys.exit(1)

    main(args.bands, args.maxn, config)