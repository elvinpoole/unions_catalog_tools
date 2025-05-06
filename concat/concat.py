import argparse
import sys
import yaml
import cattools

required_config_var = ["base_path", "cat_file", "concat_output_label"]

def main(bands, maxn, config):
    c = cattools.ConCat(bands, maxn, config, verbose=True)
    c.run()

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