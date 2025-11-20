import argparse
import os
import subprocess
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Clone GitHub repos listed in a CSV file (column: 'name')."
    )
    parser.add_argument(
        "--csv",
        default="repos.csv",
        help="CSV file with a 'name' column (defauclealt: repos.csv)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=500,
        help="Number of rows to use from the CSV (default: 500)",
    )
    parser.add_argument(
        "--dest",
        default="repos",
        help="Destination folder to clone repos into (default: ./repos)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df_sample = df.head(args.rows)

    repo_list = [
        f"https://github.com/{full_name}.git"
        for full_name in df_sample["name"]
    ]

    os.makedirs(args.dest, exist_ok=True)

    for i, repo_url in enumerate(repo_list, start=1):
        repo_name = repo_url.split('/')[-1].rstrip('.git')
        dest_path = os.path.join(args.dest, repo_name)
        if os.path.exists(dest_path):
            continue

        print(f"[{i}/{len(repo_list)}] Cloning {repo_name}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_path])

if __name__ == "__main__":
    main()
