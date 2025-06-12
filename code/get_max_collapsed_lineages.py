import glob
import os

def get_max_collapsed_lineages(summary_dir, pattern="*.collapsed_lineage_count.txt"):
    summary_files = glob.glob(os.path.join(summary_dir, pattern))
    
    if not summary_files:
        print(f"No summary files found in {summary_dir}")
        return None

    max_val = 0
    max_file = None

    for file in summary_files:
        try:
            with open(file, 'r') as f:
                val = int(f.readline().strip())
                if val > max_val:
                    max_val = val
                    max_file = file
        except Exception as e:
            print(f"Skipping {file}: {e}")

    print(f"Max collapsed lineages: {max_val} (from {max_file})")
    return max_val

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find max number of unique collapsed lineages.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .collapsed_lineage_count.txt files")
    parser.add_argument("--pattern", type=str, default="*.collapsed_lineage_count.txt", help="Glob pattern for summary files")

    args = parser.parse_args()
    get_max_collapsed_lineages(args.dir, args.pattern)
