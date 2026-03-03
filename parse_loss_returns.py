import re
import sys
import glob
import argparse
import os
import csv
from pathlib import Path

def parse_file(filepath):
    """
    Parse a single .txt file and return {loss_rate: [trial_dicts]} as a .csv file.
    Each trial_dict contains optional keys: 'trial', 'return'.
    """
    results = {}
    current_loss = None
    trial_index_map = {}
    last_index = {}
    loss_re = re.compile(r'loss\s*=\s*([\d.]+)', re.IGNORECASE)
    trial_re = re.compile(r'Trial\s+(\d+)\s+completed:\s*Return\s*=\s*([-\d.]+)', re.IGNORECASE)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect new loss rate section
            loss_match = loss_re.search(line)
            if loss_match and 'parameters' in line.lower():
                current_loss = float(loss_match.group(1))
                if current_loss not in results:
                    results[current_loss] = []
                    trial_index_map[current_loss] = {}
                    last_index[current_loss] = None

            if current_loss is None:
                continue

            # Trial return
            m = trial_re.search(line)
            if m:
                tn = int(m.group(1))
                rv = float(m.group(2))
                # Find or create entry for this trial number
                idx_map = trial_index_map[current_loss]
                if tn in idx_map:
                    idx = idx_map[tn]
                    results[current_loss][idx]['return'] = rv
                else:
                    idx = len(results[current_loss])
                    idx_map[tn] = idx
                    results[current_loss].append({'trial': tn, 'return': rv})
                last_index[current_loss] = idx

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse .txt logs and extract sorted returns per loss rate.")
    parser.add_argument('files', nargs='+', help='One or more input .txt files (globs allowed)')
    parser.add_argument('--out', '-o', help='Output csv path. If omitted, a name based on input file(s) is used')
    args = parser.parse_args()

    # Expand globs
    files = []
    for arg in args.files:
        expanded = glob.glob(arg)
        files.extend(expanded if expanded else [arg])

    files = [str(Path(f)) for f in files]
    if not files:
        print('No input files found.')
        sys.exit(1)

    # Merge results across all files
    merged = {}
    for fp in files:
        print(f"Parsing: {fp}")
        file_results = parse_file(fp)
        for loss, returns in file_results.items():
            merged.setdefault(loss, []).extend(returns)

    if not merged:
        print("No trial return data found in the provided files.")
        sys.exit(1)

    # Normalise merged entries to numeric returns 
    for loss, items in merged.items():
        normalised = []
        for it in items:
            if isinstance(it, dict):
                if 'return' in it and it['return'] is not None:
                    try:
                        normalised.append(float(it['return']))
                    except Exception:
                        continue
                elif 'value' in it and it['value'] is not None:
                    try:
                        normalised.append(float(it['value']))
                    except Exception:
                        continue
                else:
                    continue
            else:
                try:
                    normalised.append(float(it))
                except Exception:
                    continue
        merged[loss] = normalised

    # Sort loss rates and sort returns descending within each
    sorted_losses = sorted(merged.keys())
    for loss in sorted_losses:
        merged[loss] = sorted(merged[loss], reverse=False)
        print(f"  loss={loss}: {len(merged[loss])} trials")

    # Build rows with loss rates as columns 
    max_len = max(len(v) for v in merged.values())
    col_names = [f"loss_{loss}" for loss in sorted_losses]
    rows = []
    for i in range(max_len):
        row = []
        for loss in sorted_losses:
            vals = merged[loss]
            row.append('' if i >= len(vals) else vals[i])
        rows.append(row)

    # Determine output filename 
    if args.out:
        out_path = Path(args.out)
    else:
        basenames = [Path(f).stem for f in files]
        if len(basenames) == 1:
            out_name = f"{basenames[0]}_returns.csv"
        else:
            common = os.path.commonprefix(basenames)
            # sanitise common prefix
            common = re.sub(r'[^A-Za-z0-9_-]+$', '', common)
            if common:
                out_name = f"merged_{common}_returns.csv"
            else:
                out_name = "merged_returns_by_loss.csv"
        out_path = Path(out_name)

    # Write csv 
    with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(col_names)
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved {out_path}  ({len(rows)} rows x {len(col_names)} loss rates)")


if __name__ == "__main__":
    main()
