"""Download a UTC date range of GH Archive hourly files.

Example:
    python scripts/download_gharchive.py --start 2026-10-12 --end 2026-11-15 --out raw_data
"""

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path


def day_hours(start, end, end_hour=23):
    day = start
    while day <= end:
        for hour in range(end_hour + 1 if day == end else 24):
            yield f"{day.isoformat()}-{hour}.json.gz"
        day += timedelta(days=1)


def download(name, out_dir):
    path = out_dir / name
    if path.exists() and path.stat().st_size > 0:
        return "existing", name

    part = out_dir / f"{name}.part"
    url = f"https://data.gharchive.org/{name}"
    result = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error",
         "--retry", "3", "--connect-timeout", "15",
         "--max-time", "120", "--output", str(part), url],
        capture_output=True, text=True,
    )
    if result.returncode or not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return "failed", f"{name}: {result.stderr.strip() or 'empty download'}"
    part.replace(path)
    return "downloaded", name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--end-hour", type=int, default=23,
                        help="last UTC hour on the end date, inclusive (0-23)")
    parser.add_argument("--out", type=Path, default=Path("raw_data"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.end < args.start or args.workers < 1 or not 0 <= args.end_hour <= 23:
        parser.error("end must be on or after start; workers positive; end-hour 0-23")
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(day_hours(args.start, args.end, args.end_hour))
    counts = {"existing": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, name, args.out) for name in names]
        for index, future in enumerate(as_completed(futures), 1):
            status, detail = future.result()
            counts[status] += 1
            if status == "failed":
                print(f"FAILED {detail}", flush=True)
            if index % 100 == 0 or index == len(names):
                print(f"{index}/{len(names)} files: {counts}", flush=True)
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
