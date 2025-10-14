import yaml
from datetime import datetime, timedelta
import os

# Parameters
start_date = datetime.strptime("2020-08-08", "%Y-%m-%d")  # earliest reference start
end_date = datetime.strptime("2024-03-31", "%Y-%m-%d")    # exclusive
reference_days = 30
test_days = 7
output_dir = "configs"

# Step 1: Generate all sliding windows
date_ranges = []
current_test_start = start_date + timedelta(days=reference_days)

while True:
    reference_start = current_test_start - timedelta(days=reference_days + test_days - 1)
    reference_end = current_test_start - timedelta(days=1)
    test_end = current_test_start + timedelta(days=test_days - 1)

    if test_end >= end_date:
        break

    date_ranges.append({
        "reference": [reference_start.strftime("%Y-%m-%d"), reference_end.strftime("%Y-%m-%d")],
        "test": [current_test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")]
    })

    current_test_start = test_end + timedelta(days=1)

# Step 2: Manually split (first 9 have 7, rest have 4)
os.makedirs(output_dir, exist_ok=True)

chunks = []
first_nine_size = 7
rest_size = 4

# First 9 chunks of size 7
index = 0
for _ in range(9):
    chunk = date_ranges[index:index + first_nine_size]
    if chunk:
        chunks.append(chunk)
    index += first_nine_size

# Remaining chunks of size 4
while index < len(date_ranges):
    chunk = date_ranges[index:index + rest_size]
    if chunk:
        chunks.append(chunk)
    index += rest_size

# Step 3: Write to files
for i, chunk in enumerate(chunks):
    config_data = {"date_ranges": chunk}
    filename = f"{output_dir}/config_part_{i+1:02}.yaml"
    with open(filename, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)
    print(f"Wrote {len(chunk)} windows to {filename}")

print(f"\nTotal: {len(date_ranges)} windows split into {len(chunks)} config files.")
