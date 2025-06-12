import yaml
from datetime import datetime, timedelta

# Parameters
start_date = datetime.strptime("2020-08-08", "%Y-%m-%d")  # earliest reference start
end_date = datetime.strptime("2024-03-31", "%Y-%m-%d")    # exclusive
reference_days = 30
test_days = 7

# Initialize
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

# Save to config.yaml
with open("config.yaml", "w") as f:
    yaml.dump({"date_ranges": date_ranges}, f, sort_keys=False)

print(f"Generated {len(date_ranges)} date windows in config.yaml.")
