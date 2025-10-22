#!/usr/bin/env python3
"""
Make a DataFrame with *all* designated Pango lineages
in their usual (aliased) form and in fully unaliased form.

Output →  pango_alias_mapping.csv  (columns: lineage_alias, lineage_unaliased)
"""

import pandas as pd
import requests
from pango_aliasor.aliasor import Aliasor

# 1.  Get the current list of designated lineages ----------------------------
NOTES_URL = (
    "https://raw.githubusercontent.com/"
    "cov-lineages/pango-designation/master/lineage_notes.txt"
)
notes_txt = requests.get(NOTES_URL, timeout=30).text
designations = [line.split(maxsplit=1)[0] for line in notes_txt.splitlines() if line]

# 2.  Expand each lineage with pango-aliasor ---------------------------------
aliasor = Aliasor()                          # pulls latest alias_key.json
df = pd.DataFrame(
    {
        "lineage_alias": designations,
        "lineage_unaliased": [aliasor.uncompress(lin) for lin in designations],
    }
).sort_values("lineage_alias").reset_index(drop=True)

# 3.  Save or inspect ---------------------------------------------------------
df.to_csv("summary_files/pango_alias_mapping.csv", index=False)
print(df.head(10))
