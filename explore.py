import pandas as pd
CSV_PATH = "tables/HECKTOR_2026_training_data.csv"
BOTTLENECK_CSV_PATH = "tables/bottleneck.csv"

patients = pd.read_csv(CSV_PATH)
patients["PatientID"] = patients["PatientID"].astype(str)
df = pd.read_csv(BOTTLENECK_CSV_PATH)
df["PatientID"] = df["PatientID"].astype(str)
df = df.merge(patients[["PatientID", "split", "T-stage", "N-stage"]],
on="PatientID", how="inner")

for field in ("T-stage", "N-stage"):
    print(f"\n=== {field} ===")
    print(df.groupby("split")[field].value_counts().unstack(fill_value=0))