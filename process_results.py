import pandas as pd
import os

results_filepath = "results.csv"

data = pd.read_csv(results_filepath)

# Find the unique age groups in the data
age_groups = data["age_group"].unique()
print("Unique age groups:", age_groups)

genders = data["gender"].unique()
print("Unique genders:", genders)

output_columns = ["name", "tops", "zones", "top_attempts", "zone_attempts"]

for age_group in age_groups:
	for gender in genders:
		subset = data[(data["age_group"] == age_group) & (data["gender"] == gender)]
		if subset.empty:
			print(f"No data for age group {age_group} and gender {gender}.")
			continue

		# Sort by tops and then by zones and then top_attempts and then zone_attempts
		sorted_subset = subset.sort_values(by=["tops", "zones", "top_attempts", "zone_attempts"], ascending=[False, False, True, True])
		print(f"\nScores for age group {age_group} and gender {gender}:")
		print(sorted_subset[output_columns])

		# Make sure the results directory exists
		if not os.path.exists("results"):
			os.makedirs("results")
		filename = f"results/scores_{age_group}_{gender}.csv"
		sorted_subset[output_columns].to_csv(filename, index=False)
