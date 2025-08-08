import pandas as pd
from numpy import nan
import numpy as np

og_data = pd.read_excel("data\\shdf_property_summaries_with_llm.xlsx")
new_llm_data = pd.read_excel("data\\shdf_property_summaries_with_llm_2.xlsx")
#rasheed_data = pd.read_excel("data\\Evaluation_Framework DG Cities New.xlsx", sheet_name="MeasuresCompletion")
llm_anaysis_column_names = [
    "damp_before_severity",
    "damp_before_frequency",
    "damp_after_severity",
    "damp_after_frequency",
    "windows_doors_before_severity",
    "windows_doors_before_frequency",
    "windows_doors_after_severity",
    "windows_doors_after_frequency",
    "leaks_before_severity",
    "leaks_before_frequency",
    "leaks_after_severity",
    "leaks_after_frequency",
    "structural_before_severity",
    "structural_before_frequency",
    "structural_after_severity",
    "structural_after_frequency",
    "summary",
    "uprn"
]

takeaway_cols = ["damp_mould_takeaway",	"windows_doors_takeaway","leaks_takeaway","structural_takeaway", "uprn"]

final_df = pd.merge(og_data, new_llm_data[takeaway_cols], on="uprn", how="left")
#priorities = pd.read_excel("data\\shdf_priorities.xlsx")

#final_df = pd.merge(priorities, og_data[column_names], on="uprn", how="left")
# pre_sap_scores = og_data["Revised SAP Score "].copy()
# pre_sap_scores.loc[pre_sap_scores.isna()] = og_data.loc[pre_sap_scores.isna(), "Confirmed pre-works SAP score"]

# og_data["sap_difference"] = og_data["Post-works SAP score"]

# og_data.loc[og_data["sap_difference"].notnull(), "sap_difference"] = (
#     og_data.loc[og_data["sap_difference"].notnull(), "sap_difference"] - pre_sap_scores[og_data["sap_difference"].notnull()]
# )

# Calculate 'Cost per SAP score improvement'
og_data['Cost per SAP score improvement per measure'] = (
    og_data['Average Cost per measure'] / og_data['sap_difference']
).replace([np.inf, -np.inf], nan).replace(0, nan)

og_data['Cost per SAP score improvement'] = (
    og_data['Total Order Value'] / og_data['sap_difference']
).replace([np.inf, -np.inf], nan).replace(0, nan)


completion_columns = [
    'EWI  completion (date)',
    'CWI completion (date)',
    'Loft insulation completion (date)',
    'Windows Installation completion (date)',
    'Doors Installation completion (date)',
    'Ventilation (dMEV) completion (date)'
]

abbreviations = {
    'EWI  completion (date)': 'EWI',
    'CWI completion (date)': 'CWI',
    'Loft insulation completion (date)': 'LI',
    'Windows Installation completion (date)': 'Windows',
    'Doors Installation completion (date)': 'Doors',
    'Ventilation (dMEV) completion (date)': 'MEV'
}

# 3. Iterate through the columns and create new boolean columns
# for col in completion_columns:
#     new_col_name = col.replace(' completion (date)', ' completed').replace(' ', '_').lower()
#     og_data[new_col_name] = og_data[col].notna()

# 4. Create a text string column with all works done (abbreviated)
# def get_works_done(row):
#     done = [abbr for col, abbr in abbreviations.items() if pd.notna(row[col])]
#     return ', '.join(done) if done else ''

# og_data['works_done'] = og_data.apply(get_works_done, axis=1)

# og_data["Total Time From First Contact"] = (
#     pd.to_datetime(og_data["All works signed off & handover document provided to client (date)"], errors='coerce') -
#     pd.to_datetime(og_data["First letter sent to resident (date)"], errors='coerce')
# ).dt.days

# og_data["First Contact to Assessment Time"] = (
#     pd.to_datetime(og_data["Retrofit assessment booked (date)"], errors='coerce') -
#     pd.to_datetime(og_data["First letter sent to resident (date)"], errors='coerce')
# ).dt.days

# og_data["Assessment to Design Time"] = (
#     pd.to_datetime(og_data["Design Sign off (date) "], errors='coerce') -
#     pd.to_datetime(og_data["Retrofit assessment booked (date)"], errors='coerce')
# ).dt.days

# og_data["Assessment to Works Start Time"] = (
#     pd.to_datetime(og_data["Property open (date)"], errors='coerce') -
#     pd.to_datetime(og_data["Retrofit assessment booked (date)"], errors='coerce')
# ).dt.days

# og_data["Design to Works Start Time"] = (
#     pd.to_datetime(og_data["Property open (date)"], errors='coerce') -
#     pd.to_datetime(og_data["Design Sign off (date) "], errors='coerce')
# ).dt.days

# og_data["Works Start to Completion Time"] = (
#     pd.to_datetime(og_data["All works signed off & handover document provided to client (date)"], errors='coerce') -
#     pd.to_datetime(og_data["Property open (date)"], errors='coerce')
# ).dt.days   

# og_data["Overrun Time"] = (
#     pd.to_datetime(og_data["All works signed off & handover document provided to client (date)"], errors='coerce') -
#     pd.to_datetime(og_data["Estimated completion of works (date)"], errors='coerce')
# ).dt.days

# time_columns = [
#     "Total Time From First Contact",
#     "First Contact to Assessment Time",
#     "Assessment to Design Time",
#     "Assessment to Works Start Time",
#     "Design to Works Start Time",
#     "Works Start to Completion Time",
#     "Overrun Time"
# ]

# for col in time_columns:
#     og_data[col] = og_data[col].where(og_data[col].notna(), nan)

# work_columns = [
#     'ewi_completed',
#     'cwi_completed',
#     'loft_insulation_completed',
#     'windows_installation_completed',
#     'doors_installation_completed',
#     'ventilation_(dmev)_completed'
# ]

# # Ensure all work columns exist in the DataFrame and are of boolean type.
# # This loop will convert 'True'/'False' strings to actual booleans if needed.
# for col in work_columns:
#     if col in og_data.columns:
#         # This handles cases where the column might be object type (e.g., strings 'True'/'False')
#         if og_data[col].dtype == 'object':
#             og_data[col] = og_data[col].str.lower().eq('true')
#         # Ensure the column is treated as a numeric type for summing (True=1, False=0)
#         og_data[col] = og_data[col].astype(int)
#     else:
#         print(f"Warning: Column '{col}' not found in the DataFrame.")


# # 2. Add the 'Number of works completed' column.
# # This sums the values (True=1, False=0) across the specified columns for each row.
# og_data['Number of works completed'] = og_data[work_columns].sum(axis=1)


# # 3. Add the 'Average Cost per measure' column.
# # We divide 'Total Order Value' by 'Number of works completed'.
# # To prevent a DivideByZeroError, we replace any resulting 'inf' or 'NaN' values with 0.
# # This happens when 'Number of works completed' is 0.
# og_data['Average Cost per measure'] = (og_data['Total Order Value'] / og_data['Number of works completed']).replace([np.inf, -np.inf], 0).fillna(0)


#final_df = og_data.copy()
final_df.to_excel("data\\shdf_property_summaries_with_llm.xlsx", index=False)