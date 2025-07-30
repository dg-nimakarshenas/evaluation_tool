import pandas as pd
import io
import os

# --- Configuration ---
# Define the input file path.
input_filename = "data\\sharing_cities_property_summaries.xlsx"
# Define the output directory.
output_dir = "data"
# Define the output Excel file name
output_excel_filename = os.path.join(output_dir, "sharing_cities_property_summaries.xlsx")

# Define the completion dates for specific assets.
SHARING_CITIES_COMPLETION_DATES = {
    "ERNEST DENCE ESTATE": pd.to_datetime("2022-05-01"),
    "FLAMSTEAD ESTATE": pd.to_datetime("2022-11-01")
}

# Define the overall analysis period.
ANALYSIS_START_DATE = pd.to_datetime("2020-01-01")
ANALYSIS_END_DATE = pd.to_datetime("2025-01-01")

# --- Data Loading ---
# Create the output directory if it doesn't exist.
os.makedirs(output_dir, exist_ok=True)

# Load the dataset from the Excel file.
try:
    df = pd.read_excel(input_filename)
except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    # Create a dummy dataframe to prevent further errors.
    df = pd.DataFrame({
        'asset': [], 'summary': [],
        'damp_before_severity': [], 'damp_after_severity': [], 'damp_before_frequency': [], 'damp_after_frequency': [],
        'windows_doors_before_severity': [], 'windows_doors_after_severity': [], 'windows_doors_before_frequency': [], 'windows_doors_after_frequency': [],
        'leaks_before_severity': [], 'leaks_after_severity': [], 'leaks_before_frequency': [], 'leaks_after_frequency': [],
        'structural_before_severity': [], 'structural_after_severity': [], 'structural_before_frequency': [], 'structural_after_frequency': []
    })

# --- Frequency Normalization ---
print("--- Adjusting Frequencies Based on Completion Dates ---")
repair_types = ['damp', 'windows_doors', 'leaks', 'structural']

# Create a copy of the original DataFrame to modify for frequency normalization
df_normalized = df.copy()

for estate_name, completion_date in SHARING_CITIES_COMPLETION_DATES.items():
    print(f"Processing asset: {estate_name}")
    
    # Calculate the duration of the 'before' and 'after' periods in years.
    before_duration_days = (completion_date - ANALYSIS_START_DATE).days
    after_duration_days = (ANALYSIS_END_DATE - completion_date).days
    
    # Convert days to years for normalization.
    before_years = before_duration_days / 365.25
    after_years = after_duration_days / 365.25

    if before_years <= 0 or after_years <= 0:
        print(f"  -> Warning: Invalid time period for {estate_name}. Skipping frequency adjustment.")
        continue
        
    print(f"  -> Normalizing frequencies over {before_years:.2f} years (before) and {after_years:.2f} years (after).")

    # Create a mask to select rows corresponding to the current asset.
    asset_mask = df_normalized['estate'] == estate_name

    # Normalize the frequency columns for each repair type for the current asset.
    for repair in repair_types:
        before_freq_col = f'{repair}_before_frequency'
        after_freq_col = f'{repair}_after_frequency'

        if before_freq_col in df_normalized.columns and after_freq_col in df_normalized.columns:
            # Ensure columns are numeric before division
            df_normalized[before_freq_col] = pd.to_numeric(df_normalized[before_freq_col], errors='coerce')
            df_normalized[after_freq_col] = pd.to_numeric(df_normalized[after_freq_col], errors='coerce')

            # Apply normalization to the specific asset's rows
            df_normalized.loc[asset_mask, before_freq_col] = df_normalized.loc[asset_mask, before_freq_col] / before_years
            df_normalized.loc[asset_mask, after_freq_col] = df_normalized.loc[asset_mask, after_freq_col] / after_years

print("--- Frequency Adjustment Complete ---\n")


# --- Data Processing and Analysis ---
if 'asset' in df_normalized.columns and not df_normalized['asset'].empty:
    assets = df_normalized['asset'].unique()
else:
    assets = []
    print("Warning: 'asset' column not found or is empty. No analysis will be performed.")

for repair in repair_types:
    # Updated filename to reflect content
    output_filename = os.path.join(output_dir, f'{repair}_worsened_summaries.txt')
    any_summaries_written = False

    with open(output_filename, 'w') as f:
        print(f"--- Processing Repair Type: {repair.upper()} ---")
        
        for asset in assets:
            asset_df = df_normalized[df_normalized['asset'] == asset].copy()
            total_properties_in_asset = len(asset_df)

            if total_properties_in_asset == 0:
                continue

            before_sev_col = f'{repair}_before_severity'
            after_sev_col = f'{repair}_after_severity'
            before_freq_col = f'{repair}_before_frequency'
            after_freq_col = f'{repair}_after_frequency'

            required_cols = [before_sev_col, after_sev_col, before_freq_col, after_freq_col]
            if not all(col in asset_df.columns for col in required_cols):
                print(f"Skipping '{asset}' for '{repair}' due to missing columns.")
                continue

            # Convert severity columns to numeric (frequency is already numeric)
            asset_df[before_sev_col] = pd.to_numeric(asset_df[before_sev_col], errors='coerce')
            asset_df[after_sev_col] = pd.to_numeric(asset_df[after_sev_col], errors='coerce')

            relevant_repairs_df = asset_df[asset_df[before_sev_col].notna()].copy()
            
            # Identify worsened repairs to include them
            worsened_repairs_df = relevant_repairs_df[
                (
                    ((relevant_repairs_df[after_sev_col] > relevant_repairs_df[before_sev_col]) |
                     (relevant_repairs_df[after_freq_col] > relevant_repairs_df[before_freq_col])) &
                    ((relevant_repairs_df[after_sev_col] > 1) & (relevant_repairs_df[after_freq_col] > 1))
                )
            ]
            
            worsened_summaries = worsened_repairs_df['summary']
            
            # --- Writing to Text File ---
            f.write(f"============================================================\n")
            f.write(f"ASSET: {asset}\n")
            f.write(f"REPAIR TYPE: {repair}\n")
            f.write(f"------------------------------------------------------------\n")
            f.write(f"STATISTICS:\n")
            f.write(f"  - Total Properties in Asset: {total_properties_in_asset}\n")
            f.write(f"  - Properties with Worsened Conditions: {len(worsened_summaries)}\n")
            f.write(f"============================================================\n\n")

            # Write the summaries for properties that worsened.
            if not worsened_summaries.empty:
                any_summaries_written = True
                f.write("SUMMARIES OF WORSENED PROPERTIES:\n\n")
                for summary in worsened_summaries:
                    f.write(f"- {str(summary)}\n")
                f.write("\n\n") # Add extra space before the next asset block.
                print(f"  -> Found {len(worsened_summaries)} worsened properties for '{asset}'. Summaries written.")
            else:
                 print(f"  -> No worsened repairs found for '{asset}'. Only statistics were written.")

    if any_summaries_written:
        print(f"\nExported statistics and summaries for '{repair}' to {output_filename}\n")
    else:
        print(f"\nNo worsened repairs found for any asset for repair type '{repair}'.")
        print(f"File '{output_filename}' contains statistics only.\n")

# --- Final Excel Export ---
try:
    print(f"--- Exporting DataFrame with updated frequencies to Excel ---")
    # Export the DataFrame with the normalized frequencies
    df_normalized.to_excel(output_excel_filename, index=False)
    print(f"Successfully saved updated data to '{output_excel_filename}'")
except Exception as e:
    print(f"Error saving Excel file: {e}")

print("\n--- Analysis Complete ---")
