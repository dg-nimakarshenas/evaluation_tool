import pandas as pd

def convert_scientific_to_full_string(value):
    try:
        # Try to convert to float, then format back to a full string
        # This handles cases like "6.5702E+14"
        # It also handles regular numbers like "123"
        # And leaves non-numeric strings like "ABC12345" untouched
        float_val = float(value)
        # Use f-string formatting to avoid scientific notation.
        # The 'f' specifier with no precision gives full float representation.
        # You might need to specify precision like ':.0f' if you know it's always an integer
        # Or ':.15f' for high precision.
        return f'{float_val:f}'.rstrip('0').rstrip('.') if '.' in f'{float_val:f}' else f'{float_val:f}'
    except ValueError:
        # If it's not a valid number (e.g., "ABC12345"), return as is
        return value

def load_priorities(shdf_list_path: tuple[str, str], priorities_path: str, shdf_uprn_column_name:str = "uprn", priortities_uprn_column_name: str = "uprn_") -> pd.DataFrame:
    """
    Loads a subset of property data and filters it based on UPRN values
    found in a specified column within the same DataFrame.

    This function reads two CSV files: one containing a list of properties allocated
    to a specific SHDF (Social Housing Development Fund) and another containing
    priority data for properties. It filters the priority data to include only those
    properties whose UPRN values match those in the SHDF list.


    Args:
        shdf_list_path (str): The path to the CSV file containing the list of properties    
                                    allocated to the SHDF. This file should have a column
                                    named 'uprn' or a specified column name.
        priorities_path (str): The path to the CSV file containing priority data for properties.
        shdf_uprn_column_name (str): The name of the column in the SHDF list that contains the UPRN values.
        priortities_uprn_column_name (str): The name of the column in the priorities data that contains the UPRN values.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered priority data, where the 'uprn' values
                      match those found in the SHDF list. If no matching records are found,
                      an empty DataFrame is returned.     

    """
    shdf_data = pd.read_excel(shdf_list_path[0], sheet_name=shdf_list_path[1], dtype={shdf_uprn_column_name: str})
    priorities_data = pd.read_csv(priorities_path, dtype={priortities_uprn_column_name: str})
    if shdf_uprn_column_name not in shdf_data.columns:
        print("Error: 'uprn' column not found in the provided shdf data.")
        return pd.DataFrame()
    if priortities_uprn_column_name not in priorities_data.columns:
        print("Error: 'uprn' column not found in the provided priorities data.")
        return pd.DataFrame()

    # Extract the unique UPRNs from the specified filter column to create the filter set
    uprn_filter_set = shdf_data[shdf_uprn_column_name].apply(lambda x: str(x).lstrip('0')).unique()

    # Filter the dataset based on the 'uprn' column, checking against the extracted UPRNs
    filtered_data = priorities_data[priorities_data[priortities_uprn_column_name].astype(str).isin(uprn_filter_set)].copy()

    print(f"Successfully filtered {len(filtered_data)} properties.")
    return filtered_data

def merge_shdf_and_priorities(shdf_df: pd.DataFrame, priorities_df: pd.DataFrame, shdf_key_column: str = "UPRN", priorities_key_column: str = "uprn_") -> pd.DataFrame:
    """
    Merges SHDF properties with priorities data based on a common key.

    This function takes two DataFrames: one containing SHDF properties and another
    containing priorities data. It merges them on a specified key column, ensuring
    that the resulting DataFrame contains all columns from both DataFrames.

    Args:
        shdf_df (pd.DataFrame): The DataFrame containing SHDF properties.
        priorities_df (pd.DataFrame): The DataFrame containing priorities data.
        shdf_key_column (str): The name of the column in the SHDF DataFrame to merge on.
        priorities_key_column (str): The name of the column in the priorities DataFrame to merge on.

    Returns:
        pd.DataFrame: A merged DataFrame containing all columns from both input DataFrames.
                       If no matching records are found, an empty DataFrame is returned.
    """
    if shdf_key_column not in shdf_df.columns or priorities_key_column not in priorities_df.columns:
        print(f"Error: Key columns '{shdf_key_column}' or '{priorities_key_column}' not found in the provided DataFrames.")
        return pd.DataFrame()

    # Create temporary columns for merging to avoid modifying the original columns
    shdf_df = shdf_df.copy()
    priorities_df = priorities_df.copy()
    
    # Convert both columns to strings and strip leading zeros from SHDF key
    shdf_df['_merge_key'] = shdf_df[shdf_key_column].astype(str).apply(lambda x: x.lstrip('0'))
    priorities_df['_merge_key'] = priorities_df[priorities_key_column].astype(str)
    
    # Perform a left join using the temporary columns
    merged_df = pd.merge(shdf_df, priorities_df, left_on='_merge_key', right_on='_merge_key', how='left')
    
    # Drop the temporary merge columns
    merged_df = merged_df.drop(columns=['_merge_key'])
    
    print(f"Successfully merged {len(merged_df)} records.")
    return merged_df

def process_repairs_data(shdf_properties_df: pd.DataFrame, repairs_dfs: list[pd.DataFrame], shdf_key_column: str, repairs_key_column: str, start_year: int = 21) -> pd.DataFrame:
    """
    Processes repair data DataFrames, filtering and joining them based on a list of shdf properties.

    This function accepts a DataFrame containing properties allocated to a shdf and
    multiple DataFrames containing yearly repair data. It filters the repair data to include
    only those properties present in the shdf properties list and then combines all
    filtered yearly data into a single output DataFrame.

    Args:
        shdf_properties_df (pd.DataFrame): The DataFrame containing the list of
                                            properties allocated to the shdf. This DataFrame
                                            should have the 'shdf_key_column'.
        repairs_dfs (list[pd.DataFrame]): A list of DataFrames, each containing
                                          yearly repair data for the whole portfolio. Each
                                          DataFrame should also have the 'repairs_key_column'.
        shdf_key_column (str): The name of the column that serves as a common key
                                between the shdf properties DataFrame and the repair
                                data DataFrames (e.g., 'PropertyID', 'AssetNumber').
        repairs_key_column (str): The name of the column in the repair data DataFrames
                                  that corresponds to the common key in the shdf.
        start_year (int): The starting year to assign to the first repairs DataFrame.
                          Subsequent DataFrames will have incremented years. Defaults to 21.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and filtered repair data.
                      If no matching records are found, an empty DataFrame is returned.
    """
    print(f"Starting data processing...")

    # 1. Process the shdf properties DataFrame
    if shdf_key_column not in shdf_properties_df.columns:
        raise ValueError(f"'{shdf_key_column}' not found in the shdf properties DataFrame.")

    # Get the unique list of property IDs from the shdf
    shdf_property_ids = set(shdf_properties_df[shdf_key_column].apply(lambda x: str(x).lstrip('0')).unique())
    print(f"Loaded {len(shdf_property_ids)} unique shdf properties.")

    # 2. Initialize a list to store filtered repairs DataFrames
    filtered_repairs_dfs = []

    # 3. Iterate through each repairs DataFrame
    for i, repairs_df in enumerate(repairs_dfs):
        print(f"Processing repair DataFrame {i+1}/{len(repairs_dfs)}...")

        # Ensure column names are lowercase for consistent access
        repairs_df.columns = repairs_df.columns.str.lower()

        if repairs_key_column not in repairs_df.columns:
            print(f"Warning: '{repairs_key_column}' not found in repair DataFrame {i+1}. Skipping this DataFrame.")
            continue

        # Filter the repairs data to include only properties in the shdf
        filtered_df = repairs_df[repairs_df[repairs_key_column].apply(lambda x: str(x).lstrip('0')).isin(shdf_property_ids)].copy()
        filtered_df["RepairYear"] = start_year

        if not filtered_df.empty:
            filtered_repairs_dfs.append(filtered_df)
            print(f"Filtered {len(filtered_df)} records from repair DataFrame {i+1}.")
        else:
            print(f"No matching records found in repair DataFrame {i+1}.")
        
        start_year += 1  # Increment the year for the next DataFrame

    # 4. Concatenate all filtered DataFrames into a single DataFrame
    if filtered_repairs_dfs:
        final_combined_df = pd.concat(filtered_repairs_dfs, ignore_index=True)
        print(f"Successfully combined {len(final_combined_df)} records across all filtered repair DataFrames.")
    else:
        print("No repair data was filtered and combined.")
        return pd.DataFrame() # Return an empty DataFrame if no data is combined

    return final_combined_df

def process_shdf():
    """
    Placeholder function for future SHDF processing logic.
    Currently, it does nothing but can be expanded later.
    """
    print("Processing SHDF data...")
    shdf_properties = pd.read_excel("data\\wates_tracker_shdf.xlsx")
    priorities_df = pd.read_excel("data\\AssetData_Priorities_llm_v2.xlsx")
    repairs_files = ["data\\21_22_Analysis.xlsx", "data\\22_23_Analysis.xlsx", "data\\23_24_Analysis.xlsx", "data\\2024-25_Full.xlsx"]

    # Define key columns for merging
    shdf_key_column = "uprn"
    repairs_key_column = "50_property_ref"
    priorities_key_column = "uprn_"
    priorities_nlpg_uprn_column = "nlpg_uprn_(move_to_end)"

    # Load and filter priorities data
    print("Loading and filtering priorities data...")
    filtered_priorities = merge_shdf_and_priorities(
        shdf_df=shdf_properties, 
        priorities_df=priorities_df, 
        shdf_key_column=shdf_key_column, 
        priorities_key_column=priorities_nlpg_uprn_column
    )

    # Process repairs data
    print("Processing repairs data...")
    # filtered_repairs_data = process_repairs_data(
    #     shdf_properties_df=filtered_priorities, 
    #     repairs_dfs=[pd.read_excel(file) for file in repairs_files], 
    #     shdf_key_column=priorities_key_column, 
    #     repairs_key_column=repairs_key_column
    # )

    print("Saving processed data to Excel files...")
    filtered_priorities.to_excel("data\\shdf_priorities.xlsx", index=False)
    print("Filtered priorities data saved to 'data\\shdf_priorities.xlsx'.")
    #filtered_repairs_data.to_excel("data\\shdf_repairs_data.xlsx", index=False)
    print("Filtered repairs data saved to 'data\\shdf_repairs_data.xlsx'.")

    






if __name__ == "__main__":
    process_shdf()