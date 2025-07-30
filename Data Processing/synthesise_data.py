import pandas as pd
import numpy as np
import random
import json
from openai import OpenAI
import os
from typing import List
from pydantic import BaseModel, Field


# Define the main structure the LLM should return: a list of feedback items.
class FeedbackList(BaseModel):
    feedback_list: List[str] = Field(description="A list of resident feedback entries.")

def synthesise_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesises health data for properties in the DataFrame.
    This function adds two new columns: 'health_status_before' and 'health_status_after',
    simulating the health status of residents before and after some intervention.

    The health status is randomly assigned based on predefined distributions for 'before' and 'after' states.
    Args:
        df (pd.DataFrame): The input DataFrame containing property data.
    Returns:
        pd.DataFrame: The DataFrame with two new columns for health status.
    """
    health_base_distribution = {
        'Very Good Health': 0.473,
        'Good Health': 0.329,
        'Fair Health': 0.136,
        'Bad Health': 0.047,
        'Very Bad Health': 0.015
    }
    health_new_distribution = {
        'Very Good Health': 0.52,
        'Good Health': 0.35,
        'Fair Health': 0.08,
        'Bad Health': 0.04,
        'Very Bad Health': 0.01
    }
    # Create a new column for health status
    df['health_status_before'] = np.random.choice(
        list(health_base_distribution.keys()),
        size=len(df),
        p=list(health_base_distribution.values())
    )
    df['health_status_after'] = np.random.choice(
        list(health_new_distribution.keys()),
        size=len(df),
        p=list(health_new_distribution.values())
    )
    return df


def assign_contractors(df, col1, col2, group_size=30, min_group_size=8):
    """
    Assigns unique contractors based on group sizes and combinations of two columns.

    This function categorizes groups based on their size:
    1.  Groups with size < `min_group_size` are collected together and then
        re-grouped into new contractor assignments of size `group_size`.
    2.  Groups with size >= `min_group_size` are treated as standard. If a
        standard group's size exceeds `group_size`, it is split into
        sub-groups.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.
        group_size (int, optional): The maximum number of rows for each
                                    contractor group. Defaults to 20.
        min_group_size (int, optional): The threshold for a group to be
                                        considered "small". Defaults to 5.

    Returns:
        pd.DataFrame: The DataFrame with an added 'Contractor' column.
    """
    # Ensure the original index is preserved for re-combining later
    df = df.copy()
    df['original_index'] = df.index

    # Calculate the size of each group defined by (col1, col2)
    df['group_size'] = df.groupby([col1, col2])[col1].transform('count')

    # --- Step 1: Isolate and process "standard" groups (size >= min_group_size) ---
    standard_mask = df['group_size'] >= min_group_size
    standard_df = df[standard_mask].copy()
    
    num_standard_contractors = 0
    if not standard_df.empty:
        # Create a 0-based index within each standard group
        group_idx = standard_df.groupby([col1, col2]).cumcount()
        # Create a sub-group ID by splitting large groups
        sub_group_id = group_idx // group_size
        # Create a composite key that uniquely identifies each sub-group
        composite_key = list(zip(standard_df[col1], standard_df[col2], sub_group_id))
        # Factorize to get unique integer codes for each contractor group
        codes, uniques = pd.factorize(composite_key)
        standard_df['Contractor'] = [f"Contractor {code + 1}" for code in codes]
        num_standard_contractors = len(uniques)

    # --- Step 2: Isolate and process "small" groups (size < min_group_size) ---
    small_mask = df['group_size'] < min_group_size
    small_df = df[small_mask].copy()

    if not small_df.empty:
        # Bundle all small groups together and create new sub-groups of `group_size`
        small_sub_group_ids = np.arange(len(small_df)) // group_size
        # Assign contractor numbers, offsetting by the number of standard contractors
        # to ensure all contractor names are unique across the entire DataFrame.
        small_df['Contractor'] = [f"Contractor {num_standard_contractors + sub_id + 1}" for sub_id in small_sub_group_ids]

    # --- Step 3: Combine the results ---
    # Concatenate the processed dataframes back together
    final_df = pd.concat([standard_df, small_df])
    # Sort by the original index to restore the initial order of the DataFrame
    final_df = final_df.sort_values('original_index').drop(columns=['original_index', 'group_size'])
    
    return final_df

def process_measures(df: pd.DataFrame, measures_col: str, contractor_col: str) -> pd.DataFrame:
    """
    Cleans and synthesises missing data in a 'Measures' column of a DataFrame.

    This function performs two main operations:
    1. It replaces common abbreviations for building measures with their full descriptions.
    2. For rows where the measures data is missing, it generates a random sample of
       measures, assuming that properties handled by the same contractor receive
       similar types of work.
    3. After processing, creates a column for each available measure, with 1 if present in 'Measures_Cleaned', else 0.

    Args:
        df (pd.DataFrame): The input DataFrame.
        measures_col (str): The name of the column containing the measures data (e.g., 'Measures').
        contractor_col (str): The name of the column identifying the contractor.

    Returns:
        pd.DataFrame: A new DataFrame with an added 'Measures_Cleaned' column containing
                      the processed and synthesised data, and binary columns for each measure.
    """
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # --- 1. Abbreviation Replacement ---
    abbreviation_map = {
        "CWI": "Cavity Wall Insulation",
        "MEV's": "Mechanical Extract Ventilation",
        "LI": "Loft Insulation",
        "EWI": "External Wall Insulation"
    }

    # Create the new column to store the results
    result_df['Measures_Cleaned'] = result_df[measures_col].astype(str)

    # Replace all occurrences of abbreviations with their full text
    for abbr, full_text in abbreviation_map.items():
        result_df['Measures_Cleaned'] = result_df['Measures_Cleaned'].str.replace(abbr, full_text, regex=False)

    # --- 2. Synthesise Missing Data ---
    available_measures = [
        'Cavity Wall Insulation',
        'Mechanical Extract Ventilation',
        'Loft Insulation',
        'External Wall Insulation',
        'Windows',
        'Doors'
    ]
    
    # Identify rows with missing or placeholder values in the original measures column
    # Considers NaN, None, empty strings, and common placeholders like 'nan'
    missing_mask = result_df[measures_col].isnull() | (result_df[measures_col].astype(str).str.strip() == '') | (result_df[measures_col].astype(str).str.lower() == 'nan')

    # Group by contractor to process missing data
    for contractor in result_df[contractor_col].unique():
        # Get indices for rows that belong to the current contractor AND are missing measures
        contractor_missing_indices = result_df.index[missing_mask & (result_df[contractor_col] == contractor)]
        if not contractor_missing_indices.empty:
            # Get the indices of rows for the current contractor that are missing measures
            num_measures_to_sample = random.randint(4, min(7, len(available_measures)))
            random_measures_sample = random.sample(available_measures, num_measures_to_sample)
            synthesised_measures_str = ', '.join(random_measures_sample)
            result_df.loc[contractor_missing_indices, 'Measures_Cleaned'] = synthesised_measures_str

    # --- 3. Create binary columns for each available measure ---
    for measure in available_measures:
        result_df[measure] = result_df['Measures_Cleaned'].str.contains(measure, case=False, na=False).astype(int)

    return result_df

def synthesise_feedback_openai(df: pd.DataFrame, contractor_col: str, measures_col: str, client: OpenAI) -> pd.DataFrame:
    """
    Generates simulated resident feedback using the OpenAI API with Pydantic for structured output.

    Args:
        df (pd.DataFrame): The input DataFrame.
        contractor_col (str): The column name for the contractor.
        measures_col (str): The column name for the cleaned measures.
        client (OpenAI): An initialized OpenAI client instance.

    Returns:
        pd.DataFrame: The DataFrame with 'Resident_Feedback'columns.
    """
    result_df = df.copy()
    result_df['Resident_Feedback'] = pd.Series(dtype=str)

    # Process each contractor group
    for contractor_name in result_df[contractor_col].unique():
        contractor_df = result_df[result_df[contractor_col] == contractor_name]
        
        overall_sentiment = random.randint(4, 10)
        num_properties = len(contractor_df)
        
        all_works_list = contractor_df[measures_col].str.split(', ').explode().unique()
        all_works_str = ', '.join(filter(None, all_works_list))

        # The user prompt now focuses only on the task, not the format.
        user_prompt = f"""
        A contractor named '{contractor_name}' has completed work on {num_properties} properties.
        The types of work included: {all_works_str}.
        The overall satisfaction for this contractor's work was rated {overall_sentiment} out of 10.

        Please generate {num_properties} unique feedback entries for these properties. They should be varied in their length and style, and variance between them is encouraged.

        Feedback can and most often should cover aspects such as:
        - Quality of work
        - Communication with residents
        - Timeliness of repairs
        - Overall satisfaction with the contractor's service
        - Relationship with the contractor
        - Any specific issues or positive experiences related to the work done
        - Any other relevant details that would be typical in resident feedback.

        Each piece of feedback should roughly reflect the overall sentiment of {overall_sentiment}/10, but with slight variations of plus and minus 2 sentiment points to simulate different personalities and property specific outcomes.
        Each feedback should consist of 5 or 6 sentences.

        **Very important**: 'feedback_list' in the returned JSON object MUST contain {num_properties} entries.
        """

        try:
            print(f"\nGenerating feedback for {num_properties} properties by {contractor_name}...")

            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {
                        "role": "system", ""
                        "content": "You are a data synthesiser creating realistic resident feedback. "
                                    "Your task is to call the provided tool with the generated feedback data."
                    }, 
                    {
                        "role": "user", "content": user_prompt
                    }
                ],
                text_format=FeedbackList,
                temperature=0.8
            )
            
            # Validate the data using the Pydantic model
            validated_data = FeedbackList.model_validate(response.output_parsed)
            feedback_list = validated_data.feedback_list

            if len(feedback_list) == num_properties:
                # Assign validated feedback to the DataFrame
                result_df.loc[contractor_df.index, 'Resident_Feedback'] = [item for item in feedback_list]
            else:
                print(f"Warning: API returned {len(feedback_list)} responses for {contractor_name}, but {num_properties} were expected. Skipping.")

        except Exception as e:
            print(f"An error occurred while generating feedback for {contractor_name}: {e}")

    return result_df

if __name__ == "__main__":
    df = pd.read_excel('data\\shdf_property_summaries_with_llm.xlsx')
    df_completed = df[df["Property Status "] == "Completed"]
    df_completed = assign_contractors(df_completed, 'asset', 'property_type_(parity)', min_group_size=1, group_size=200)
    df = pd.merge(df, df_completed[['uprn', 'Contractor']], on='uprn', how='left')
    df.to_excel('data\\shdf_property_summaries_with_llm.xlsx', index=False)    

    # contractor_col = 'Contractor'
    # measures_col = 'Measures_Cleaned'
    # # 1. Create a sample DataFrame
    # df = pd.read_excel('data\\property_summaries_with_contractors.xlsx')

    # df = assign_contractors(df, 'asset', 'property_type_(parity)')
    # df = process_measures(df, measures_col="Measures", contractor_col=contractor_col)

    # df_with_synthesised_feedback = synthesise_feedback_openai(
    #     df, contractor_col=contractor_col, measures_col=measures_col,
    #     client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # )

    # df_with_synthesised_feedback.to_excel('data\\property_summaries_with_synthesised_feedback.xlsx', index=False)

