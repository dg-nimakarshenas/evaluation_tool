import pandas as pd
import os
from openai import OpenAI
from typing import Tuple, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm # Using tqdm for a user-friendly progress bar


# --- Configuration ---
PROPERTIES_FILE_PATH = 'data\\shdf_property_summaries_with_llm.xlsx'
REPAIRS_FILE_PATH = 'data\\shdf_repairs_data.xlsx'
COMMON_KEY = 'uprn'
OUTPUT_FILE_PATH = 'data\\shdf_property_summaries_with_llm_2.xlsx'
SHARING_CITIES_COMPLETION_DATES = {
    "ERNEST DENCE ESTATE": pd.to_datetime("2022-05-01"),
    "FLAMSTEAD ESTATE": pd.to_datetime("2022-11-01")
}
SHDF_COMPLETION_DATE_COLUMNS = [
    "EWI completion (date)",
    "CWI extraction completion (date)",
    "CWI completion (date)",
    "Loft insulation completion (date)",
    "CWI Brick Repairs completion (date)",
    "Core Hole completion (date)",
    "EWI Render Repairs (date)",
    "Asbestos Removal completion (date)",
    "Windows Installation completion (date)",
    "Doors Installation completion (date)",
    "Door Undercuts completion (date)",
    "Ventilation (dMEV) completion (date)"
]

SHDF_COMPLETION_DICT = {
    "structural": [
        "EWI completion (date)",
        "CWI completion (date)"],
    "damp & mould": [
        "Loft insulation completion (date)",
        "Windows Installation completion (date)",
        "Ventilation (dMEV) completion (date)"],
    "leaks": [
        "Windows Installation completion (date)",
        "CWI completion (date)",
        "EWI completion (date)",
        ],
    "windows & doors": [
        "Windows Installation completion (date)",
        "Doors Installation completion (date)",
    ]    
}

class RepairAssessmentHelper(BaseModel):
    """Defines the severity and frequency assessment for a repair category."""
    severity: Optional[float] = Field(
        description="Overall severity of the issues on a scale from 1 (minor) to 5 (very severe) or 0 if there are NO recorded issues.",
        le=5, ge=0
    )
    frequency: Optional[float] = Field(
        description="The number of times this issue has occurred in the given window, or 0 if there are no recorded issues. BEWARE, there is a chance that there are duplicate entries in the repair history so discount those when counting.",
    )

class RepairAssessment(BaseModel):
    """Contains the assessment for a property before and after the retrofit. for the current repair category."""
    before: RepairAssessmentHelper = Field(description="Assessment of the property before retrofit.")
    after: Optional[RepairAssessmentHelper] = Field(description="Assessment of the property after retrofit. Leave empty if the cut-off date for this category is 'Not Applicable'. If there are no repairs for this specific repair type, default to 0 for both severity and frequency.")


class RetrofitAssessment(BaseModel):
    """The final, top-level model for assessing all repair categories."""
    damp: RepairAssessment = Field(description="Assessment of damp & mould issues in the property. For the severity field, an example of a severe case would be a property with "
    "black mould or mouldy walls in multiple rooms. Low severity would be a single patch of mould in a corner of a room that is easily cleaned up, or no sign of mould at all, default both severity and frequency to 0 if there are no recorded damp or mould related issues."),
    windows_doors: RepairAssessment = Field(description="Assessment of windows and doors in the property. For the severity field, an example of a severe case would be a property with "
    "broken windows or doors that do not close properly. Low severity would be a property with no issues with windows or doors."), 
    leaks: RepairAssessment = Field(description="Assessment of leaks in the property. For the severity field, an example of a severe case would be a property with "
    "severe leaks that cause damage to the property or require significant repairs. Low severity would be a property with no leaks or minor leaks that are easily fixed."),
    structural: RepairAssessment = Field(description="Assessment of structural issues in the property. For the severity field, an example of a severe case would be a property with " \
    "significant structural damage that requires major repairs or poses a safety risk. Low severity would be a property with no structural issues or minor issues that do not affect the safety of the property."),
    summary: str = Field(
        description="A high-level summary of the property's repair history and assessment before and after the retrofit works, ignoring duplicate entries. "
    )

class DampAssessment(BaseModel):
    """Contains the assessment for damp & mould issues in a property."""
    rooms: Optional[int] = Field(
        description="Number of rooms with recorded mould issues."
    )
    severity: Optional[float] = Field(
        description="Overall severity of mould issues in the property on a scale from 1 (minor) to 5 (very severe)."
    )

class StartingComplications(BaseModel):
    """Contains an assessment of further complications in a property."""
    complication_count: Optional[int] = Field(
        description="Total number of distinct complications or defects mentioned."
    )
    severity: Optional[float] = Field(
        description="Overall severity of the complications on a scale from 1 (minor) to 5 (very severe)."
    )

def load_and_merge_data(properties_path: str, repairs_path: str, common_key: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads property and repair data, creates dummy data if files are not found,
    and merges them on a common key.

    Args:
        properties_path: Path to the properties CSV file.
        repairs_path: Path to the repairs CSV file.
        common_key: The column name to join the two dataframes on.

    Returns:
        A tuple containing the merged DataFrame and the original property DataFrame.
        Returns (None, None) if loading fails catastrophically.
    """
    try:
        property_data = pd.read_excel(properties_path)
        repairs_history = pd.read_excel(repairs_path)
        print(f"Successfully loaded '{properties_path}' and '{repairs_path}'.")
    except FileNotFoundError:
        print(f"Could not find '{properties_path}' or '{repairs_path}'.")
        print("Creating dummy data for demonstration purposes...")
        property_data = pd.DataFrame({
            '50_property_ref': ['prop1', 'prop2', 'prop3'],
            'address': ['123 Fake Street, London', '456 Main Avenue, Manchester', '789 Oak Lane, Bristol']
        })
        repairs_history = pd.DataFrame({
            '50_property_ref': ['prop1', 'prop1', 'prop1', 'prop2', 'prop3'],
            '13_works_order_description': [
                'Leaking pipe under kitchen sink fixed.',
                'Boiler pressure checked and topped up during annual service.',
                'Follow-up visit to investigate recurring damp patch in living room. No root cause found.',
                'Replaced broken window pane in the front bedroom.',
                'No repairs on record.'
            ],
            '18_reported_completion_date': ['2023-05-12', '2024-01-20', '2024-06-01', '2023-11-01', '2023-01-01']
        })

    # Merge the two DataFrames using a 'left' join
    merged_data = pd.merge(property_data, repairs_history, on=common_key, how='left')
    print("\n--- Data after merging ---")
    print(merged_data.head())
    return merged_data, property_data

def aggregate_and_format_histories(merged_data: pd.DataFrame, common_key: str) -> pd.DataFrame:
    """
    Aggregates and formats repair histories for each property chronologically.

    Args:
        merged_data: The combined DataFrame of properties and repairs.
        common_key: The property identifier column.

    Returns:
        A DataFrame with one row per property and its aggregated repair history text.
    """
    def _aggregate_repairs_helper(repairs_df: pd.DataFrame) -> str:
        """Helper function to format repair descriptions for a single property."""
        if repairs_df['13_works_order_description'].isnull().all():
            return "No repair history available."
        
        history_lines = []
        for _, row in repairs_df.iterrows():
            if pd.notna(row['18_reported_completion_date']) and pd.notna(row['13_works_order_description']):
                history_lines.append(
                    f"Date: {row['18_reported_completion_date'].date()} - Description: {row['13_works_order_description']}"
                )
        
        return "\n".join(history_lines) if history_lines else "No valid repair history entries found."

    merged_data['18_reported_completion_date'] = pd.to_datetime(merged_data['18_reported_completion_date'], errors='coerce')
    merged_data = merged_data.sort_values(by=[common_key, '18_reported_completion_date'])
    
    property_summary_text = merged_data.groupby(common_key).apply(_aggregate_repairs_helper).reset_index(name='repair_history_text')
    
    print("\n--- Aggregated Repair Histories ---")
    print(property_summary_text)
    return property_summary_text

def generate_llm_summaries(histories_df: pd.DataFrame, client: OpenAI) -> pd.DataFrame:
    """
    Generates a summary for each property's repair history using an LLM.

    Args:
        histories_df: DataFrame containing the aggregated repair history text.
        client: An initialized OpenAI client instance.

    Returns:
        The input DataFrame with an added 'llm_summary' column.
    """
    def _summarize_single_history(history_text: str, llm_client: OpenAI) -> str:
        """Sends a single repair history to the LLM for summarization."""
        if history_text in ["No repair history available.", "No valid repair history entries found."]:
            return history_text

        system_prompt = (
            "You are an expert assistant for property maintenance. "
            "Your task is to provide a brief, high-level summary of a property's repair history "
            "for a surveyor or contractor who needs quick context before a visit. "
            "Focus on recurring issues, major works (like boiler or roof repairs), and recent activity. "
            "Use bullet points for clarity. If there is little or no history, state that clearly."
            "Here is an example of a good summary, and a format you should follow:\n\n"
            "#Example Summary:\n"
            "- Frequent Boiler/Heating Issues: The property has experienced multiple boiler faults and losses of heating/hot water since October 2021, with repairs including sensor and valve replacements. Annual gas servicing has been consistent.\n"
            "- Persistent Door Problems: Both the front and back doors have been consistently difficult to open and close since July 2022, with the issue worsening over time.\n"
            "- Minor Plumbing Issues: There were isolated incidents of a toilet cistern failure (August 2021) and a blocked sink (May 2022)."
            "- Recent Activity: The most recent repairs involved a boiler sensor replacement (September 2023) and a garden clearance (November 2023)."
            "#Instructions:\n"
            "1. Summarize the repair history in a concise manner, focusing on key and or recurring issues.\n"
            "2. You must stick to the following format:\n"
            "   - [Issue Title]: [Description of the issue, including dates and repairs if applicable].\n"   
            "3. Use bullet points for each issue.\n"
        )
        try:
            response = llm_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please summarize the following repair history:\n\n{history_text}"}
                ],
                temperature=0, max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred while generating summary: {e}"

    print("\nGenerating LLM summaries... (This may take a moment)")
    # Use a lambda function to pass the client instance to the apply method
    histories_df['llm_summary'] = histories_df['repair_history_text'].apply(
        lambda text: _summarize_single_history(text, client)
    )
    return histories_df

def create_repair_history_string(property_repairs: pd.DataFrame) -> str:
    """Splits repairs into before/after lists and formats them into a single string."""
    estate = property_repairs['estate'].iloc[0]
    completion_date = SHARING_CITIES_COMPLETION_DATES[estate]

    property_repairs['formatted_repair'] = property_repairs.apply(
        lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
    )

    before_repairs = property_repairs[property_repairs['17_issued_date'] < completion_date]
    after_repairs = property_repairs[property_repairs['17_issued_date'] >= completion_date]

    before_str = "; ".join(before_repairs['formatted_repair'].tolist()) if not before_repairs.empty else "None"
    after_str = "; ".join(after_repairs['formatted_repair'].tolist()) if not after_repairs.empty else "None"

    return f"REPAIRS BEFORE RETROFIT: [{before_str}]\nREPAIRS AFTER RETROFIT: [{after_str}]"

def get_retrofit_assessment(property_history: str, client: OpenAI) -> RetrofitAssessment:
    """Sends the repair history to the LLM and gets a structured assessment."""
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    # Call the LLM with the history, asking for a structured response
    assessment = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14", # Recommended model for complex structured output
        text_format=RetrofitAssessment,
        input=[
            {
                "role": "system",
                "content": "You are an expert in building maintenance. Your task is to assess retrofit effectiveness by analyzing repair histories. Based on the text, categorize repairs into 'damp & mould', 'windows & doors', 'leaks', and 'structural'. "
                "For each category, assess the severity and frequency of issues before and after the retrofit on a scale of 1 to 5. Base your judgment solely on the provided text. If a category has no repairs for a particular repair category, rate its frequency as 0 and severity as 0."
                "Also, provide a detailed summary of the property's repair history and assessment before and after the retrofit works in a detailed paragraph, this should include how occurances "
                "for each repair type may have changed, and the causes of the issues, if the information is present in the repair description, these descriptions will "
                "later be used by a housing expert to assess the impact of the works done in the property so make sure detail is kept regarding each repair type, include this in the 'summary' field in the required schmema."
                "The Schema for the response is as follows:\n"
                f"```json {RetrofitAssessment.model_json_schema()}"
            },
            {
                "role": "user",
                "content": f"Please analyse the following repair history for a property and return your assessment:\n\n{property_history}"
            }
        
        ],
        temperature=0
    )
    return assessment.output_parsed

def assess_sharing_cities(properties_df: pd.DataFrame, repairs_df: pd.DataFrame, llm_client) -> pd.DataFrame:
    """
    Takes property and repair data, gets LLM assessments, and merges results back to the property data.
    """
    if not llm_client:
        print("\nSkipping LLM assessment because OpenAI client could not be initialized.")
        return properties_df

    # 1. Merge 'estate' from properties_df into repairs_df to ensure it's available for processing.
    repairs_with_estate = pd.merge(repairs_df, properties_df[['uprn_', 'estate']], left_on='50_property_ref', right_on="uprn_")

    # 2. Create repair history strings from the detailed repairs_df
    print("\n--- 2. Aggregating Repair History Strings for LLM ---")
    property_summary = repairs_with_estate.groupby('50_property_ref').apply(create_repair_history_string).reset_index(name='repair_history')
    print(property_summary)
    print("-" * 50)
    
    # 3. Iterate through summaries and get LLM assessments
    results = []
    for _, row in property_summary.iterrows():
        print(f"\nAnalyzing property {row['50_property_ref']}...")
        try:
            assessment_result = get_retrofit_assessment(row['repair_history'], llm_client)
            flat_result = {
                '50_property_ref': row['50_property_ref'],
                'damp_before_severity': assessment_result.damp.before.severity,
                'damp_before_frequency': assessment_result.damp.before.frequency,
                'damp_after_severity': assessment_result.damp.after.severity,
                'damp_after_frequency': assessment_result.damp.after.frequency,
                'windows_doors_before_severity': assessment_result.windows_doors.before.severity,
                'windows_doors_before_frequency': assessment_result.windows_doors.before.frequency,
                'windows_doors_after_severity': assessment_result.windows_doors.after.severity,
                'windows_doors_after_frequency': assessment_result.windows_doors.after.frequency,
                'leaks_before_severity': assessment_result.leaks.before.severity,
                'leaks_before_frequency': assessment_result.leaks.before.frequency,
                'leaks_after_severity': assessment_result.leaks.after.severity,
                'leaks_after_frequency': assessment_result.leaks.after.frequency,
                'structural_before_severity': assessment_result.structural.before.severity,
                'structural_before_frequency': assessment_result.structural.before.frequency,
                'structural_after_severity': assessment_result.structural.after.severity,
                'structural_after_frequency': assessment_result.structural.after.frequency,
                'summary': assessment_result.summary
            }
            results.append(flat_result)
            print(f"Successfully assessed property {row['50_property_ref']}.")
        except Exception as e:
            print(f"Could not process property {row['50_property_ref']}: {e}")
    
    if not results:
        return properties_df

    # 4. Merge assessment results back into the original properties DataFrame
    assessment_df = pd.DataFrame(results)
    property_with_assement_df = pd.merge(properties_df, assessment_df, right_on='50_property_ref', left_on='uprn_')
    final_df = pd.merge(property_with_assement_df, property_summary[['50_property_ref', 'repair_history']], on='50_property_ref')
    return final_df

def create_shdf_prompt_and_cutoffs(property_row: pd.Series, all_repairs_df: pd.DataFrame) -> str:
    """
    Creates a detailed prompt for the LLM including dynamic, per-category cut-off dates,
    taking only the first repair description per day.
    """
    prop_id = property_row['uprn']
    property_repairs = all_repairs_df[all_repairs_df['nlpg_uprn_(move_to_end)'] == prop_id]

    # 1. Determine the cut-off date for each issue type
    cutoff_texts = []
    for issue_type, work_cols in SHDF_COMPLETION_DICT.items():
        # Get all relevant completion dates for the property for this issue type
        dates = [pd.to_datetime(property_row.get(col), errors='coerce') for col in work_cols]
        # Filter out any NaT (Not a Time) values
        valid_dates = [d for d in dates if pd.notna(d)]
        
        if valid_dates:
            # Find the latest date among the valid ones
            latest_date = max(valid_dates)
            cutoff_texts.append(f"- {issue_type}: {latest_date.strftime('%Y-%m-%d')}")
        else:
            cutoff_texts.append(f"- {issue_type}: Not Applicable (no relevant work completed)")
    
    cutoff_section = "\n".join(cutoff_texts)

    # 2. Format the entire repair history into a single string
    if not property_repairs.empty:
        # Filter out NaT values from '17_issued_date' before applying strftime
        valid_repairs = property_repairs[pd.notna(property_repairs['17_issued_date'])].copy()

        if not valid_repairs.empty:
            # --- MODIFICATION START ---
            # Ensure the date column is datetime and sort chronologically
            valid_repairs.loc[:, '17_issued_date'] = pd.to_datetime(valid_repairs['17_issued_date'])
            valid_repairs = valid_repairs.sort_values(by='17_issued_date', ascending=True)

            # Group by the calendar date and take the first entry for each day.
            # This handles cases where multiple repairs were logged on the same day.
            unique_day_repairs = valid_repairs.groupby(valid_repairs['17_issued_date'].dt.date).first()
            
            # Format the unique repairs into the desired string format
            repair_texts = unique_day_repairs.apply(
                lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
            ).tolist()
            # --- MODIFICATION END ---
            repair_section = "; ".join(repair_texts)
        else:
            repair_section = "No repairs with valid dates on record."
    else:
        repair_section = "No repairs on record."

    # 3. Construct the final prompt
    prompt = (
        f"Here is the complete repair history for a property: [{repair_section}]\n\n"
        f"Please assess the property based on the following cut-off dates for each category:\n{cutoff_section}\n\n"
        "For each category with a valid date, analyze repairs before and after that date. "
        "For categories marked 'Not Applicable', all repairs fall into the 'before' period, and the 'after' assessment should be a default of 0 for severity and frequency."
    )

    repair_history = f"{repair_section}\n\nCut-off Dates:\n{cutoff_section}"
    return prompt, repair_history

def get_shdf_retrofit_assessment(prompt: str, client) -> RetrofitAssessment:
    """Sends the detailed SHDF prompt to the LLM and gets a structured assessment."""
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    system_prompt = f"""You are an expert in building maintenance. Your task is to assess retrofit effectiveness by analyzing a property's repair history against specific work completion dates.
      You will be given a full repair history and a list of cut-off dates for 4 different issue categories ('damp & mould', 'windows & doors', 'leaks', and 'structural'). 
      For each category, you must assess the severity and frequency of issues 'before' and 'after' its specific cut-off date. 
      #Key Rules:
      1. leave the 'after' field empty ONLY IF the cut-off date is 'Not Applicable' for that category, otherwise, default the 'after' severity and frequency to 1.
      2. If a category has no recorded repairs for the particular repair type before or after the cut off point, rate its frequency as 0 and severity as 0, DO NOT leave it empty. 
      3. PLEASE put the frequency and severity as 0 if there are no recorded issues for that category.
      The Schema for the response is as follows:\n
    ```json {RetrofitAssessment.model_json_schema()}"""

    assessment = client.responses.parse(
        model="gpt-4.1-mini", # Recommended model for complex structured output
        text_format=RetrofitAssessment,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        
        ],
        temperature=0
    )
    return assessment.output_parsed

def process_shdf_assessments(properties_df: pd.DataFrame, repairs_df: pd.DataFrame, client) -> pd.DataFrame:
    """
    Main function to process SHDF data, get LLM assessments, and merge results.
    """
    if not client:
        print("\nSkipping LLM assessment because OpenAI client could not be initialized.")
        return properties_df

    # Filter for properties with 'Completed' status before processing
    if 'Property Status ' in properties_df.columns:
        properties_to_process = properties_df[properties_df['Property Status '] == 'Completed'].copy()
        print(f"\nFound {len(properties_to_process)} properties with status 'Completed'. Processing these...")
    else:
        print("\n'Property Status ' column not found. Processing all properties.")
        properties_to_process = properties_df.copy()

    results = []
    # Iterate only over the filtered DataFrame
    for _, row in properties_to_process.iterrows():
        prop_id = row['uprn']
        print(f"\nAnalyzing property {prop_id}...")
        
        # 1. Create the dynamic prompt for this property
        prompt, repair_history = create_shdf_prompt_and_cutoffs(row, repairs_df)
        print("   - Generated Prompt Snippet:")
        print(f"   {prompt.splitlines()[2]}\n   {prompt.splitlines()[3]}...") # Print a snippet for verification
        
        # 2. Get the assessment from the LLM
        try:
            assessment_result = get_shdf_retrofit_assessment(prompt, client)
            flat_result = {'uprn': prop_id, 'repair_history': repair_history}

            # Helper function to safely get severity/frequency, handling None/NaN
            def get_value(assessment_helper: Optional[RepairAssessmentHelper], field_name: str):
                if assessment_helper:
                    # Access attribute dynamically
                    value = getattr(assessment_helper, field_name)
                    # Convert None to NaN for consistency when putting into DataFrame
                    return value if value is not None else pd.NA
                return pd.NA # If the helper itself is None, return NaN for its fields

            # Process 'damp' category
            flat_result['damp_before_severity'] = get_value(assessment_result.damp.before, 'severity')
            flat_result['damp_before_frequency'] = get_value(assessment_result.damp.before, 'frequency')
            flat_result['damp_after_severity'] = get_value(assessment_result.damp.after, 'severity')
            flat_result['damp_after_frequency'] = get_value(assessment_result.damp.after, 'frequency')

            # Process 'windows_doors' category
            flat_result['windows_doors_before_severity'] = get_value(assessment_result.windows_doors.before, 'severity')
            flat_result['windows_doors_before_frequency'] = get_value(assessment_result.windows_doors.before, 'frequency')
            flat_result['windows_doors_after_severity'] = get_value(assessment_result.windows_doors.after, 'severity')
            flat_result['windows_doors_after_frequency'] = get_value(assessment_result.windows_doors.after, 'frequency')

            # Process 'leaks' category
            flat_result['leaks_before_severity'] = get_value(assessment_result.leaks.before, 'severity')
            flat_result['leaks_before_frequency'] = get_value(assessment_result.leaks.before, 'frequency')
            flat_result['leaks_after_severity'] = get_value(assessment_result.leaks.after, 'severity')
            flat_result['leaks_after_frequency'] = get_value(assessment_result.leaks.after, 'frequency')

            # Process 'structural' category
            flat_result['structural_before_severity'] = get_value(assessment_result.structural.before, 'severity')
            flat_result['structural_before_frequency'] = get_value(assessment_result.structural.before, 'frequency')
            flat_result['structural_after_severity'] = get_value(assessment_result.structural.after, 'severity')
            flat_result['structural_after_frequency'] = get_value(assessment_result.structural.after, 'frequency')

            # Add summary
            flat_result['summary'] = assessment_result.summary

            results.append(flat_result)
            print(f"    - Successfully assessed property {prop_id}.")
        except Exception as e:
            print(f"   - Could not process property {prop_id}: {e}")
    
    if not results:
        return properties_df

    # 3. Merge assessment results back into the original properties DataFrame
    assessment_df = pd.DataFrame(results)
    final_df = pd.merge(properties_df, assessment_df, on='uprn')
    return final_df

def get_damp_assessment(property_history: str, client: OpenAI) -> DampAssessment:
    """Sends the repair history to the LLM and gets a structured assessment."""
    if not client:
        raise ConnectionError("OpenAI client not initialized.")
    system_prompt = f"""
        You are an expert building surveyor specializing in damp and mould assessment. Your task is to analyze a text description of damp and mould in a property and provide a structured assessment.

        Base your assessment *solely* on the provided text.

        You must determine two key metrics:
        1.  `rooms`: The total number of unique rooms or areas where damp or mould is mentioned. Count distinct locations like 'Hall', 'Bathroom', 'Bedroom 1', 'Kitchen', 'W/C', and 'Stairs and Landing' as separate rooms. If a room is mentioned multiple times, it only counts as one affected room.
        2.  `severity`: The overall severity of the issue on a scale of 1 (minor) to 5 (very severe). Consider both the number of affected rooms and the described extent of the problem in your rating. A single, small patch of mould in one room is a 1. Severe, widespread mould across multiple rooms is a 5.

        Here are some examples to guide your assessment:

        ---
        **Example 1:**
        *Input Text:* "Hall - Damp below windows. Bathroom - condensation on window, mould on side wall. Bedroom 3 - mould on front side wall."
        *Correct Assessment:* {{"rooms": 3, "severity": 3.0}}
        *Reasoning:* Three distinct rooms are affected with localized issues.

        ---
        **Example 2:**
        *Input Text:* "Bathroom - mould on corner of outside wall. Bedroom 1 - wall damp, vent covered, dampness could be due to failure of gutter."
        *Correct Assessment:* {{"rooms": 2, "severity": 2.0}}
        *Reasoning:* Two rooms are affected. The issues are relatively minor or localized.

        ---
        **Example 3:**
        *Input Text:* "Kitchen - Mould on the ceiling near the extractor, Signs of condensation problems on the top window reveal. Dining Room - There is an open archway between the Dining room and Living room and an Artex ceiling is present. Living Room - There is an open archway between the Living room and Dining room. Artex ceiling is present. Stairs and Landing - Artex ceiling in Hallway. Bedroom 1 - the wallpaper has been removed, Signs of condensation problems on the window frames and reveals. Bedroom 2 - Damp and mould on the walls, ceiling and window reveals, the door has been removed from the frame. Bedroom 3 - The door has been removed from the frame, Artex ceiling. W/C - Signs of mould around the edges of the window frame, Artex ceiling. Bathroom - Mould on the ceiling and tiles, Artex ceiling."
        *Correct Assessment:* {{"rooms": 7, "severity": 5.0}}
        *Reasoning:* At least seven distinct areas (Kitchen, Dining/Living, Stairs/Landing, Bed 1, Bed 2, W/C, Bathroom) show signs of issues. The description for Bedroom 2 is particularly severe ("Damp and mould on the walls, ceiling and window reveals"), justifying the highest severity rating.

        ---
        **Example 4:**
        *Input Text:* "Mould to lower corner by front wall and communal passage"
        *Correct Assessment:* {{"rooms": 1, "severity": 1.0}}
        *Reasoning:* The issue is confined to one general area and is localized ("lower corner").

        ---

        Now, analyze the user's input. The required JSON schema for your response is:
        {DampAssessment.model_json_schema()}
        """
    # Call the LLM with the history, asking for a structured response
    assessment = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14", # Recommended model for complex structured output
        text_format=DampAssessment,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Please analyse the following repair history for a property and return your assessment:\n\n{property_history}"
            }
        
        ],
        temperature=0
    )
    return assessment.output_parsed

def process_damp_assessments(df: pd.DataFrame, input_column_name: str, client: OpenAI) -> pd.DataFrame:
    """
    Processes a DataFrame column with mould descriptions, gets assessments via an LLM,
    and adds the results to new columns.

    Args:
        df: The pandas DataFrame to process.
        input_column_name: The name of the column containing the mould descriptions.
        client: An initialized OpenAI client.

    Returns:
        The DataFrame with two new columns: 'damp_wates_locations' and 'damp_wates_severity'.
    """
    if input_column_name not in df.columns:
        raise ValueError(f"Column '{input_column_name}' not found in the DataFrame.")

    locations_list = []
    severity_list = []

    # Create a filtered series to iterate over non-NaN/non-null values
    valid_descriptions = df[df[input_column_name].notna()][input_column_name]

    # Use tqdm for a progress bar, which is helpful for long-running processes
    for description in tqdm(valid_descriptions, desc="Assessing Damp Issues"):
        if isinstance(description, str) and description.strip():
            try:
                assessment = get_damp_assessment(description, client)
                locations_list.append(assessment.rooms)
                severity_list.append(assessment.severity)
            except Exception as e:
                print(f"Could not process description: '{description}'. Error: {e}")
                locations_list.append(None)
                severity_list.append(None)
        else:
            # This case handles values that are not strings or are empty strings
            locations_list.append(None)
            severity_list.append(None)
    
    # Create new series from the results with the same index as the valid descriptions
    locations_series = pd.Series(locations_list, index=valid_descriptions.index)
    severity_series = pd.Series(severity_list, index=valid_descriptions.index)

    # Map the results back to the original DataFrame, ensuring alignment
    df['damp_wates_locations'] = locations_series
    df['damp_wates_severity'] = severity_series

    return df

def get_complication_assessment(complication_description: str, client: OpenAI) -> StartingComplications:
    """
    Sends a description of property complications to an LLM and gets a structured assessment.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    system_prompt = f"""
        You are an expert building surveyor. Your task is to analyze a text description of various property defects and provide a structured assessment. Base your assessment *solely* on the provided text.

        You must determine two key metrics:
        1.  `complication_count`: The total number of distinct issues, defects, or complications mentioned. For example, 'broken fan' is one issue, 'cracks on ceiling' is another.
        2.  `severity`: The overall severity of the combined issues on a scale of 1 (minor/cosmetic) to 5 (very severe/structural/urgent). Consider the number and nature of the problems. A single instance of flaking paint is a 1. Multiple structural issues like overflowing gutters and large gaps in brickwork would be a 5.

        Here are some examples to guide your assessment:
        ---
        **Example 1:**
        *Input Text:* "Kitchen-extractor fan not working, locking mechanism can only lock from inside. Hall-window broken single glazed,front door not closing properly,undercuts as per adjoining rooms. Bathroom-extractor fan broken. Landing-window has a sticker on, undercuts as per adjoining rooms. External elevation-guttering overflows."
        *Correct Assessment:* {{"complication_count": 7, "severity": 4.0}}
        *Reasoning:* Multiple issues including two broken fans, broken window, door issue, and overflowing gutters suggest high count and severity.
        ---
        **Example 2:**
        *Input Text:* "Lounge diner - cracks on the ceiling. Bedroom 3 - large gap in window letting in cold wind through it. Bathroom - extractor fan not working, electric shower has not worked for years."
        *Correct Assessment:* {{"complication_count": 4, "severity": 3.0}}
        *Reasoning:* Four distinct issues of moderate severity.
        ---
        **Example 3:**
        *Input Text:* "Toilet - paint flaking above toilet. Vent covered and failure of gutter. During heavy rain bottom hopper is too small"
        *Correct Assessment:* {{"complication_count": 4, "severity": 3.0}}
        *Reasoning:* Gutter and vent issues are significant, raising the severity despite the seemingly minor paint flaking.
        ---
        **Example 4:**
        *Input Text:* "Gap in the brickwork, Vent - Very dirty in Right Elevation / Corridor. Staining on the brickwork due to moisture in Rear Elevation."
        *Correct Assessment:* {{"complication_count": 3, "severity": 4.0}}
        *Reasoning:* A gap in brickwork is a serious structural issue, warranting a high severity rating even with a lower count of problems.
        ---
        Now, analyze the user's input. The required JSON schema for your response is:
        {StartingComplications.model_json_schema()}
        """
    assessment = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14", # Recommended model for complex structured output
        text_format=StartingComplications,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Please analyse the following set of initial complications for the property and return your assessment:\n\n{complication_description}"
            }
        
        ],
        temperature=0
    )
    return assessment.output_parsed

def process_complication_assessments(df: pd.DataFrame, input_column_name: str, client: OpenAI) -> pd.DataFrame:
    """
    Processes a DataFrame column with complication descriptions, gets assessments via an LLM,
    and adds the results to new columns.
    """
    if input_column_name not in df.columns:
        raise ValueError(f"Column '{input_column_name}' not found in the DataFrame.")

    count_list = []
    severity_list = []
    valid_descriptions = df[df[input_column_name].notna()][input_column_name]

    for description in tqdm(valid_descriptions, desc="Assessing Complications"):
        if isinstance(description, str) and description.strip():
            try:
                assessment = get_complication_assessment(description, client)
                count_list.append(assessment.complication_count)
                severity_list.append(assessment.severity)
            except Exception as e:
                print(f"Could not process description: '{description}'. Error: {e}")
                count_list.append(None)
                severity_list.append(None)
        else:
            count_list.append(None)
            severity_list.append(None)
    
    count_series = pd.Series(count_list, index=valid_descriptions.index)
    severity_series = pd.Series(severity_list, index=valid_descriptions.index)

    # Add results to new columns in the original DataFrame
    df['complications_count'] = count_series
    df['complications_severity'] = severity_series

    return df

def generate_sharing_cities_summaries():
    """Generates summaries for properties in the Sharing Cities dataset using LLM assessments.
    """
    # Load the Sharing Cities data
    print("\n--- Loading Sharing Cities Data ---")
    sharing_cities_df = pd.read_excel("data\\sharing_cities_property_list.xlsx")
    repairs_df = pd.read_excel("data\\sharing_cities_repairs_data.xlsx")

    if sharing_cities_df.empty or repairs_df.empty:
        print("No data found in Sharing Cities files. Please check the input files.")
        return
    print("\n--- Generating Sharing Cities Summaries ---")
    print(f"Loaded {len(sharing_cities_df)} properties and {len(repairs_df)} repairs.")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client:
        print("OpenAI client could not be initialized. Skipping LLM assessment.")
        return sharing_cities_df
    
    # Process the Sharing Cities data to get LLM assessments
    print("\n--- Assessing Sharing Cities Properties ---")
    final_df = assess_sharing_cities(sharing_cities_df, repairs_df, client)
    if final_df.empty:
        print("No valid assessments were generated. Please check the input data.")
        return
    print("--- Saving Final Results ---")
    final_df.to_excel("data\\sharing_cities_property_summaries.xlsx", index=False)

def generate_shdf_summaries():
    """Generates summaries for properties in the SHDF dataset using LLM assessments.
    """
    # Load the SHDF data
    print("\n--- Loading SHDF Data ---")
    properties_df = pd.read_excel(PROPERTIES_FILE_PATH)
    repairs_df = pd.read_excel(REPAIRS_FILE_PATH)

    if properties_df.empty or repairs_df.empty:
        print("No data found in SHDF files. Please check the input files.")
        return
    print(f"Loaded {len(properties_df)} properties and {len(repairs_df)} repairs.")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client:
        print("OpenAI client could not be initialized. Skipping LLM assessment.")
        return properties_df
    
    # Process the SHDF data to get LLM assessments
    print("\n--- Assessing SHDF Properties ---")
    final_df = process_shdf_assessments(properties_df, repairs_df, client)
    if final_df.empty:
        print("No valid assessments were generated. Please check the input data.")
        return
    print("--- Saving Final Results ---")
    final_df.to_excel(OUTPUT_FILE_PATH, index=False)

def run_wates_analysis():
    df = pd.read_excel("data\\shdf_property_summaries_with_llm.xlsx")
    #df = process_damp_assessments(df, 'Mould Location and if Urgent', OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    df = process_complication_assessments(df, 'Further Complications', OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))
    df.to_excel("data\\shdf_property_summaries_with_llm.xlsx", index=False)

def main():
    """
    Main function to execute the data processing and summarization pipeline.
    """
    # --- Setup ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set your API key to generate summaries. Exiting.")
        return
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return

    # --- Pipeline Execution ---
    # 1. Load and merge data
    merged_data, property_data = load_and_merge_data(PROPERTIES_FILE_PATH, REPAIRS_FILE_PATH, COMMON_KEY)
    if merged_data is None:
        return

    # 2. Aggregate histories
    aggregated_histories = aggregate_and_format_histories(merged_data, COMMON_KEY)

    # 3. Generate LLM summaries
    summaries_df = generate_llm_summaries(aggregated_histories, client)

    # 4. Combine results and save
    final_output = pd.merge(property_data, summaries_df, on=COMMON_KEY, how='left')
    
    print("\n--- Final Output with LLM Summaries ---")
    with pd.option_context('display.max_colwidth', None):
        print(final_output)

    final_output.to_excel(OUTPUT_FILE_PATH, index=False)
    print(f"\nSuccessfully saved the final results to '{OUTPUT_FILE_PATH}'")


if __name__ == "__main__":
    #main()
    #generate_sharing_cities_summaries()
    generate_shdf_summaries()
    #run_wates_analysis()