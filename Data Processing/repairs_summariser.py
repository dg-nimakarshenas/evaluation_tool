import pandas as pd
import os
from openai import OpenAI
from typing import Tuple, Optional, Dict
from pydantic import BaseModel, Field
from tqdm import tqdm 
from datetime import datetime


# --- Configuration ---
PROPERTIES_FILE_PATH = 'data\\shdf_property_summaries_with_llm.xlsx'
REPAIRS_FILE_PATH = 'data\\shdf_repairs_data.xlsx'
COMMON_KEY = 'uprn'
OUTPUT_FILE_PATH = 'data\\shdf_property_summaries_with_llm.xlsx'
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

class OverallRepairSummary(BaseModel):
    """Summary model for overall repair analysis across all categories."""
    summary: str = Field(
        description="A comprehensive summary of the property's repair history across all categories (damp, windows/doors, leaks, structural), "
        "including how each category changed after their respective interventions, overall effectiveness assessment, "
        "and detailed information that will be used by housing experts to assess the impact of works."
    )


class SHDFRepairAssessment(BaseModel):
    """Assessment model for a single repair category in one time period."""
    severity: Optional[float] = Field(
        description="Overall severity of the issues on a scale from 1 (minor) to 5 (very severe) or 0 if there are NO recorded issues.",
        le=5, ge=0
    )
    frequency: Optional[float] = Field(
        description="The number of times this issue has occurred in the given list of repairs, or 0 if there are no recorded issues. BEWARE, there is a chance that there are duplicate entries in the repair history so discount those when counting.",
        ge=0
    )

class RepairTypeKeyTakeaways(BaseModel):
    """Key takeaways model for each repair type after retrofit analysis."""
    damp_mould_takeaway: str = Field(
        description="Key takeaway for damp and mould issues: summarize the damp/mould problems before and after the retrofit, "
        "including effectiveness, any remaining issues, and overall trend. Keep to a few bullet points."
    )
    windows_doors_takeaway: str = Field(
        description="Key takeaway for windows and doors issues: summarize the window/door problems before and after the retrofit, "
        "including effectiveness, any remaining issues, and overall trend. Keep to a few bullet points."
    )
    leaks_takeaway: str = Field(
        description="Key takeaway for leak issues: summarize the leak problems before and after the retrofit, "
        "including effectiveness, any remaining issues, and overall trend. Keep to a few bullet points."
    )
    structural_takeaway: str = Field(
        description="Key takeaway for structural issues: summarize the structural problems before and after the retrofit, "
        "including effectiveness, any remaining issues, and overall trend. Keep to a few bullet points. "
    )


class RepairAssessment(BaseModel):
    """Defines the severity and frequency assessment for a repair category."""
    severity: Optional[float] = Field(
        description="Overall severity of the issues on a scale from 1 (minor) to 5 (very severe) or 0 if there are NO recorded issues.",
        le=5, ge=0
    )
    frequency: Optional[float] = Field(
        description="The number of times this issue has occurred in the given list of repairs, or 0 if there are no recorded issues for the current repair category",
        ge=0
    )


class RetrofitAssessment(BaseModel):
    """The final, top-level model for assessing all repair categories."""
    damp: RepairAssessment = Field(description="Assessment of damp & mould issues in the property. For the severity field, an example of a severe case would be a property with "
    "black mould or mouldy walls in multiple rooms. Low severity would be a single patch of mould in a corner of a room that is easily cleaned up, or no sign of mould at all, "
    "default both severity and frequency to 0 if there are no recorded damp or mould related issues. ONLY count it IF it is explicitly mentioned as a mould or damp issue in the repair history."),
    windows_doors: RepairAssessment = Field(description="Assessment of windows and doors in the property. For the severity field, an example of a severe case would be a property with "
    "broken windows or doors that do not close properly. Low severity would be a property with no issues with windows or doors. ONLY count it IF it is explicitly mentioned as a window "
    "or door issue in the repair history, IGNORE lost keys or minor issues that do not affect the functionality of the window or door."), 
    leaks: RepairAssessment = Field(description="Assessment of leaks in the property. For the severity field, an example of a severe case would be a property with "\
    "severe leaks that cause damage to the property or require significant repairs. Low severity would be a property with no leaks or minor leaks that are easily fixed. " \
    "ONLY count it IF it is explicitly mentioned as a leak in the repair history. Igore blockages or minor plumbing issues that do not cause leaks."),
    structural: RepairAssessment = Field(description="Assessment of structural issues in the property. For the severity field, an example of a severe case would be a property with "
    "significant structural damage that requires major repairs or poses a safety risk. Low severity would be a property with no structural issues or minor issues that do not affect the safety of the property."
    "ONLY count it IF it is explicitly mentioned as structural or wall damage in the repair history.")

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

def create_repair_history_lists(property_repairs: pd.DataFrame) -> tuple[str, str]:
    """
    Splits repairs into before/after lists based on the retrofit completion date
    and formats them into two separate strings.
    If multiple repairs occur on the same day, only the first one is kept.

    Args:
        property_repairs: DataFrame of repairs for a single property.

    Returns:
        A tuple containing two strings: (before_repairs_str, after_repairs_str).
    """
    # This assumes SHARING_CITIES_COMPLETION_DATES is a dict mapping 'estate' to a completion date
    estate = property_repairs['estate'].iloc[0]
    completion_date = SHARING_CITIES_COMPLETION_DATES[estate]

    # --- FIX: De-duplicate repairs correctly by date ---
    # Sort by the full timestamp to ensure we keep the earliest entry of the day, and create a copy to avoid warnings.
    property_repairs = property_repairs.sort_values('17_issued_date').copy()
    
    # Create a temporary column with just the date part for de-duplication.
    property_repairs['temp_date'] = property_repairs['17_issued_date'].dt.date
    
    # Drop duplicates based on the new temporary date column, keeping the first entry.
    property_repairs_deduped = property_repairs.drop_duplicates(subset='temp_date', keep='first')
    # --- END FIX ---

    # Format each repair row into a descriptive string using the de-duplicated dataframe
    property_repairs_deduped['formatted_repair'] = property_repairs_deduped.apply(
        lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
    )

    # Split the de-duplicated DataFrame into 'before' and 'after' based on the completion date
    before_df = property_repairs_deduped[property_repairs_deduped['17_issued_date'] < completion_date]
    after_df = property_repairs_deduped[property_repairs_deduped['17_issued_date'] >= completion_date]

    # Create semicolon-separated strings, handling cases with no repairs
    before_str = "; ".join(before_df['formatted_repair'].tolist()) if not before_df.empty else "None"
    after_str = "; ".join(after_df['formatted_repair'].tolist()) if not after_df.empty else "None"

    return before_str, after_str

# --- LLM Interaction Functions ---

def get_repair_assessment(repair_history: str, client) -> RetrofitAssessment:
    """
    Sends a single repair history (either before or after) to the LLM for blind assessment.

    Args:
        repair_history: A string containing a list of repairs for one period.
        client: The LLM client instance.

    Returns:
        A RetrofitAssessment object with severity and frequency for each category.
    """
    if not client:
        raise ConnectionError("LLM client not initialized.")
    
    # If there's no repair history, return a default object with all zeros.
    if repair_history == "None":
        return RetrofitAssessment(
            damp=RepairAssessment(severity=0, frequency=0),
            windows_doors=RepairAssessment(severity=0, frequency=0),
            leaks=RepairAssessment(severity=0, frequency=0),
            structural=RepairAssessment(severity=0, frequency=0)
        )

    # Call the LLM with the history, asking for a structured response based on the new schema.
    # Note: The prompt is now simplified to focus only on the provided text, with no "before/after" context.
    assessment = client.responses.parse(
        model="gpt-4.1-mini",
        text_format=RetrofitAssessment,
        input=[
            {
                "role": "system",
                "content": "You are an expert in building maintenance. Your task is to analyze a list of repairs. "
                           "Based ONLY on the text provided, categorize each repair into 'damp & mould', 'windows & doors', 'leaks', and 'structural'. "
                           "For each category, assess the overall severity on a scale of 0 to 5 and count the frequency of issues. "
                           "If a category has no relevant repairs, you MUST rate its frequency as 0 and severity as 0. "
                           "Base your judgment solely and strictly on the provided text."
                           "ONLY MARK A REPAIR AS BELONGING TO A CERTAIN CATEGORY if it is EXPLICITLY mentioned as such in the repair history, "
                           "for expample, 'leak in the roof' would be counted as a leak, but 'blocked sink' would not be counted as a leak, and "
                           "a damp or mould issue is NOT inferred from a leak unless in the SAME repair description for that day it is EXPLICITLY mentions a damp and mould issue"
                           f"The Schema for the response is as follows:\n```json {RetrofitAssessment.model_json_schema()}```"
            },
            {
                "role": "user",
                "content": f"Please analyse the following repair history and return your assessment:\n\n{repair_history}"
            }
        ],
        temperature=0
    )
    return assessment.output_parsed

def get_comparison_summary(before_history: str, after_history: str, client) -> str:
    """
    Generates a detailed comparative summary by sending both histories to an LLM.

    Args:
        before_history: The string of repairs before the retrofit.
        after_history: The string of repairs after the retrofit.
        client: The LLM client instance.

    Returns:
        A detailed paragraph summarizing the changes.
    """
    if not client:
        raise ConnectionError("LLM client not initialized.")

    # Call the LLM with a prompt specifically for generating a comparative summary.
    summary_response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14", # Using a powerful model for nuanced summary generation
        messages=[
            {
                "role": "system",
                "content": "You are an expert in building maintenance. Your task is to write a summary comparing a property's repair history before and after a retrofit. "
                           "Your summary should be a detailed paragraph for a housing expert. "
                           "Analyze how the occurrences for each repair type (damp, leaks, structural, windows/doors) may have changed between the two periods. "
                           "If possible, infer the causes of issues based on the repair descriptions. "
                           "Directly compare the 'BEFORE' and 'AFTER' lists to highlight improvements or new problems."
            },
            {
                "role": "user",
                "content": f"Please provide a comparative assessment of the following repair histories.\n\n"
                           f"REPAIRS BEFORE RETROFIT: [{before_history}]\n\n"
                           f"REPAIRS AFTER RETROFIT: [{after_history}]"
            }
        ],
        temperature=0.1
    )
    return summary_response.choices[0].message.content

# --- Main Orchestration Function ---

def assess_sharing_cities(properties_df: pd.DataFrame, repairs_df: pd.DataFrame, llm_client) -> pd.DataFrame:
    """
    Orchestrates the end-to-end assessment process:
    1. Splits repairs into 'before' and 'after' periods.
    2. Gets independent LLM assessments for each period.
    3. Generates a comparative summary.
    4. Merges all results back into the property data.
    """
    if not llm_client:
        print("\nSkipping LLM assessment because the client could not be initialized.")
        return properties_df

    # 1. Merge 'estate' from properties_df into repairs_df.
    repairs_with_estate = pd.merge(repairs_df, properties_df[['uprn_', 'estate']], left_on='50_property_ref', right_on="uprn_")

    # 2. Group repairs by property reference.
    grouped_repairs = repairs_with_estate.groupby('50_property_ref')
    
    results = []
    # 3. Iterate through each property, get assessments, and generate summary.
    for property_ref, group_df in tqdm(grouped_repairs, desc="Assessing Properties"):
        print(f"\nAnalyzing property {property_ref}...")
        try:
            # Get the completion date to calculate the time spans
            estate = group_df['estate'].iloc[0]
            completion_date_ts = SHARING_CITIES_COMPLETION_DATES[estate]
            
            # Define the overall period start and end dates
            period_start_date = datetime(2021, 1, 1).date()
            period_end_date = datetime(2024, 12, 31).date()

            # Ensure completion_date is a date object for comparison
            completion_date = completion_date_ts.date() if isinstance(completion_date_ts, pd.Timestamp) else completion_date_ts

            # Calculate the number of days in each period for normalization
            days_before = (completion_date - period_start_date).days
            days_after = (period_end_date - completion_date).days

            # Separate repairs into before/after strings
            before_str, after_str = create_repair_history_lists(group_df)

            before_assessment = get_repair_assessment(before_str, llm_client)
            after_assessment = get_repair_assessment(after_str, llm_client)
            summary = get_comparison_summary(before_str, after_str, llm_client)

            def annualize(frequency, days):
                if frequency is None or frequency == 0 or days <= 0:
                    return 0.0
                return (frequency / days) * 365

            # Flatten all results, normalizing the frequency to a per-year rate
            flat_result = {
                '50_property_ref': property_ref,
                'repair_history_before': before_str,
                'repair_history_after': after_str,
                'damp_before_severity': before_assessment.damp.severity,
                'damp_before_frequency': annualize(before_assessment.damp.frequency, days_before),
                'damp_after_severity': after_assessment.damp.severity,
                'damp_after_frequency': annualize(after_assessment.damp.frequency, days_after),
                'windows_doors_before_severity': before_assessment.windows_doors.severity,
                'windows_doors_before_frequency': annualize(before_assessment.windows_doors.frequency, days_before),
                'windows_doors_after_severity': after_assessment.windows_doors.severity,
                'windows_doors_after_frequency': annualize(after_assessment.windows_doors.frequency, days_after),
                'leaks_before_severity': before_assessment.leaks.severity,
                'leaks_before_frequency': annualize(before_assessment.leaks.frequency, days_before),
                'leaks_after_severity': after_assessment.leaks.severity,
                'leaks_after_frequency': annualize(after_assessment.leaks.frequency, days_after),
                'structural_before_severity': before_assessment.structural.severity,
                'structural_before_frequency': annualize(before_assessment.structural.frequency, days_before),
                'structural_after_severity': after_assessment.structural.severity,
                'structural_after_frequency': annualize(after_assessment.structural.frequency, days_after),
                'summary': summary
            }
            results.append(flat_result)
            print(f"Successfully assessed property {property_ref}.")

        except Exception as e:
            print(f"Could not process property {property_ref}: {e}")
    
    if not results:
        print("No properties were successfully assessed.")
        return properties_df

    # 4. Merge assessment results back into the original properties DataFrame
    assessment_df = pd.DataFrame(results)
    final_df = pd.merge(properties_df, assessment_df, left_on='uprn_', right_on='50_property_ref')
    
    return final_df


def create_category_repair_histories(property_row: pd.Series, all_repairs_df: pd.DataFrame, 
                                   issue_type: str, work_cols: list) -> Tuple[str, str, Optional[str]]:
    """
    Creates separate before/after repair histories for a specific category.
    Returns: (before_repairs, after_repairs, cutoff_date_str)
    """
    prop_id = property_row['uprn']
    property_repairs = all_repairs_df[all_repairs_df['nlpg_uprn_(move_to_end)'] == prop_id]
    
    # Determine cutoff date for this category
    dates = [pd.to_datetime(property_row.get(col), errors='coerce') for col in work_cols]
    valid_dates = [d for d in dates if pd.notna(d)]
    
    if not valid_dates:
        # No work completed for this category - all repairs are "before"
        if not property_repairs.empty:
            valid_repairs = property_repairs[pd.notna(property_repairs['17_issued_date'])].copy()
            if not valid_repairs.empty:
                valid_repairs.loc[:, '17_issued_date'] = pd.to_datetime(valid_repairs['17_issued_date'])
                valid_repairs = valid_repairs.sort_values(by='17_issued_date', ascending=True)
                unique_day_repairs = valid_repairs.groupby(valid_repairs['17_issued_date'].dt.date).first()
                repair_texts = unique_day_repairs.apply(
                    lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
                ).tolist()
                before_repairs = "; ".join(repair_texts)
            else:
                before_repairs = "None"
        else:
            before_repairs = "None"
        return before_repairs, "None", None
    
    # Get the latest completion date for this category
    cutoff_date = max(valid_dates)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    
    # Filter and format repairs
    if not property_repairs.empty:
        valid_repairs = property_repairs[pd.notna(property_repairs['17_issued_date'])].copy()
        if not valid_repairs.empty:
            valid_repairs.loc[:, '17_issued_date'] = pd.to_datetime(valid_repairs['17_issued_date'])
            valid_repairs = valid_repairs.sort_values(by='17_issued_date', ascending=True)
            unique_day_repairs = valid_repairs.groupby(valid_repairs['17_issued_date'].dt.date).first()
            
            # Split into before/after based on cutoff
            before_mask = unique_day_repairs['17_issued_date'] < cutoff_date
            after_mask = unique_day_repairs['17_issued_date'] >= cutoff_date
            
            before_df = unique_day_repairs[before_mask]
            after_df = unique_day_repairs[after_mask]
            
            # Format repair strings
            if not before_df.empty:
                before_texts = before_df.apply(
                    lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
                ).tolist()
                before_repairs = "; ".join(before_texts)
            else:
                before_repairs = "None"
                
            if not after_df.empty:
                after_texts = after_df.apply(
                    lambda row: f"({row['17_issued_date'].date()}) {row['13_works_order_description']}", axis=1
                ).tolist()
                after_repairs = "; ".join(after_texts)
            else:
                after_repairs = "None"
        else:
            before_repairs = "None"
            after_repairs = "None"
    else:
        before_repairs = "None"
        after_repairs = "None"
    
    return before_repairs, after_repairs, cutoff_date_str


def get_category_period_assessment(repair_history: str, category_type: str, client: OpenAI) -> SHDFRepairAssessment:
    """Analyzes a single category's repair history for one time period."""
    if not client:
        raise ConnectionError("OpenAI client not initialized.")
    
    # Category-specific instructions
    category_instructions = {
        'damp & mould': "Assess damp and mould issues. Only count repairs explicitly mentioning mould, damp, condensation, or moisture problems. Severe cases involve black mould or mouldy walls in multiple rooms.",
        'windows & doors': "Assess windows and doors issues. Only count repairs explicitly mentioning window or door functionality problems. Ignore lost keys or minor issues. Severe cases involve broken windows or doors that don't close properly.",
        'leaks': "Assess leak issues. Only count repairs explicitly mentioning leaks, water ingress, or water damage. Ignore blockages or minor plumbing issues. Severe cases involve significant water damage requiring major repairs.",
        'structural': "Assess structural and wall issues. Only count repairs explicitly mentioning structural damage, wall damage, or safety-related building issues. Severe cases involve damage requiring major repairs or posing safety risks."
    }
    
    instruction = category_instructions.get(category_type, f"Assess {category_type} issues based on the repair descriptions.")
    
    assessment = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14",
        text_format=SHDFRepairAssessment,
        input=[
            {
                "role": "system",
                "content": f"You are an expert in building maintenance. {instruction} "
                "Assess the severity (1-5 scale, 0 if none) and frequency (count of occurrences, 0 if none) "
                "based solely on the provided repair history. You are analyzing repairs for a specific time period - "
                "focus only on the repairs listed and do not make assumptions about other time periods."
                "ONLY MARK A REPAIR AS BELONGING TO A CERTAIN CATEGORY if it is EXPLICITLY mentioned as such in the repair history, "
                "for expample, 'leak in the roof' would be counted as a leak, but 'blocked sink' would not be counted as a leak, and "
                "a damp or mould issue is NOT inferred from a leak unless in the SAME repair description for that day it is EXPLICITLY mentions a damp and mould issue"
            },
            {
                "role": "user",
                "content": f"Please analyze these {category_type} repairs and assess severity and frequency:\n\nREPAIRS: [{repair_history}]"
            }
        ],
        temperature=0
    )
    return assessment.output_parsed

def get_overall_repair_summary(all_category_data: Dict, client: OpenAI) -> OverallRepairSummary:
    """Creates a comprehensive summary of all repair categories and their assessments."""
    if not client:
        raise ConnectionError("OpenAI client not initialized.")
    
    # Format all category data for the summary
    summary_data = "COMPREHENSIVE REPAIR ANALYSIS\n\n"
    
    for category, data in all_category_data.items():
        cutoff_info = f"Intervention completed: {data['cutoff_date']}" if data['cutoff_date'] else "No intervention completed"
        
        summary_data += f"{category.upper()}:\n"
        summary_data += f"- {cutoff_info}\n"
        summary_data += f"- Before: Repairs [{data['before_history']}] | Severity {data['before_assessment'].severity}, Frequency {data['before_assessment'].frequency}\n"
        summary_data += f"- After: Repairs [{data['after_history']}] | Severity {data['after_assessment'].severity}, Frequency {data['after_assessment'].frequency}\n\n"
    
    summary = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14",
        text_format=OverallRepairSummary,
        input=[
            {
                "role": "system",
                "content": "You are a housing expert analyzing comprehensive repair data across multiple categories. "
                "Create a detailed summary that compares repair patterns before and after interventions across all categories. "
                "Include specific details about repair types, causes when available, and assess the overall effectiveness "
                "of the retrofit interventions. This summary will be used by housing experts to evaluate intervention impacts."
            },
            {
                "role": "user",
                "content": f"Create a comprehensive summary based on this multi-category repair analysis:\n\n{summary_data}"
            }
        ],
        temperature=0
    )
    return summary.output_parsed


def process_shdf_assessments(properties_df: pd.DataFrame, repairs_df: pd.DataFrame, client) -> pd.DataFrame:
    """
    Main function to process SHDF data with separate category assessments.
    """
    if not client:
        print("\nSkipping LLM assessment because OpenAI client could not be initialized.")
        return properties_df
    
    # Filter for properties with 'Completed' status
    if 'Property Status ' in properties_df.columns:
        properties_to_process = properties_df[properties_df['Property Status '] == 'Completed'].copy()
        print(f"\nFound {len(properties_to_process)} properties with status 'Completed'. Processing these...")
    else:
        print("\n'Property Status ' column not found. Processing all properties.")
        properties_to_process = properties_df.copy()
    
    results = []

    for _, row in tqdm(properties_to_process.iterrows(), desc="Assessing Properties"):
        prop_id = row['uprn']
        print(f"\nAnalyzing property {prop_id}...")
        
        property_result = {'uprn': prop_id}
        repair_histories = {}
        all_category_data = {}
        
        try:
            # Process each category separately
            for issue_type, work_cols in SHDF_COMPLETION_DICT.items():
                print(f"  - Processing {issue_type}...")
                
                # Get separate repair histories for this category
                before_history, after_history, cutoff_date = create_category_repair_histories(
                    row, repairs_df, issue_type, work_cols
                )
                
                # Store repair histories
                repair_histories[f'{issue_type}_before_history'] = before_history
                repair_histories[f'{issue_type}_after_history'] = after_history
                repair_histories[f'{issue_type}_cutoff_date'] = cutoff_date
                
                # Get separate assessments for before and after periods
                print(f"    - Assessing before period for {issue_type}...")
                before_assessment = get_category_period_assessment(before_history, issue_type, client)
                
                print(f"    - Assessing after period for {issue_type}...")
                after_assessment = get_category_period_assessment(after_history, issue_type, client)
                
                # Store all data for this category for the overall summary
                all_category_data[issue_type] = {
                    'before_history': before_history,
                    'after_history': after_history,
                    'cutoff_date': cutoff_date,
                    'before_assessment': before_assessment,
                    'after_assessment': after_assessment
                }
                
                # Store results with consistent naming
                category_prefix = issue_type.replace(' & ', '_').replace(' ', '_')
                property_result[f'{category_prefix}_before_severity'] = before_assessment.severity
                property_result[f'{category_prefix}_before_frequency'] = before_assessment.frequency
                property_result[f'{category_prefix}_after_severity'] = after_assessment.severity
                property_result[f'{category_prefix}_after_frequency'] = after_assessment.frequency
            
            # Create overall summary in a separate LLM pass
            print("  - Creating overall repair summary...")
            overall_summary = get_overall_repair_summary(all_category_data, client)
            
            # Store final results
            property_result['repair_history'] = str(repair_histories)
            property_result['summary'] = overall_summary.summary
            
            results.append(property_result)
            print(f"  Successfully assessed property {prop_id}.")
            
        except Exception as e:
            print(f"  Could not process property {prop_id}: {e}")
    
    if not results:
        return properties_df
    
    # Merge results back into original DataFrame
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

def extract_key_takeaways(repair_summary: str, client) -> RepairTypeKeyTakeaways:
    """
    Extracts key takeaways for each repair type from a comprehensive repair summary.
    
    Args:
        repair_summary: The detailed comparative repair summary
        client: The LLM client instance
    
    Returns:
        RepairTypeKeyTakeaways model with concise takeaways for each repair category
    """
    if not client:
        raise ConnectionError("LLM client not initialized.")
    
    system_prompt = """You are a housing expert analyzing retrofit effectiveness. Your task is to extract key takeaways for each repair category from a comprehensive repair summary.

Context: This summary analyzes the impact of retrofitting works (like insulation, heating system upgrades, window replacements, etc.) on different types of property maintenance issues. The retrofit aims to improve energy efficiency and reduce maintenance problems.

For each repair category (damp/mould, windows/doors, leaks, structural), provide a concise takeaway that:
1. Summarizes whether the retrofit was effective for that repair type
2. Notes any significant improvements or ongoing issues
3. Indicates the overall trend (improved, worsened, no change)

Keep each takeaway to 2-3 sentences maximum. Focus on the practical impact for housing managers and residents."""

    takeaways = client.responses.parse(
        model="gpt-4.1-nano-2025-04-14",  
        text_format=RepairTypeKeyTakeaways,
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": f"Please extract key takeaways for each repair type from this comprehensive repair analysis:\n\n{repair_summary}"
            }
        ],
        temperature=0
    )
    return takeaways.output_parsed

def process_repair_summary_takeaways(properties_with_summaries_df: pd.DataFrame, client) -> pd.DataFrame:
    """
    Processes existing repair summaries to extract key takeaways for each repair type.
    
    Args:
        properties_with_summaries_df: DataFrame containing properties with 'summary' column
        client: The LLM client instance
        
    Returns:
        DataFrame with additional takeaway columns for each repair type
    """
    if not client:
        print("\nSkipping takeaway extraction because LLM client could not be initialized.")
        return properties_with_summaries_df
    
    if 'summary' not in properties_with_summaries_df.columns:
        print("\nNo 'summary' column found in the DataFrame. Cannot extract takeaways.")
        return properties_with_summaries_df
    
    results = []
    
    for _, row in tqdm(properties_with_summaries_df.iterrows(), 
                      total=len(properties_with_summaries_df), 
                      desc="Extracting Key Takeaways",
                      ):
        
        property_id = row.get('uprn', row.get('uprn_', row.get('50_property_ref', 'Unknown')))
        summary = row['summary']
        
        try:
            print(f"Extracting takeaways for property {property_id}...")
            
            # Extract key takeaways using LLM
            takeaways = extract_key_takeaways(summary, client)
            
            # Create result record
            result = {
                'property_id': property_id,
                'damp_mould_takeaway': takeaways.damp_mould_takeaway,
                'windows_doors_takeaway': takeaways.windows_doors_takeaway,  
                'leaks_takeaway': takeaways.leaks_takeaway,
                'structural_takeaway': takeaways.structural_takeaway
            }
            
            results.append(result)
            print(f"  Successfully extracted takeaways for property {property_id}.")
            
        except Exception as e:
            print(f"  Could not extract takeaways for property {property_id}: {e}")
            # Add empty takeaways for failed extractions
            result = {
                'property_id': property_id,
                'damp_mould_takeaway': "Takeaway extraction failed",
                'windows_doors_takeaway': "Takeaway extraction failed",
                'leaks_takeaway': "Takeaway extraction failed", 
                'structural_takeaway': "Takeaway extraction failed"
            }
            results.append(result)
    
    if not results:
        print("No takeaways were successfully extracted.")
        return properties_with_summaries_df
    
    # Convert results to DataFrame
    takeaways_df = pd.DataFrame(results)
    
    # Determine the correct column name for merging
    merge_column = None
    for col in ['uprn', 'uprn_', '50_property_ref']:
        if col in properties_with_summaries_df.columns:
            merge_column = col
            break
    
    if merge_column is None:
        print("Could not find a suitable column for merging takeaways back to the original data.")
        return properties_with_summaries_df
    
    # Merge takeaways back to original DataFrame
    final_df = pd.merge(properties_with_summaries_df, takeaways_df, 
                       left_on=merge_column, right_on='property_id', how='left')
    
    # Drop the temporary property_id column
    final_df = final_df.drop('property_id', axis=1)
    
    print(f"\nSuccessfully added takeaway columns to {len(final_df)} properties.")
    return final_df

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
    sharing_cities_df = pd.read_excel("data\\sharing_cities_property_summaries.xlsx")
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

def run_key_takeaways_analysis():
    df = pd.read_excel(OUTPUT_FILE_PATH)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client:
        print("\nSkipping key takeaways extraction because OpenAI client could not be initialized.")
        return
    df_with_takeaways = process_repair_summary_takeaways(df, client)
    df_with_takeaways.to_excel(OUTPUT_FILE_PATH, index=False)

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
    #generate_shdf_summaries()
    #run_wates_analysis()
    run_key_takeaways_analysis()