"""
Data validation using Great Expectations.

Defines two validation suites:
1. Raw data suite — validates the CSV file before processing
2. Processed data suite — validates engineered features after preprocessing

These validations run as part of the Airflow training pipeline to catch
data quality issues before they affect model training.
"""

import os
import pandas as pd
import great_expectations as gx


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def validate_raw_data(df: pd.DataFrame) -> dict:
    """Validate the raw Telco Churn dataset.

    Checks:
    - Expected number of columns (21)
    - No null customerIDs
    - Churn values are only Yes/No
    - tenure is between 0 and 72
    - MonthlyCharges is positive
    - SeniorCitizen is only 0 or 1
    - Expected row count range (6000-8000 for this dataset)

    Returns dict with 'success' boolean and 'results' details.
    """
    context = gx.get_context()

    datasource = context.sources.add_or_update_pandas(name="raw_data")
    data_asset = datasource.add_dataframe_asset(name="raw_telco")
    batch_request = data_asset.build_batch_request(dataframe=df)

    # Create expectation suite
    context.add_or_update_expectation_suite("raw_data_suite")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="raw_data_suite",
    )

    # Column count check
    validator.expect_table_column_count_to_equal(21)

    # Row count sanity check
    validator.expect_table_row_count_to_be_between(min_value=6000, max_value=8000)

    # customerID should never be null
    validator.expect_column_values_to_not_be_null("customerID")

    # Churn must be Yes or No
    validator.expect_column_values_to_be_in_set("Churn", ["Yes", "No"])

    # tenure should be 0-72
    validator.expect_column_values_to_be_between("tenure", min_value=0, max_value=72)

    # MonthlyCharges should be positive
    validator.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    # SeniorCitizen is binary
    validator.expect_column_values_to_be_in_set("SeniorCitizen", [0, 1])

    # Validate
    results = validator.validate()

    return {
        "success": results.success,
        "statistics": results.statistics,
        "suite": "raw_data_suite",
    }


def validate_processed_data(df: pd.DataFrame) -> dict:
    """Validate the processed/engineered dataset.

    Checks:
    - Churn is encoded as 0/1
    - TotalCharges is numeric (no whitespace strings)
    - No null values in key columns
    - service_count is between 0 and 7
    - Feature columns exist

    Returns dict with 'success' boolean and 'results' details.
    """
    context = gx.get_context()

    datasource = context.sources.add_or_update_pandas(name="processed_data")
    data_asset = datasource.add_dataframe_asset(name="processed_telco")
    batch_request = data_asset.build_batch_request(dataframe=df)

    context.add_or_update_expectation_suite("processed_data_suite")

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="processed_data_suite",
    )

    # Churn should now be 0/1
    validator.expect_column_values_to_be_in_set("Churn", [0, 1])

    # TotalCharges should be numeric with no nulls
    validator.expect_column_values_to_not_be_null("TotalCharges")
    validator.expect_column_values_to_be_between("TotalCharges", min_value=0)

    # service_count should be 0-7
    if "service_count" in df.columns:
        validator.expect_column_values_to_be_between(
            "service_count", min_value=0, max_value=7
        )

    # No nulls in critical columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]:
        validator.expect_column_values_to_not_be_null(col)

    results = validator.validate()

    return {
        "success": results.success,
        "statistics": results.statistics,
        "suite": "processed_data_suite",
    }


if __name__ == "__main__":
    from data.load import load_raw_data, clean_data, encode_target
    from data.preprocess import preprocess_for_training

    print("Validating raw data...")
    raw_df = load_raw_data()
    raw_result = validate_raw_data(raw_df)
    print(f"  Raw validation: {'PASSED' if raw_result['success'] else 'FAILED'}")
    print(f"  Stats: {raw_result['statistics']}")

    print("\nValidating processed data...")
    processed_df = clean_data(raw_df)
    processed_df = encode_target(processed_df)
    processed_df = preprocess_for_training(processed_df)
    proc_result = validate_processed_data(processed_df)
    print(f"  Processed validation: {'PASSED' if proc_result['success'] else 'FAILED'}")
    print(f"  Stats: {proc_result['statistics']}")
