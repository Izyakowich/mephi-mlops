import great_expectations as gx
import pandas as pd
import sys

def validate_data():
    print("Running Great Expectations Validation...")
    df = pd.read_csv("data/raw/UCI_Credit_Card.csv")
    df = df.rename(columns={"default.payment.next.month": "target"})
    
    context = gx.get_context()
    
    # В новых версиях GX API немного отличается, создадим упрощенную валидацию
    suite = context.add_or_update_expectation_suite("credit_data_suite")
    validator = context.sources.pandas_default.read_dataframe(df)
    validator.expectation_suite_name = "credit_data_suite"
    
    validator.expect_column_to_exist("LIMIT_BAL")
    validator.expect_column_values_to_not_be_null("LIMIT_BAL")
    validator.expect_column_values_to_be_between("AGE", min_value=18, max_value=100)
    validator.expect_column_values_to_be_in_set("target", [0, 1])
    
    results = validator.validate()
    if not results.success:
        raise ValueError("Data Validation Failed! Data anomalies detected.")
    print("Data Validation Passed successfully!")

if __name__ == "__main__":
    validate_data()