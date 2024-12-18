"""
file name: helper.py
purpose: Small functions that are,
    self explanatory.
"""
import pandas as pd


def print_dict(my_dict, n_lines):
    count = 0
    for key, value in my_dict.items():
        print(key, value)
        count += 1
        if count == n_lines:
            break
    print("Number of keys:", len(my_dict))


def save_dataframe_to_csv(file_path, dataframe):
    try:
        dataframe.to_csv(file_path, index=False)  # Save without including the index column
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")


def load_data(path): 
    return pd.read_csv(path)


def count_value_occurrences(my_dict, value_name):
    value_count = 0
    total_values = 0
    for value in my_dict.values():
        if isinstance(value, list):
            value_count += value.count(value_name)
            total_values += len(value)
        else:
            if value == value_name:
                value_count += 1
            total_values += 1
    print(f"The value '{value_name}' appears {value_count} time(s).")
    print(f"Total keys: {len(my_dict)}, Total values: {total_values}.")


def append_dict_to_dataframe(df, data_dict, new_column_name='Processed_entities'):
    df[new_column_name] = 0
    for key, value in data_dict.items():
        if key in df.index:
            df.at[key, new_column_name] = value
        else:
            df.loc[key] = [0] * len(df.columns)
            df.at[key, new_column_name] = value
    return df

