import pandas as pd
from augment import augment_df


def get_json_data():
    df = pd.read_json("data/data.txt")
    df = df.drop("found_duplicate", axis=1)
    return df


def main():
    df = get_json_data()
    
    print("Original dataframe:", df.head())
    
    augmented_df = augment_df(df.head())
    
    print("Augmented dataframe:", augmented_df.head(10))


if __name__ == "__main__":
    main()
