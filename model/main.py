import pandas as pd
from generate import QuestionAnsweringModel
import pickle


def get_json_data():
    df = pd.read_json("data/data.txt")
    df = df.drop("found_duplicate", axis=1)
    return df


def main():
    df = pd.read_csv("data/final.csv")
    model = QuestionAnsweringModel(df)
    
    # export model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    
    

        
if __name__ == "__main__":
    main()
