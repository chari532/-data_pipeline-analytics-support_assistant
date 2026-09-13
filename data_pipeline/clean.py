import numpy as np
import pandas as pd

def load_and_clean(input_csv="books_raw.csv"):
    df = pd.read_csv(input_csv)
    df["price"] = df["price"].astype(str).str.strip().str.replace("Â£","", regex = False)
    df["price"]= pd.to_numeric(df["price"], errors = "coerce")
    df.rename(columns = {"price" : "price_gbp"}, inplace=True)

    df["star_rating"] = df["star_rating"].map({"One":1,"Two":2,"Three":3,"Four":4,"Five":5})
    df.rename(columns = {"star_rating" : "rating"}, inplace=True)

    df["in_stock"] = df["availability"].astype(str).str.contains("in stock", case=False, na=False)
   

    df["price_inr"] = (df["price_gbp"] * 105.50).round(2)

    final_columns = ["title","price_gbp","price_inr","rating","category","in_stock"]
    df_clean = df[final_columns].reset_index(drop=True) 
    print(f"Final cleaned dataset: {len(df_clean)} rows")
    return df_clean

if __name__ == "__main__":
    df_clean = load_and_clean("books_raw.csv")
    df_clean.to_csv("books_clean.csv", index=False)
    print("Saved cleaned data to books_clean.csv")
    print(df_clean.head())