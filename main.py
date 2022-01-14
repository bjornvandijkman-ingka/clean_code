import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
from datetime import *
from tqdm import tqdm
import re


supply_words = ["pan", "rasp", "kom"]

filepath = "data/lunch_recipes.csv"


def load_data(filepath):
    """
    Fetch and load the recipe data
    Function takes path of data file as input
    Loads it as pd dataframe
    """
    df = pd.read_csv(filepath)  # Read lunch recipes dataframe.
    # drop unnecessary columns
    df.drop(["url", "dish"], axis=1, inplace=True)
    return df


def clean_text(string):
    """Remove non-alphanumeric characters
    and apply lowercase

    Args:
        string (str): Recipe row text

    Returns:
        str: Cleaned recipe row text
    """
    new_string = re.sub(r"[^\w\s]", "", string).lower()
    return new_string


def find_dishes(string):
    """Map existence of extra dishes within the recipe
    Return boolean if there is a match

    Args:
        string (str): The cleaned recipe text

    Returns:
        bool: True for hit, false for absence
    """
    clean_string = clean_text(string)
    return any(word in clean_string for word in supply_words)


def check_if_extra_dishes():
    """Check if extra dishes are mentioned within the recipe

    Returns:
        pd.DataFrame: Dataframe with bool for extra recipes
    """
    df = load_data(filepath)
    df["extra_dishes"] = df["recipe"].apply(lambda x: find_dishes(x))
    df = df.drop("servings", axis=1)
    df = df.drop("recipe", axis=1)
    df["date"] = df.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

    return df


def read_attendance_sheet():
    df = pd.read_csv("data/key_tag_logs.csv")

    df["date"] = df["timestamp"].apply(lambda x: datetime.strptime(x[:10], "%Y-%m-%d"))

    df["timestamp"] = df["timestamp"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    )

    df["time"] = df["timestamp"].apply(lambda x: x.time())
    df = df.drop("timestamp", axis=1)

    result = pd.DataFrame(np.array(df["date"]), columns=["date"]).drop_duplicates()

    for name in df["name"].unique():
        lunchdates = []
        for datum in df["date"].unique():
            df2 = df[df["name"] == name]
            df2 = df2[df2["date"] == datum]

            dataframe_check_in = df2[df2["event"] == "check in"]
            dataframe_check_in = dataframe_check_in[
                dataframe_check_in.time < time(12, 0, 0)
            ]

            df_check_out = df2[df2["event"] == "check out"]
            df_check_out = df_check_out[df_check_out["time"] > time(12, 0, 0)]
            if df_check_out.shape[0] > 0 and dataframe_check_in.shape[0] > 0:
                lunchdates.append(datum)

        result[f"{name}"] = result.date.apply(
            lambda x: 1 if x in list(lunchdates) else 0
        )

    return result


def train_model(alpha=0.1):
    recipes = check_if_extra_dishes()
    attendance = read_attendance_sheet()
    df_logs = pd.read_csv("data/dishwasher_log.csv")
    df_logs["date"] = df_logs.date.apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

    df = recipes.merge(attendance, on="date", how="outer").merge(df_logs).fillna(0)
    reg = LinearRegression(fit_intercept=False, positive=True).fit(
        df.drop(["dishwashers", "date"], axis=1), df["dishwashers"]
    )
    print(reg.coef_)
    return dict(zip(reg.feature_names_in_, [round(c, 3) for c in reg.coef_]))


if __name__ == "__main__":

    print(train_model())
