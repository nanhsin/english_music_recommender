import pandas as pd

def process_data(file_path):
    df = pd.read_json(file_path)

    df['danceability_level'] = categorize_level(df['danceability'])
    df['valence_level'] = categorize_level(df['valence'])
    df['speechiness_level'] = categorize_numeric_level(df['speechiness'])
    df['fres_level'] = categorize_numeric_level(df['fres'])
    df['vocabComplex_level'] = categorize_numeric_level(df['vocabComplex'])
    df['avgSyllable_level'] = categorize_numeric_level(df['avgSyllable'])

    df['difficulty'] = calculate_difficulty(df)

    df['difficulty_level'] = categorize_difficulty(df['difficulty'])

    return df

def categorize_level(column):
    percentiles = column.quantile([0, 0.33, 0.66, 1])
    bins = [percentiles.iloc[0], percentiles.iloc[1], percentiles.iloc[2], percentiles.iloc[3]]
    labels = ['Low', 'Medium', 'High']
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

def categorize_numeric_level(column):
    percentiles = column.quantile([0, 0.33, 0.66, 1])
    bins = [percentiles.iloc[0], percentiles.iloc[1], percentiles.iloc[2], percentiles.iloc[3]]
    labels = [1, 2, 3]
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True).astype(int)

def calculate_difficulty(df):
    return df['speechiness_level'] + df['fres_level'] + df['vocabComplex_level'] + df['avgSyllable_level']

def categorize_difficulty(column):
    percentiles = column.quantile([0, 0.33, 0.66, 1])
    bins = [percentiles.iloc[0], percentiles.iloc[1], percentiles.iloc[2], percentiles.iloc[3]]
    labels = ["Low", "Medium", "High"]
    return pd.cut(column, bins=bins, labels=labels, include_lowest=True)

def recommendation(df, dance_choice, valence_choice, difficulty_choice):
    if dance_choice == "Low":
        df = df[df['danceability_level'] == "Low"]
    elif dance_choice == "Medium":
        df = df[df['danceability_level'] == "Medium"]
    elif dance_choice == "High":
        df = df[df['danceability_level'] == "High"]

    if valence_choice == "Negative":
        df = df[df['valence_level'] == "Low"]
    elif valence_choice == "Neutral":
        df = df[df['valence_level'] == "Medium"]
    elif valence_choice == "Positive":
        df = df[df['valence_level'] == "High"]

    if difficulty_choice == "Easy":
        df = df[df['difficulty_level'] == "Low"]
    elif difficulty_choice == "Medium":
        df = df[df['difficulty_level'] == "Medium"]
    elif difficulty_choice == "Hard":
        df = df[df['difficulty_level'] == "High"]

    chosen = df.sample() # random choose 1 song
    return chosen