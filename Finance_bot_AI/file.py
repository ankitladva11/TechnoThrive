import pandas as pd
import json

# Load the CSV file into a DataFrame
df = pd.read_csv("Finance_Q&A.csv")
# Define a list to store the prompt-completion pairs
pairs = []

# Iterate over the rows of the DataFrame and create JSONL entries
for index, row in df.iterrows():
    prompt = row["instruction"]
    completion = row["output"]
   
    # Create a dictionary for the JSON object
    pair = {
        "prompt": prompt,
        "completion": completion
    }
   
    # Append the JSON object to the list
    pairs.append(pair)

# Write the JSONL data to a file
with open("outputf.jsonl", "w") as jsonl_file:
    for pair in pairs:
        jsonl_file.write(json.dumps(pair) + "\n")

print("Conversion to JSONL completed.")