import csv

# Define the data
data = [
    ["Sentence", "Label"],
    ["She runs like the wind.", "Simile"],
    ["The world is a stage.", "Metaphor"],
    ["The flowers danced in the breeze.", "Personification"],
    ["He went to the store to buy some bread.", "None"],
    ["Her smile was like sunshine.", "Simile"],
    ["The stars danced in the night sky.", "Personification"],
    ["This assignment is a nightmare.", "Metaphor"],
]

# Specify the file path
file_path = "dataset.csv"

# Write to a CSV file
with open(file_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Dataset saved to {file_path}")
