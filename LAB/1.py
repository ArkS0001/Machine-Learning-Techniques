import pandas as pd

def find_s_algorithm(data):
    # Initialize hypothesis to the first positive example
    hypothesis = None
    for example in data:
        if example[-1] == 'Yes':
            hypothesis = example[:-1]
            break
            
    # Find the most specific hypothesis
    for example in data:
        if example[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if example[i] != hypothesis[i]:
                    hypothesis[i] = '?'
    return hypothesis

# Load data from CSV file
df = pd.read_csv('training_data.csv')
data = df.values.tolist()

# Apply FIND-S algorithm
hypothesis = find_s_algorithm(data)
print("Most specific hypothesis:", hypothesis)
