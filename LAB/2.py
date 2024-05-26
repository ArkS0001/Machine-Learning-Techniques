def candidate_elimination_algorithm(data):
    num_attributes = len(data[0]) - 1
    G = [['?' for _ in range(num_attributes)]]
    S = ['0' for _ in range(num_attributes)]

    for example in data:
        if example[-1] == 'Yes':  # Positive example
            for i in range(num_attributes):
                if S[i] == '0':
                    S[i] = example[i]
                elif S[i] != example[i]:
                    S[i] = '?'
            G = [g for g in G if all((g[i] == '?' or g[i] == S[i]) for i in range(num_attributes))]
        else:  # Negative example
            G = [g for g in G if not all((g[i] == '?' or g[i] == example[i]) for i in range(num_attributes))]
            G = [g[:i] + [example[i]] + g[i + 1:] for g in G for i in range(num_attributes) if g[i] == '?']

    return S, G

# Apply Candidate-Elimination algorithm
S, G = candidate_elimination_algorithm(data)
print("S:", S)
print("G:", G)
