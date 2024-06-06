import sys
from collections import defaultdict

# Define the variables and their possible values.
variables = ['B', 'G', 'C', 'F']
values = {'B': [True, False], 'G': [True, False], 'C': [True, False], 'F': [True, False]}

# Initialize the counts for each combination of variables.
counts = {(b, g, c, f): 0 for b in values['B'] for g in values['G'] for c in values['C'] for f in values['F']}

task1 = False
task2 = False
task3 = False

if len(sys.argv) == 2: 
    task1 = True

if len(sys.argv) > 2: 
    task2 = True

if len(sys.argv) > 4:
    if sys.argv[4] == 'given' or len(sys.argv) == 4:
        task3 = True
        task2 = False


# Read the training data from the file.
with open(sys.argv[1], 'r') as f:
    for line in f:
        # Convert each line to a list of Boolean values.
        data = [bool(int(x)) for x in line.split()]

        # Increment the count for this combination of variables.
        counts[tuple(data)] += 1

# Calculate the conditional probability tables.
# Task 1
conditional_probs = {}
for b in values['B']:
    for g in values['G']:
        for c in values['C']:
            # P(F | B, G, C) = count(B, G, C, F) / count(B, G, C)
            count_bgc = sum(counts[(b, g, c, f)] for f in values['F'])
            for f in values['F']:
                count_bgcf = counts[(b, g, c, f)]
                conditional_probs[(f, b, g, c)] = count_bgcf / count_bgc
                if task1: print(f'P({variables[3]}={f} | {variables[0]}={b}, {variables[1]}={g}, {variables[2]}={c}) = {count_bgcf} / {count_bgc} = {count_bgcf/count_bgc:.2f}')
# Calculate the joint probability distribution.
joint_probs = {}
for b in values['B']:
    for g in values['G']:
        for c in values['C']:
            for f in values['F']:
                joint_probs[(b, g, c, f)] = conditional_probs[(f, b, g, c)] * counts[(b, g, c, f)] / len(counts)

# Retrieve the user input for Task 2
if task2:
    B = True if sys.argv[2].lower() == 'bt' else False
    G = True if sys.argv[3].lower() == 'gt' else False
    C = True if sys.argv[4].lower() == 'ct' else False
    F = True if sys.argv[5].lower() == 'ft' else False

    # Calculate the requested probability.
    requested_prob = joint_probs[(B, G, C, F)]

    # Display the calculated probability.
    print(f'P(B={B}, G={G}, C={C}, F={F}) = {requested_prob:.2f}')

if task3:
    # This will load the training data from a file
    def load_training_data(training_data_file):
        training_data = []
        with open(training_data_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                training_data.append({
                    'B': bool(int(values[0])),
                    'G': bool(int(values[5])),
                    'C': bool(int(values[10])),
                    'F': bool(int(values[15]))
                })
        return training_data

    # Train the Bayesian Network
    def train_bayesian_network(training_data):
        # Count the number of occurrences of each variable and each combination of variables
        counts = {'B': [0, 0], 'G': [0, 0], 'C': [0, 0], 'F': [0, 0]}
        joint_counts = {(False, False, False, False): [0, 0],
                        (False, False, False, True): [0, 0],
                        (False, False, True, False): [0, 0],
                        (False, False, True, True): [0, 0],
                        (False, True, False, False): [0, 0],
                        (False, True, False, True): [0, 0],
                        (False, True, True, False): [0, 0],
                        (False, True, True, True): [0, 0],
                        (True, False, False, False): [0, 0],
                        (True, False, False, True): [0, 0],
                        (True, False, True, False): [0, 0],
                        (True, False, True, True): [0, 0],
                        (True, True, False, False): [0, 0],
                        (True, True, False, True): [0, 0],
                        (True, True, True, False): [0, 0],
                        (True, True, True, True): [0, 0]}
        for row in training_data:
            counts['B'][int(row['B'])] += 1
            counts['G'][int(row['G'])] += 1
            counts['C'][int(row['C'])] += 1
            counts['F'][int(row['F'])] += 1
            joint_counts[(row['B'], row['G'], row['C'], row['F'])][1] += 1

        # Calculate the conditional probability tables
        B_table = [counts['B'][True] / sum(counts['B']), counts['B'][False] / sum(counts['B'])]
        G_table = [counts['G'][True] / sum(counts['G']), counts['G'][False] / sum(counts['G'])]
        C_table = [counts['C'][True] / sum(counts['C']), counts['C'][False] / sum(counts['C'])]
        F_table = [[0, 0], [0, 0]]
        for j in range(2):
            for i in range(2):
                F_table[j][i] = joint_counts[(True, True, i, j)][1] / counts['G'][True] + \
                                joint_counts[(True, False, i, j)][1] / counts['G'][False]
                F_table[j][i] /= 2

        #return (B_table, G_table, C_table, F_table)

        B_table = {}
        for b in [True, False]:
            B_table[b] = defaultdict(int)
            for g in [True, False]:
                for c in [True, False]:
                    for f in [True, False]:
                        if b:
                            prob = 0.7 if g else 0.2
                        else:
                            prob = 0.3 if g else 0.8
                        prob *= 0.9 if c else 0.1
                        prob *= 0.9 if f else 0.1
                        B_table[b][(g, c, f)] += prob
            total = sum(B_table[b].values())
            for key in B_table[b]:
                B_table[b][key] /= total

        # Calculate the conditional probability distribution for G
        G_table = {}
        for g in [True, False]:
            G_table[g] = defaultdict(int)
            for b in [True, False]:
                for c in [True, False]:
                    for f in [True, False]:
                        if g:
                            prob = 0.9 if b else 0.3
                        else:
                            prob = 0.1 if b else 0.7
                        prob *= 0.9 if c else 0.1
                        prob *= 0.9 if f else 0.1
                        G_table[g][(b, c, f)] += prob
            total = sum(G_table[g].values())
            for key in G_table[g]:
                G_table[g][key] /= total

        # Calculate the conditional probability distribution for C
        C_table = {}
        for c in [True, False]:
            C_table[c] = defaultdict(int)
            for b in [True, False]:
                for g in [True, False]:
                    for f in [True, False]:
                        if c:
                            prob = 0.5
                        else:
                            prob = 1.0
                        prob *= 0.9 if f else 0.1
                        C_table[c][(b, g, f)] += prob
            total = sum(C_table[c].values())
            for key in C_table[c]:
                C_table[c][key] /= total

        # Calculate the conditional probability distribution for F
        F_table = {}
        for f in [True, False]:
            F_table[f] = defaultdict(int)
            for b in [True, False]:
                for g in [True, False]:
                    for c in [True, False]:
                        prob = 1.0 if f else 0.0
                        F_table[f][(b, g, c)] += prob
            total = sum(F_table[f].values())
            for key in F_table[f]:
                if total !=0 :
                    F_table[f][key] /= total

        return (B_table, G_table, C_table, F_table)


    tdata = load_training_data(sys.argv[1])

    B_table, G_table, C_table, F_table = train_bayesian_network(tdata)
    bval = 0
    gval = 0

    for key, value in B_table.items():
        for sub_key, sub_value in value.items():
            bval = sub_value
    
    for key, value in C_table.items():
        for sub_key, sub_value in value.items():
            gval = sub_value

    print(f'Probability: {bval/gval}')