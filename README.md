# Calculating-Probability-based-on-Bayesian-Network
This program performs various tasks related to calculating and utilizing probabilities from a set of training data, specifically for a Bayesian network involving four variables: B, G, C, and F. It can handle different tasks based on command-line arguments.

Variables and Their Values
- Variables: B, G, C, F
- Values: Each variable can be either True or False.
- Task Flags
task1: Set to True if there is only one command-line argument.
task2: Set to True if there are more than two command-line arguments.
task3: Set to True if the fourth command-line argument is "given" or if the total number of arguments is four.

Steps of the Program

Initialization:

- Initializes the possible values for each variable.
- Sets up a dictionary counts to keep track of the counts for each combination of the variables.
  
Reading Training Data:

- Reads a file (specified by the first command-line argument) containing training data.
- Each line in the file is converted to a list of boolean values (representing the values of B, G, C, F).
- The count for each combination of values is incremented in the counts dictionary.

Calculating Conditional Probability Tables (Task 1):

Calculates the conditional probabilities 
1. 𝑃(𝐹∣𝐵,𝐺,𝐶)
2. P(F∣B,G,C)

These probabilities are stored in conditional_probs.

If task1 is True, it prints the calculated probabilities.

Calculating Joint Probability Distribution:

Calculates the joint probability distribution for all combinations of the variables using the conditional probabilities and the counts.

The joint probabilities are stored in joint_probs.

Task 2: Calculating Requested Probability:

If task2 is True, retrieves the user input values for B, G, C, F from the command-line arguments.

Converts these values to boolean.

Calculates and prints the joint probability for the given combination of values using joint_probs.

Task 3: Training Bayesian Network:

If task3 is True, it reads and processes the training data into a structured format.

Trains the Bayesian network by calculating the counts and conditional probability tables for each variable given the others.

Calculates and prints the probability ratio of B and C values from the trained tables.

Task 3 Functions:

load_training_data(training_data_file):

Loads the training data from a file and structures it into a list of dictionaries with keys 'B', 'G', 'C', and 'F'

- train_bayesian_network(training_data):
Counts the occurrences of each variable and their combinations.
Calculates the conditional probability tables for each variable given the others.
Returns the probability tables for B, G, C, and F.

#Example Usage

- python script.py data.txt
  
Processes the training data and performs task 1.

- python script.py data.txt bt gt ct ft
  
Processes the training data and performs task 2, calculating the joint probability for B=True, G=True, C=True, F=True.

- python script.py data.txt given
  
Processes the training data and performs task 3, training the Bayesian network and calculating a probability ratio.
