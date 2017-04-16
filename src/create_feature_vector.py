from sets import Set
import nltk

# put words into set to eliminate duplicates

feature_set = set()

with open('../res/selected_words.txt') as fp:
    for line in fp:
        feature_set.add(line.split(',', 1)[0])

'''
f = open("../res/cleaned_selected_words.txt", "w")

for feature in feature_set:
    f.write(feature + '\n')

f.close()'''

# transform from set into dictionary

count = 0
feature_table = {}

for feature in feature_set:
    feature_table[feature] = count
    count += 1

# run through training text and create feature vector 

with open('../res/cleaned_train.txt') as f_in:
    f_out = open("../res/x_training.txt", "w")

    for line in f_in:
        vector = [0] * count

        tokens = nltk.word_tokenize(line)  
        for token in tokens:
            index = feature_table.get(token)
            if index != None:
                vector[index] = 1

        f_out.write(", ".join(map(str, vector)) + '\n')

    f_out.close()