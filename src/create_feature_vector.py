from sets import Set

feature_set = set()

with open('../res/selected_words.txt') as fp:
    for line in fp:
        feature_set.add(line.split(',', 1)[0])

print feature_set

f = open("../res/cleaned_selected_words.txt", "w")

for feature in feature_set:
    f.write(feature + '\n')

f.close()