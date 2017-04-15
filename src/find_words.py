import sys
import nltk
from collections import defaultdict
import operator

RES_PREFIX = '../res/'

def find_words(pos, input_file, output_file=None):
    input_txt = open(RES_PREFIX + input_file, 'r')
    rows = input_txt.readlines()
    word_dict = defaultdict(int)
    for row in rows:
        tokens = nltk.word_tokenize(row)
        tagged = nltk.pos_tag(tokens)
        for tag in tagged:
            if tag[1] == pos:
                word_dict[tag[0]] += 1

    sorted_word_list = sorted(word_dict.items(), key=operator.itemgetter(1))
    sorted_word_list.reverse()

    if output_file:
        output_txt = open(RES_PREFIX + output_file, 'w')
        for word in sorted_word_list:
            output_txt.write(word[0] + ', ' + str(word[1]) + '\n')
    else:
        for i in range(100):
            word = sorted_word_list[i]
            print word[0] + ',' + word[1]

if __name__ == '__main__':
    if len(sys.argv) not in [3, 4]:
        print 'Running Format: python find_words.py [NN|JJ] [inputfile] [outputfile]'
        sys.exit()
    pos = sys.argv[1]
    input_file = sys.argv[2]
    if len(sys.argv) == 3:
        find_words(pos, input_file)
    if len(sys.argv) == 4:
        output_file = sys.argv[3]
        find_words(pos, input_file, output_file)
