RES_PREFIX = '../res/'
RAW_FILE = 'train.txt'
OUT_FILE_TEST = 'sentiment_labels_test.txt'
OUT_FILE_TRAIN = 'sentiment_labels_train.txt'
TEST_PERCENT = 0.2

input_txt = open(RES_PREFIX + RAW_FILE, 'r')
output_test_txt = open(RES_PREFIX + OUT_FILE_TEST, 'w')
output_train_txt = open(RES_PREFIX + OUT_FILE_TRAIN, 'w')
rows = input_txt.readlines()
result = []

for row in rows:
    columns = row.split('","')
    s1 = columns[4] # Can't tell
    s2 = columns[5] # Negative
    s3 = columns[6] # Neutral / author is just sharing information
    s4 = columns[7] # Positive
    s5 = columns[8] # Tweet not related to weather condition
    max_label = max(s1, s2, s3, s4, s5)
    if max_label == s1:
        result.append('0')
    if max_label == s2:
        result.append('1')
    if max_label == s3:
        result.append('2')
    if max_label == s4:
        result.append('3')
    if max_label == s5:
        result.append('4')

test_row_num = int(len(rows) * TEST_PERCENT)

for i in range(test_row_num):
    output_test_txt.write(result[i] + '\n')

for i in range(test_row_num, len(rows)):
    output_train_txt.write(result[i] + '\n')

input_txt.close()
output_test_txt.close()
output_train_txt.close()
