import sys

RES_PREFIX = '../res/'
TEST_PERCENT = 0.2

def split_data(input_file, output_train, output_test):
    input_txt = open(RES_PREFIX + input_file, 'r')
    output_train_txt = open(RES_PREFIX + output_train, 'w')
    output_test_txt = open(RES_PREFIX + output_test, 'w')
    rows = input_txt.readlines()
    test_row_num = int(len(rows) * TEST_PERCENT)

    for i in range(test_row_num):
        output_test_txt.write(rows[i])

    for i in range(test_row_num, len(rows)):
        output_train_txt.write(rows[i])

    input_txt.close()
    output_train_txt.close()
    output_test_txt.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Running Format: python split_train.py [train_file] [test_file]'
        sys.exit()
    input_file, output_train, output_test = sys.argv[1], sys.argv[2], sys.argv[3]
    split_data(input_file, output_train, output_test)
