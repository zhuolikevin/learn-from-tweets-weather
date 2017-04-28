'''
Split train.csv into output_1.csv and output_2.csv
output_1.csv contains 80 percent of the data for training
output_2.csv contains 20 percent of the data for testing
'''
import os

def split(filehandler, row_limit, delimiter=',', output_name_template='output_%s.csv', output_path='../res', keep_headers=True):
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = reader.next()
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)

input_len = 77947
train_percentage = 0.8
train_len = (int) (input_len * train_percentage)  #62357

split(open('../res/train.csv', 'r'), train_len);
