import csv
import sys

RES_PREFIX = '../res/'

def clean_data(input_file, output_file):
    with open(RES_PREFIX + input_file, 'rb') as input_csv:
        with open(RES_PREFIX + output_file, 'wb') as output_csv:
            csv_reader = csv.reader(input_csv, delimiter=',', quotechar='|')
            csv_writer = csv.writer(output_csv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                text = row[1].strip().lower()
                filtered_text = ''

                # Remove `mention` and `link` info
                mention_index = text.find('@mention')
                if mention_index > -1:
                    text = text[:mention_index] + text[mention_index + 8:]
                link_index = text.find('{link}')
                if link_index > -1:
                    text = text[:link_index] + text[link_index + 6:]
                
                for i in range(len(text)):
                    if text[i] in ['(', ')']:
                        continue
                    if text[i] == ' ' and len(filtered_text) > 0 and filtered_text[-1] == ' ':
                        continue
                    if text[i] in [',', '.', ':', ';'] and i < len(text) - 1 and text[i + 1] == ' ':
                        continue
                    filtered_text += text[i]
                csv_writer.writerow([row[0], filtered_text] + row[4:9] + row[13:])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Running Format: python cleansing.py [inputfile] [outputfile]'
        sys.exit()
    input_file, output_file = sys.argv[1], sys.argv[2]
    clean_data(input_file, output_file)
