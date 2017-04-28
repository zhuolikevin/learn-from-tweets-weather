# from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

# RES_PREFIX = '../res/'
# DATA_TRAINING = 'x_training.txt'
# DATA_TESTING = 'x_testing.txt'
#
# SENTIMENT_LABEL_TRAINING = 'sentiment_labels_train.txt'
# SENTIMENT_LABEL_TESTING = 'sentiment_labels_test.txt'
#
# train_data = open(RES_PREFIX + DATA_TRAINING, 'r')
# train_label = open(RES_PREFIX + SENTIMENT_LABEL_TRAINING, 'r')
#
# test_data = open(RES_PREFIX + DATA_TESTING, 'r')
# test_label = open(RES_PREFIX + SENTIMENT_LABEL_TESTING, 'r')
#
# train_data_rows = train_data.readlines()
# train_label_rows = train_label.readlines()
#
# X = []
# Y = []
# for i in range(len(train_data_rows)):
#     features = train_data_rows[i].split(', ')
#     features = map(int, features)
#     X.append(features)
#
#     label = int(train_label_rows[i])
#     Y.append(label)
#
# X = np.array(X)
# Y = np.array(Y)
#
# np.save('../data/train_x_pn', X)
# np.save('../data/train_y_pn', Y)

X = np.load('../data/train_x_pn.npy', mmap_mode='r')
Y = np.load('../data/train_y_pn.npy', mmap_mode='r')

# print 'Finished load train data'
#
# test_data_rows = test_data.readlines()
# test_label_rows = test_label.readlines()
#
# TEST_X = []
# TEST_Y = []
# for i in range(len(test_data_rows)):
#     features = test_data_rows[i].split(', ')
#     features = map(int, features)
#     TEST_X.append(features)
#
#     label = int(test_label_rows[i])
#     TEST_Y.append(label)
#
# TEST_X = np.array(TEST_X)
# TEST_Y = np.array(TEST_Y)
#
# np.save('../data/test_x_pn', TEST_X)
# np.save('../data/test_y_pn', TEST_Y)

TEST_X = np.load('../data/test_x_pn.npy', mmap_mode='r')
TEST_Y = np.load('../data/test_y_pn.npy', mmap_mode='r')

print 'Finished load test data'

# clf = BernoulliNB()
clf = RandomForestClassifier(n_estimators=500, max_depth=50)
# clf = Ridge()
clf.fit(X, Y)

print 'Train finished!'
print 'Start predicting'

# Predict training set
predictions = clf.predict(X)

correct_count = incorrect_count = 0
for i in range(np.shape(X)[0]):
    if Y[i] == predictions[i]:
        correct_count += 1
    else:
        incorrect_count += 1

print '----- Predic Training Set -----'
print 'Correct: ' + str(correct_count)
print 'Incorrect: ' + str(incorrect_count)
print 'Accurracy: ' + str(correct_count / ((correct_count + incorrect_count) * 1.0))

# Predict testing set
predictions = clf.predict(TEST_X)

correct_count = incorrect_count = 0
for i in range(np.shape(TEST_X)[0]):
    if TEST_Y[i] == predictions[i]:
        correct_count += 1
    else:
        incorrect_count += 1

print '----- Predic Testing Set -----'
print 'Correct: ' + str(correct_count)
print 'Incorrect: ' + str(incorrect_count)
print 'Accurracy: ' + str(correct_count / ((correct_count + incorrect_count) * 1.0))

print mean_squared_error(predictions, TEST_Y)
