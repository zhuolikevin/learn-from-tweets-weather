import sys
import re
import string
import math
import numpy as np
import pandas as pn
from nltk.stem import LancasterStemmer
from nltk import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as prep

class LancasterTokenizer(object):
        def __init__(self):
            self.wnl = LancasterStemmer()
        def __call__(self, doc):
            return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]

def main(output_file=None):
    print 'Loading data...'

    if output_file:
        train_data = np.array(pn.read_csv('../res/train.csv',encoding='utf-8'))[:,:]
        test_data = np.array(pn.read_csv('../res/test.csv',encoding='utf-8'))[:,:]
    else:
        train_data = np.array(pn.read_csv('../res/new_train.csv',encoding='utf-8'))[:,:]
        test_data = np.array(pn.read_csv('../res/new_test.csv',encoding='utf-8'))[:,:]

    print 'Formatting data...'

    ys = train_data[:,4:9]    # column 4-8
    yw = train_data[:,9:13]   # column 9-12
    yk = train_data[:,13:28]  # column 13-27

    # These there groups of labels are only used when output_file=None
    ys_test = None if output_file else test_data[:,4:9]
    yw_test = None if output_file else test_data[:,9:13]
    yk_test = None if output_file else test_data[:,13:28]
    Y_test = test_data[:,4:28]

    train_data = list(train_data[:,1]) # column 1, tweets content
    test_data = list(test_data[:,1])

    X_all = train_data + test_data
    train_len = len(train_data)
    test_len = len(test_data)

    print 'Train length: %s, Test length: %s' % (train_len, test_len)

    print 'Cleaning data...'
    X_all = clean(X_all)

    print 'Feature extraction...'
    X_1 = TFIDF(X_all, 1)
    X_2 = TFIDF(X_all, 2)
    X_3 = TFIDF(X_all, 3)
    X_4 = TFIDF(X_all, 4)

    print 'Training and predicting...'
    print '>>> Feature #1'
    out1 = trainModels(X_1, ys, yw, yk, train_len, ys_test, yw_test, yk_test)
    print '>>> Feature #2'
    out2 = trainModels(X_2, ys, yw, yk, train_len, ys_test, yw_test, yk_test)
    print '>>> Feature #3'
    out3 = trainModels(X_3, ys, yw, yk, train_len, ys_test, yw_test, yk_test)
    print '>>> Feature #4'
    out4 = trainModels(X_4, ys, yw, yk, train_len, ys_test, yw_test, yk_test)

    out = (out1+out2+out3+out4)/4

    if output_file:
        write(list(out), output_file)
    else:
        print '========== Summary =========='
        print '#1 RMSE %f' % rmse(Y_test, out1)
        print '#2 RMSE %f' % rmse(Y_test, out2)
        print '#3 RMSE %f' % rmse(Y_test, out3)
        print '#4 RMSE %f' % rmse(Y_test, out4)
        print 'Average RMSE %f' % rmse(Y_test, out)

    print 'Done!'

def clean(data):
    '''
    Clean the tweets
    '''
    for index in range(0,len(data)):
        data[index] = string.lower(data[index])
        data[index] = re.sub(r'RT|\@mention',"",data[index])
        data[index] = re.sub(r'cloud\w+(\s|\W)',"cloud ",data[index])
        data[index] = re.sub(r'rain\w+(\s|\W)',"rain ",data[index])
        data[index] = re.sub(r'hot\w+(\s|\W)',"hot ",data[index])
        data[index] = re.sub(r'thunder\w+(\s|\W)',"thunder ",data[index])
        data[index] = re.sub(r'freeze\w+(\s|\W)',"freeze ",data[index])
        data[index] = re.sub(r'rain\w+(\s|\W)',"rain ",data[index])
        data[index] = re.sub(r' sun\w+(\s|\W)'," sun ",data[index])
        data[index] = re.sub(r' wind\w+(\s|\W)',"wind ",data[index])
    return data

def TFIDF(data , choice):
    '''
    Choose and apply different kinds of feature extraction
    '''
    if (choice == 1):
        print '>>> TfidfVectorizer word (1,3) ngram'
        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',  analyzer='word',token_pattern=r'\w{2,}',ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
        #, tokenizer=Snowball()
        vect = tfv.fit_transform(data)
        return vect
    elif (choice == 2):
        print '>>> TfidfVectorizer char (2,7) ngram'
        tfvc = TfidfVectorizer(norm='l2', min_df=3, max_df=1.0, strip_accents='unicode', analyzer='char', ngram_range=(2,7), use_idf=1, smooth_idf=1, sublinear_tf=1)
        vectc = tfvc.fit_transform(data)
        return vectc
    elif (choice == 3):
        print '>>> TfidfVectorizer word (1,2) ngram LancasterTokenizer'
        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, tokenizer=LancasterTokenizer())
        #, tokenizer=Snowball()
        vect = tfv.fit_transform(data)
        return vect
    elif (choice == 4):
        print '>>> CountVectorizer word (1,3) ngram'
        tfv = CountVectorizer(min_df=3, max_features=None, strip_accents='unicode',  analyzer='word', token_pattern=r'\w{2,}', ngram_range=(1, 3), binary=True)
        vect = tfv.fit_transform(data)
        return vect
    else:
        return []

def trainModels(X_all, ys, yw, yk, train_len, ys_test=None, yw_test=None, yk_test=None):
    '''
    Train models for each of the three groups
    '''

    X = X_all[:train_len]
    X_test = X_all[train_len:]

    y = np.hstack((np.array(ys), np.array(yw), np.array(yk)))
    y_test = np.hstack((np.array(ys_test), np.array(yw_test), np.array(yk_test)))


    print '----- Training `s` labels -----'
    outs_t, outs_p = ridge(X, ys, X_test, ys_test)
    #outs_p = np.clip(outs_p, 0, 1)
    #outs_p = prep.normalize(outs_p, norm='l1')

    print '----- Training `w` labels -----'
    outw_t, outw_p = ridge(X, yw, X_test, yw_test)
    #outw_p = prep.normalize(outw_p, norm='l1')
    #outw_p = np.clip(outw_p, 0, 1)

    print '----- Training `k` labels -----'
    outk_t, outk_p = ridge(X, yk, X_test, yk_test)
    #outk_p = np.clip(outk_p, 0, 1)

    out_t = np.hstack((outs_t, outw_t, outk_t))
    out_p = np.hstack((outs_p, outw_p, outk_p))

    second_out_t, second_out_p = ridge(out_t, y, out_p)
    #second_out_t, second_out_p = randomForest(out_t, y, out_p, y_test)

    outs = np.clip(second_out_p[:,0:5], 0, 1)
    outs = prep.normalize(outs, norm='l1')

    outw = np.clip(second_out_p[:,5:9], 0, 1)
    outw = prep.normalize(outw, norm='l1')
    
    outk = np.clip(second_out_p[:,9:24], 0, 1)

    out = np.hstack((outs, outw, outk))

    return out

def ridge(X, Y, X_test, Y_test=None):

    '''
    Train and predict with Ridge
    '''
    ri = Ridge(alpha=1,tol=0.001,solver='auto',fit_intercept=True)
    ri.fit(X, Y)
    training = ri.predict(X)
    prediction = ri.predict(X_test)

    print 'Training RMSE %f' % rmse(Y, training)

    if Y_test != None:
        print 'Group RMSE %f' % rmse(Y_test, prediction)

    return np.array(training),np.array(prediction)


def randomForest(X, Y, X_test, Y_test=None):
    
    '''
    Train and predict with RF
    '''

    rf = RandomForestRegressor(max_depth = 20, n_estimators = 100, max_features = 100, n_jobs = 6)
    rf.fit(X, Y)
    training = rf.predict(X)
    prediction = rf.predict(X_test)

    print 'Training RMSE %f' % rmse(Y, training)

    if Y_test != None:
        print 'Group RMSE %f' % rmse(Y_test, prediction)

    return np.array(training),np.array(prediction)


def write(pred, output_file):
    '''
    Write data to output_file with submission format
    '''
    testfile = pn.read_csv('../res/test.csv', na_values=['?'], index_col=0)
    pred_df = pn.DataFrame(pred, index=testfile.index, columns=['s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10','k11','k12','k13','k14','k15'])
    pred_df.to_csv('../res/' + output_file)

def rmse(predict_labels, actual_labels):
    '''
    Calculate RMSE (Root Mean Squared Error) of prediciton and actual labels
    '''
    sum = 0
    for i in range(len(predict_labels)):
        for j in range(len(predict_labels[0])):
            sum += math.pow((predict_labels[i][j] - actual_labels[i][j]), 2)
    return math.sqrt(sum / (len(predict_labels[0]) * len(predict_labels)))

if __name__ == '__main__':
    if len(sys.argv) not in [1, 2]:
        print 'Running Format: python main.py (output_file)'
        sys.exit()
    if len(sys.argv) == 1:
        # Use 20% training data for test
        main()
    else:
        # Use real testing data
        output_file = sys.argv[1]
        main(output_file)
