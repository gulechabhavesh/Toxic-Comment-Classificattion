import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import gc
from sklearn.linear_model import Ridge
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier # <3
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model.stochastic_gradient import SGDClassifier
import re, string
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
subm = pd.read_csv('../input/sample_submission.csv')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class Preprocess():

    def __init__(self):
        print("Preprocess object created")

    def _feature_engineering(self,all_text):
            print("Inside word vectorizer")
            word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),min_df=2,max_df=0.5,
            max_features=60000
            )
            word_vectorizer.fit(all_text)
            train_word_features = word_vectorizer.transform(train_text)
            test_word_features = word_vectorizer.transform(test_text)
            char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            stop_words='english',
            ngram_range=(2, 6),min_df=2,max_df=0.5,
            max_features=60000
            )
            char_vectorizer.fit(all_text)
            train_char_features = char_vectorizer.transform(train_text)
            test_char_features = char_vectorizer.transform(test_text)
            train_features = hstack([train_char_features,train_word_features])
            test_features = hstack([test_char_features,test_word_features])
            print("Exiting vectorizer")
            return train_features,test_features



class Models():

    def __init__(self):
        print("Model Created")

    def logistic(self,train_features,test_features):

       print("in logistic")
       self.train_features=train_features
       self.test_features=test_features
       scores = []
       submission = pd.DataFrame.from_dict({'id': test['Id']})
       logistic_file='logistic.pckl'
       logistic_model_pkl = open(logistic_file, 'wb')
       for class_name in class_names:
            train_target = train[class_name]
            classifier = LogisticRegression(C=1, solver='sag')

            cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
            scores.append(cv_score)
            print('CV score for class {} is {}'.format(class_name, cv_score))

            classifier.fit(train_features, train_target)
            pickle.dump(classifier, logistic_model_pkl)
            submission[class_name] = classifier.predict_proba(test_features)[:, 1]

       print('Total CV score is {}'.format(np.mean(scores)))
       logistic_model_pkl.close()
       submission.to_csv('logistic.csv', index=False)


    def SGD(self,train_features,test_features):
       print("in SGD")
       self.train_features=train_features
       self.test_features=test_features
       scores = []
       submission = pd.DataFrame.from_dict({'id': test['Id']})
       SGD_file='SGD.pckl'
       SGD_model_pkl = open(SGD_file, 'wb')
       for class_name in class_names:
            train_target = train[class_name]
            classifier = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.001, random_state=42, max_iter=200, tol=0.20, learning_rate='optimal')

            cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
            scores.append(cv_score)
            print('CV score for class {} is {}'.format(class_name, cv_score))

            classifier.fit(train_features, train_target)
            pickle.dump(classifier, SGD_model_pkl)
            submission[class_name] = classifier.predict_proba(test_features)[:, 1]

       print('Total CV score is {}'.format(np.mean(scores)))
       SGD_model_pkl.close()
       submission.to_csv('SGD.csv', index=False)

    def ridge(self,train_features,test_features):
       print("in ridge")
       self.train_features=train_features
       self.test_features=test_features
       scores = []
       submission = pd.DataFrame.from_dict({'id': test['Id']})
       ridge_file='ridge.pckl'
       ridge_model_pkl = open(ridge_file, 'wb')
       for class_name in class_names:
            train_target = train[class_name]
            classifier = Ridge(alpha=29, copy_X=True, fit_intercept=True, solver='sag',
                     max_iter=150,   normalize=False, random_state=0,  tol=0.0025)
            cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
            scores.append(cv_score)
            print('CV score for class {} is {}'.format(class_name, cv_score))

            classifier.fit(train_features, train_target)
            # save the model to disk
            pickle.dump(classifier, ridge_model_pkl)
            submission[class_name] = classifier.predict(test_features)

       print('Total CV score is {}'.format(np.mean(scores)))
       ridge_model_pkl.close()
       submission.to_csv('ridge.csv', index=False)

    def xgb(self,train_features,test_features):
           print("in xgb")
           cv_scores = []
           xgb_preds = []
           submission = pd.DataFrame.from_dict({'id': test['Id']})
           self.train_features=train_features
           self.test_features=test_features
           d_test = xgb.DMatrix(test_features)
           xgb_file='xgb.pckl'
           xgb_model_pkl = open(xgb_file, 'wb')
           for class_name in class_names:
                train_target = train[class_name]
    # Split out a validation set
                X_train, X_valid, y_train, y_valid = train_test_split(
                    train_features, train_target, test_size=0.25, random_state=23)

                xgb_params = {'eta': 0.3,
                  'max_depth': 5,
                  'subsample': 0.8,
                  'colsample_bytree': 0.8,
                  'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'seed': 23
                 }

               # trn_lgbset = lgb.Dataset(csr_trn, free_raw_data=False)
                d_train = xgb.DMatrix(X_train, y_train)
                d_valid = xgb.DMatrix(X_valid, y_valid)
                watchlist = [(d_valid, 'valid')]
              #  model = xgb.train(xgb_params, d_train, 200, watchlist, verbose_eval=False, early_stopping_rounds=30)
                model = xgb.train(xgb_params, d_train, 200, watchlist, verbose_eval=False, early_stopping_rounds=30)
                pickle.dump(model, xgb_model_pkl)
                print("class Name: {}".format(class_name))
                print(model.attributes()['best_msg'])
                cv_scores.append(float(model.attributes()['best_score']))
                submission[class_name] = model.predict(d_test)
                del X_train, X_valid, y_train, y_valid
                gc.collect()
           print('Total CV score is {}'.format(np.mean(cv_scores)))
           xgb_model_pkl.close()
           submission.to_csv('xgb.csv', index=False)

    def randomforest(self,train_features,test_features):
       label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
       print("in random forest")
       re_tok = re.compile('([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
       def tokenize(s): return re_tok.sub(r' \1 ', s).split()

       n = train.shape[0]
       vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
       train_term_doc = vec.fit_transform(train['comment_text'])
       test_term_doc = vec.transform(test['comment_text'])
       preds = np.zeros((len(test), len(label_cols))) # empty np matrix to put in predictions
       random_file='randomforest.pckl'
       random_model_pkl = open(random_file, 'wb')
       for i, j in enumerate(label_cols):
         print('fit', j)
         m = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=18, random_state=21)
         m.fit(train_term_doc, train[j].values)
         pickle.dump(m, random_model_pkl)
         preds[:,i] = m.predict_proba(test_term_doc)[:,1]
       random_model_pkl.close()
       submid = pd.DataFrame({'id': subm["Id"]})
       submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
       submission.to_csv('random_forest.csv', index=False)


class Toxic():

    def __init__(self, all_text):

        #Create instance of objects
        print("Toxic object created")
        self.preprocess=Preprocess()
        self.models=Models()
        self.all_text=all_text



    def preprocessing(self, all_text):

            self.preprocess._feature_engineering(self.all_text)

    def machine_learning(self):

           self.train_features,self.test_features=self.preprocess._feature_engineering_word(self.all_text)

           self.models.randomforest(self.train_features,self.test_features)
           self.models.ridge(self.train_features,self.test_features)
           self.models.logistic(self.train_features,self.test_features)
           self.models.SGD(self.train_features,self.test_features)
           self.models.xgb(self.train_features,self.test_features)



Toxic=Toxic(all_text)
Toxic.machine_learning()
