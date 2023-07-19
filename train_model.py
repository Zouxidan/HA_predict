import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import time



#individual model
def svm(x_train, y_train):
    start = time.time()
    par = [2 ** i for i in range(-10, 6)]
    param_grid_sv = [{'kernel': ['rbf'], 'gamma': par, 'C': par}, {'kernel': ['linear'], 'C': par}]
    svm_clf = GridSearchCV(SVC(probability=True), param_grid_sv, cv=5, n_jobs=-1).fit(x_train, y_train)
    svm = svm_clf.best_estimator_.fit(x_train, y_train)
    end = time.time()
    print('*SVM Completed...\ntime: %s Seconds\n' % (end - start))
    return svm


def knn(x_train, y_train):
    par_k=x_train.shape[0] / 2
    par_k = int(par_k + 1)
    start = time.time()
    param_grid_knn = {'n_neighbors': range(1, par_k)}
    knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1).fit(x_train, y_train)
    knn = knn_clf.best_estimator_.fit(x_train, y_train)
    end = time.time()
    print('*KNN Completed...\ntime: %s Seconds\n' % (end - start))
    return knn


def rf(x_train, y_train):
    par_k=x_train.shape[0] / 2
    par_k = int(par_k + 1)
    start = time.time()
    param_grid_rf = {'n_estimators': range(1, par_k), 'max_features': range(1, 20, 5)}
    rf_clf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1).fit(x_train, y_train)
    rf = rf_clf.best_estimator_.fit(x_train, y_train)
    end = time.time()
    print('*RF Completed...\ntime: %s Seconds\n' % (end - start))
    return rf


def lr(x_train, y_train):
    start = time.time()
    lr = LogisticRegression().fit(x_train, y_train)
    end = time.time()
    print('*LR Completed...\ntime: %s Seconds\n' % (end - start))
    return lr


#ensmeble model
def individual_model(x_train, y_train):
    par_k=x_train.shape[0] / 2
    par_k = int(par_k + 1)
    model_name = []
    model_str = ['SVM', 'KNN', 'RF', 'LR']

    # SVM
    start = time.time()
    par = [2 ** i for i in range(-10, 6)]
    param_grid_sv = [{'kernel': ['rbf'], 'gamma': par, 'C': par}, {'kernel': ['linear'], 'C': par}]
    svm_clf = GridSearchCV(SVC(probability=True), param_grid_sv, cv=5, n_jobs=-1).fit(x_train, y_train)
    svm = svm_clf.best_estimator_.fit(x_train, y_train)
    model_name.append(svm)
    end = time.time()
    print('*SVM Completed...\ntime: %s Seconds\n' % (end - start))


    # KNN
    start = time.time()
    param_grid_knn = {'n_neighbors': range(1, par_k)}
    knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, n_jobs=-1).fit(x_train, y_train)
    knn = knn_clf.best_estimator_.fit(x_train, y_train)
    model_name.append(knn)
    end = time.time()
    print('*KNN Completed...\ntime: %s Seconds\n' % (end - start))


    # RF
    start = time.time()
    param_grid_rf = {'n_estimators':range(1,par_k), 'max_features':range(1,20,5)}
    rf_clf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, n_jobs=-1).fit(x_train, y_train)
    rf = rf_clf.best_estimator_.fit(x_train, y_train)
    model_name.append(rf)
    end = time.time()
    print('*RF Completed...\ntime: %s Seconds\n' % (end - start))


    # LR
    start = time.time()
    lr = LogisticRegression().fit(x_train, y_train)
    model_name.append(lr)
    end = time.time()
    print('*LR Completed...\ntime: %s Seconds\n' % (end - start))


    return model_str, model_name


def ensemble_model(x_train, y_train):
    model_str, model_name = individual_model(x_train, y_train)
    ensem = StackingClassifier(estimators=list(zip(model_str, model_name)), final_estimator=LogisticRegression()).fit(x_train, y_train)


#    for clf, label in zip([model_name[0],ensem],['SVM','Ensemble']):
#        scores = model_selection.cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
#        print("Accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))

    return ensem

def feature_subset(train_data,test_data):
    all_feature_train = pd.DataFrame(train_data[0].iloc[:,1:])
    all_feature_test = pd.DataFrame(test_data[0].iloc[:,1:])
    train_data_y = train_data[0].iloc[:,0]
    test_data_y = test_data[0].iloc[:, 0]

    for i in range(1,len(train_data)):
        train_fea = pd.DataFrame(train_data[i].iloc[:,1:])
        all_feature_train = pd.concat([all_feature_train,train_fea],axis=1)

    for i in range(1,len(test_data)):
        test_fea = pd.DataFrame(test_data[i].iloc[:,1:])
        all_feature_test = pd.concat([all_feature_test,test_fea],axis=1)
    print('*Feature fusion completed...\n')

    selector = SelectKBest(f_classif, k=773)
    selector.fit(pd.DataFrame(all_feature_train),list(train_data_y))
    feature_train = pd.DataFrame(selector.transform(all_feature_train))
    feature_index = selector.get_support(indices=True)
    feature_index = list(feature_index)
    feature_test = pd.DataFrame(all_feature_test.iloc[:,feature_index])
    train_feature = pd.DataFrame(pd.concat([train_data_y,feature_train],axis=1))
    test_feature = pd.DataFrame(pd.concat([test_data_y,feature_test],axis=1))

    return train_feature,test_feature


def IFS(train_feature,test_feature):
    feature_subset = []
    feature_subset_test = []
    scores = []

    par = [2 ** i for i in range(-5, 6)]
    param_grid_sv = [{'kernel': ['rbf'], 'gamma': par, 'C': par}, {'kernel': ['linear'], 'C': par}]

    for i in range(1,train_feature.shape[1]):
        train_x = train_feature.iloc[:, 1:]
        train_x = pd.DataFrame(train_x, columns=train_x.columns)
        train_y = pd.Series(list(train_feature.iloc[:, 0]))
        selector = SelectKBest(f_classif, k=i)
        selector.fit(train_x,train_y)
        train_x = pd.DataFrame(selector.transform(train_x))
        index = selector.get_support(indices=True)
        index_test = []
        for a in range(0,len(index)):
            index_test.append(index[a] + 1)
        index_test = list(index_test)
        print(index)
        feature_subset_test.append(pd.DataFrame(test_feature.iloc[:,index_test]))
        svm_clf = GridSearchCV(SVC(probability=True), param_grid_sv, cv=5, n_jobs=-1).fit(train_x, train_y)
        score = svm_clf.best_score_
        print("Accuracy: %0.4f  fea[%s]" % (score, i-1))
        feature_subset.append(train_x)
        scores.append(score)

    i = scores.index(max(scores))
    train_subset = pd.DataFrame(pd.concat([train_feature.iloc[:,0],feature_subset[i]],axis=1))
    test_subset = pd.DataFrame(pd.concat([test_feature.iloc[:,0],feature_subset_test[i]],axis=1))
    scores = pd.DataFrame(scores)

    return train_subset,test_subset,scores



def get_result(train_data, test_data):
    y_original_label_valid,y_original_label_test, y_proba_valid_all, y_proba_test_all = [], [], [], []
    kf = StratifiedKFold(n_splits=5)
    y = train_data.iloc[:, 0]
    x = train_data.iloc[:, 1:]
    X_test = test_data.iloc[:, 1:]
    Y_test = pd.DataFrame(test_data.iloc[:, 0])

    # individual model
    #svm_clf = svm(x,y)
    #knn_clf = knn(x,y)
    #rf_clf = rf(x,y)
    #lr_clf = lr(x,y)

    ensemble_model2 = ensemble_model(x,y)

    for train_site, valida_site in kf.split(x, y):

        Y_train, Y_valid = train_data.iloc[train_site, :].iloc[:, 0], pd.DataFrame(train_data.iloc[valida_site, :].iloc[:, 0])
        X_train = train_data.iloc[train_site, 1:]
        X_valid = train_data.iloc[valida_site, 1:]


        #individual model
        #individual_clf = rf_clf.fit(X_train,Y_train)
        #y_proba_valid = individual_clf.predict_proba(X_valid)[:, 1]

        #ensemble model
        ensemble_clf = ensemble_model2.fit(X_train, Y_train)
        y_proba_valid = ensemble_clf.predict_proba(X_valid)[:, 1]

        Y_valid = Y_valid.astype('int64')
        y_proba_valid_all.append(y_proba_valid)
        y_original_label_valid.append(Y_valid)

    #y_proba_test =rf_clf.predict_proba(X_test)[:, 1]
    y_proba_test = ensemble_model2.predict_proba(X_test)[:, 1]
    Y_test = Y_test.astype('int64')
    y_original_label_test=Y_test
    y_proba_test_all=y_proba_test

    return y_proba_valid_all, y_proba_test_all, y_original_label_valid,y_original_label_test
