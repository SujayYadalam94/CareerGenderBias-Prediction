import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

f = []
dt_chi = []
rf_chi = []
gnb_chi = []
knn_chi = []
svm_chi = []
logreg_chi = []
nn_chi = []
features_chi = []
times_chi = {}
    
def extract_features(x, y, n_features):
    print(n_features)
    f.append(n_features)
    times_chi[n_features] = {}
    
    X_norm = MinMaxScaler().fit_transform(x)
    
    #filter - Chi-squared
    a = time.time()
    chi_selector = SelectKBest(chi2, k=n_features)
    chi_selector.fit(X_norm, y.to_numpy().ravel())
    chi_support = chi_selector.get_support()
    chi_feature = x.loc[:,chi_support].columns.tolist()
    b = time.time()
    features_chi.append(chi_feature)
    times_chi[n_features]['chi'] = b-a
    x_fs = x.loc[:, chi_feature]
    X_norm = MinMaxScaler().fit_transform(x_fs)

    return X_norm

def run_all_models(x_fs, y, n_features):
    # Decision tree
    a = time.time()
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(clf, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["dt"] = b-a
    dt_chi.append(np.mean(scores))
    print ("Decision Tree: Accuracy: %f Time: %f" %(np.mean(scores),b-a))
    
    # Random forest
    a = time.time()
    model = RandomForestClassifier()
    model.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["rf"] = b-a
    rf_chi.append(np.mean(scores))
    print ("Random forest: Accuracy: %f Time: %f" %(np.mean(scores),b-a))
    
    # Gaussian Naive Bayes
    a = time.time()
    model = GaussianNB()
    model.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["gnb"] = b-a
    gnb_chi.append(np.mean(scores))
    print ("Naive Bayes: Accuracy: %f Time: %f" %(np.mean(scores),b-a))

    # KNN
    a = time.time()
    model = KNeighborsClassifier(n_neighbors=500)
    model.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["knn"] = b-a
    knn_chi.append(np.mean(scores))
    print ("KNN: Accuracy: %f Time: %f" %(np.mean(scores),b-a))
    
    # SVM
    a = time.time()
    clf = SVC()
    clf.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(clf, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["svm"] = b-a
    svm_chi.append(np.mean(scores))
    print ("SVM: Accuracy: %f Time: %f" %(np.mean(scores),b-a))
    
    # Logistic Regression
    a = time.time()
    model = LogisticRegression()
    model.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(clf, x_fs, y.to_numpy().ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["logreg"] = b-a
    logreg_chi.append(np.mean(scores))
    print ("Logistic Regression: Accuracy: %f Time: %f" %(np.mean(scores),b-a))
    
    # Neural net
    a = time.time()
    model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8,5,3), random_state=1)
    model.fit(x_fs, y.to_numpy().ravel())
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, x_fs, y.to_numpy(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    b = time.time()
    times_chi[n_features]["nn"] = b-a
    nn_chi.append(np.mean(scores))
    print ("Neural Net: Accuracy: %f Time: %f" %(np.mean(scores),b-a))

def plot_accuracy():
    print f
    print dt_chi
    plt.plot(f, dt_chi, label='Decision Tree')
    plt.plot(f, rf_chi, label='Random Forest')
    plt.plot(f, gnb_chi, label='Naive Bayes')
    plt.plot(f, knn_chi, label='KNN')
    plt.plot(f, svm_chi, label='SVM')
    plt.plot(f, logreg_chi, label='Logistic Regression')
    plt.plot(f, nn_chi, label='MLP')
    plt.title("Without Latencies: Chi-Squared")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def plot_time():
    dt_time = []
    rf_time = []
    gnb_time = []
    knn_time = []
    svm_time = []
    knn_time = []
    nn_time = []

    for features in times_chi:
        dt_time.append(times_chi[features]['dt'])
        rf_time.append(times_chi[features]['rf'])
        gnb_time.append(times_chi[features]['gnn'])
        knn_time.append(times_chi[features]['knn'])
        svm_time.append(times_chi[features]['svm'])
        logreg_time.append(times_chi[features]['logreg'])
        nn_time.append(times_chi[features]['nn'])
    plt.plot(f, dt_time, label='Decision Tree')
    plt.plot(f, rf_time, label='Random Forest')
    plt.plot(f, gnb_time, label='Naive Bayes')
    plt.plot(f, knn_time, label='KNN')
    plt.plot(f, svm_time, label='SVM')
    plt.plot(f, knn_time, label='Logistic Regression')
    plt.plot(f, nn_time, label='MLP')
    plt.title("Without Latencies: Chi-Squared")
    plt.xlabel("Number of Features")
    plt.ylabel("Runtime")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python all_models.py dataset")
        exit(1)

    # Preprocess and prepare the dataset
    df = pd.read_csv(sys.argv[1])

    # remove empty rows and other countries' y for each country
    df = df.dropna(subset=['dscore'])
    df = df.dropna(how='any')
    df = df.drop(df[df['dscore'] < -0.65].index)

    # separate data into samples and output classes
    x = df.drop(columns=['dscore'])
    y = df.loc[:,['dscore']]#.to_numpy()

    # convert the continuous y values to output classes
    for i in y['dscore'].index.values:
        if (y['dscore'][i] > 0.65):
            y['dscore'][i] = 1;
        else:
            y['dscore'][i] = 0;

    # To use all features, set num_features=len(x.columns)
    num_features = 10
    x_fs = extract_features(x, y, num_features)

    run_all_models(x, y, num_features)

    plot_time()

    plot_accuracy()
