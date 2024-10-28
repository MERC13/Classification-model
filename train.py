import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('seaborn')

df_bank = pd.read_csv('https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv')

df_bank = df_bank.drop('duration', axis=1)

# print(df_bank.info())
# print('Shape of dataframe:', df_bank.shape)
# print(df_bank.head())
# print(df_bank['deposit'].value_counts())
# print(df_bank.isnull().sum())



from sklearn.preprocessing import StandardScaler

# Copying original dataframe
df_bank_ready = df_bank.copy()

scaler = StandardScaler()
num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
df_bank_ready[num_cols] = scaler.fit_transform(df_bank_ready[num_cols])

# print(df_bank_ready.head())



from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Encode Categorical Data
df_encoded = pd.DataFrame(encoder.fit_transform(df_bank_ready[cat_cols]))
df_encoded.columns = encoder.get_feature_names_out(cat_cols)

# Replace Categotical Data with Encoded Data
df_bank_ready = df_bank_ready.drop(cat_cols ,axis=1)
df_bank_ready = pd.concat([df_encoded, df_bank_ready], axis=1)

# Encode target value
df_bank_ready['deposit'] = df_bank_ready['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

# print('Shape of dataframe:', df_bank_ready.shape)
# print(df_bank_ready.head())



# Select Features
feature = df_bank_ready.drop('deposit', axis=1)

# Select Target
target = df_bank_ready['deposit']

# Set Training and Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature , target, 
                                                    shuffle = True, 
                                                    test_size=0.2, 
                                                    random_state=1)

# Show the Training and Testing Data
# print('Shape of training feature:', X_train.shape)
# print('Shape of testing feature:', X_test.shape)
# print('Shape of training label:', y_train.shape)
# print('Shape of training label:', y_test.shape)



def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}
    
def decision_tree():
    from sklearn import tree

    # Building Decision Tree model 
    dtc = tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X_train, y_train)

    # Evaluate Model
    dtc_eval = evaluate_model(dtc, X_test, y_test)

    # Print result
    print('Accuracy:', dtc_eval['acc'])
    print('Precision:', dtc_eval['prec'])
    print('Recall:', dtc_eval['rec'])
    print('F1 Score:', dtc_eval['f1'])
    print('Cohens Kappa Score:', dtc_eval['kappa'])
    print('Area Under Curve:', dtc_eval['auc'])
    print('Confusion Matrix:\n', dtc_eval['cm'])
    
    return dtc_eval
def random_forest():
    from sklearn.ensemble import RandomForestClassifier

    # Building Random Forest model
    rf = RandomForestClassifier(random_state=0)
    rf.fit(X_train, y_train)

    # Evaluate Model
    rf_eval = evaluate_model(rf, X_test, y_test)

    # Print result
    print('\nRandom Forest Results:')
    print('Accuracy:', rf_eval['acc'])
    print('Precision:', rf_eval['prec'])
    print('Recall:', rf_eval['rec'])
    print('F1 Score:', rf_eval['f1'])
    print('Cohens Kappa Score:', rf_eval['kappa'])
    print('Area Under Curve:', rf_eval['auc'])
    print('Confusion Matrix:\n', rf_eval['cm'])
    
    return rf_eval

def naive_bayes():
    from sklearn.naive_bayes import GaussianNB

    # Building Naive Bayes model
    nb = GaussianNB()
    nb.fit(X_train, y_train)

    # Evaluate Model
    nb_eval = evaluate_model(nb, X_test, y_test)

    # Print result
    print('\nNaive Bayes Results:')
    print('Accuracy:', nb_eval['acc'])
    print('Precision:', nb_eval['prec'])
    print('Recall:', nb_eval['rec'])
    print('F1 Score:', nb_eval['f1'])
    print('Cohens Kappa Score:', nb_eval['kappa'])
    print('Area Under Curve:', nb_eval['auc'])
    print('Confusion Matrix:\n', nb_eval['cm'])
    
    return nb_eval

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    # Ensure X_train and X_test are numpy arrays
    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)

    # Check if the arrays are C-contiguous
    if not X_train_array.flags['C_CONTIGUOUS']:
        X_train_array = np.ascontiguousarray(X_train_array)
    if not X_test_array.flags['C_CONTIGUOUS']:
        X_test_array = np.ascontiguousarray(X_test_array)

    # Building K-Nearest Neighbors model
    knn = KNeighborsClassifier(n_neighbors=5)  # Specify the number of neighbors
    knn.fit(X_train_array, y_train)

    # Evaluate Model
    knn_eval = evaluate_model(knn, X_test_array, y_test)

    # Print result
    print('\nK-Nearest Neighbors Results:')
    print('Accuracy:', knn_eval['acc'])
    print('Precision:', knn_eval['prec'])
    print('Recall:', knn_eval['rec'])
    print('F1 Score:', knn_eval['f1'])
    print('Cohens Kappa Score:', knn_eval['kappa'])
    print('Area Under Curve:', knn_eval['auc'])
    print('Confusion Matrix:\n', knn_eval['cm'])
    
    return knn_eval
    
def model_comparison():
    # Intitialize figure with two plots
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(14)
    fig.set_facecolor('white')

    # First plot
    ## set bar size
    barWidth = 0.2
    dtc_score = [dtc_eval['acc'], dtc_eval['prec'], dtc_eval['rec'], dtc_eval['f1'], dtc_eval['kappa']]
    rf_score = [rf_eval['acc'], rf_eval['prec'], rf_eval['rec'], rf_eval['f1'], rf_eval['kappa']]
    nb_score = [nb_eval['acc'], nb_eval['prec'], nb_eval['rec'], nb_eval['f1'], nb_eval['kappa']]
    knn_score = [knn_eval['acc'], knn_eval['prec'], knn_eval['rec'], knn_eval['f1'], knn_eval['kappa']]

    ## Set position of bar on X axis
    r1 = np.arange(len(dtc_score))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    ## Make the plot
    ax1.bar(r1, dtc_score, width=barWidth, edgecolor='white', label='Decision Tree')
    ax1.bar(r2, rf_score, width=barWidth, edgecolor='white', label='Random Forest')
    ax1.bar(r3, nb_score, width=barWidth, edgecolor='white', label='Naive Bayes')
    ax1.bar(r4, knn_score, width=barWidth, edgecolor='white', label='K-Nearest Neighbors')

    ## Configure x and y axis
    ax1.set_xlabel('Metrics', fontweight='bold')
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'Kappa']
    ax1.set_xticks([r + (barWidth * 1.5) for r in range(len(dtc_score))], )
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_ylim(0, 1)

    ## Create legend & title
    ax1.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    ax1.legend()

    # Second plot
    ## Comparing ROC Curve
    ax2.plot(dtc_eval['fpr'], dtc_eval['tpr'], label='Decision Tree, auc = {:0.5f}'.format(dtc_eval['auc']))
    ax2.plot(rf_eval['fpr'], rf_eval['tpr'], label='Random Forest, auc = {:0.5f}'.format(rf_eval['auc']))
    ax2.plot(nb_eval['fpr'], nb_eval['tpr'], label='Naive Bayes, auc = {:0.5f}'.format(nb_eval['auc']))
    ax2.plot(knn_eval['fpr'], knn_eval['tpr'], label='K-Nearest Nieghbor, auc = {:0.5f}'.format(knn_eval['auc']))

    ## Configure x and y axis
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')

    ## Create legend & title
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc=4)

    plt.show()

# dtc_eval = decision_tree()
# rf_eval = random_forest()
# nb_eval = naive_bayes()
# knn_eval = knn()
# model_comparison()




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [50, 80, 100],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300, 500, 750, 1000]
}

# Create a base model
# rf_grids = RandomForestClassifier(random_state=0)

# # Initiate the grid search model
# grid_search = GridSearchCV(estimator=rf_grids, param_grid=param_grid, scoring='recall',
#                            cv=5, n_jobs=-1, verbose=2)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)




# Select best model with best fit
# best_grid = grid_search.best_estimator_

# # Evaluate Model
# best_grid_eval = evaluate_model(best_grid, X_test, y_test)

# Print result
# print('Accuracy:', best_grid_eval['acc'])
# print('Precision:', best_grid_eval['prec'])
# print('Recall:', best_grid_eval['rec'])
# print('F1 Score:', best_grid_eval['f1'])
# print('Cohens Kappa Score:', best_grid_eval['kappa'])
# print('Area Under Curve:', best_grid_eval['auc'])
# print('Confusion Matrix:\n', best_grid_eval['cm'])




from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

df_bank['deposit_prediction'] = rf.predict(feature)
df_bank['deposit_prediction'] = df_bank['deposit_prediction'].apply(lambda x: 'yes' if x==0 else 'no')

# Save new dataframe into csv file
df_bank.to_csv('deposit_prediction.csv', index=False)

print(df_bank.head(10))




from joblib import dump, load

# Saving model
dump(rf, 'bank_deposit_classification.joblib')
# Loading model
clf = load('bank_deposit_classification.joblib') 