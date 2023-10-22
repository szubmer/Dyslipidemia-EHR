import GPy
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, \
    roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

from feature_extra import *
from data_processing import interpolation_data


def evaluation(y_test, y_prob, threshold=0.3):
    # Calculate recall, precision and other indicators
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, recall, precision, f1score, roc_auc, sensitivity, specificity


def feature_select_ttest(feature, feature_test, label):
    # Feature selection using t-test
    w, h = np.shape(feature)
    feature_pd = pd.DataFrame(feature, columns=[f'feature_{i}' for i in range(0, h)])
    feature_test_pd = pd.DataFrame(feature_test, columns=[f'feature_{i}' for i in range(0, h)])
    t_test_results = pd.DataFrame(columns=['Feature', 't_value', 'p_value'])
    for feature in feature_pd:
        label_0 = feature_pd[label == 0][feature]
        label_1 = feature_pd[label == 1][feature]

        t_stat, p_value = stats.ttest_ind(label_0, label_1)
        t_test_results = pd.concat([t_test_results, pd.DataFrame({'Feature': [feature], 't_value': [t_stat], 'p_value': [p_value]})])

    # 筛选出显著性水平为0.05的特征
    significant_features = t_test_results[t_test_results['p_value'] < 0.05]
    feature_filtered = feature_pd[significant_features['Feature'].values]
    feature_test_filtered = feature_test_pd[significant_features['Feature'].values]

    return feature_filtered, feature_test_filtered, significant_features


def plot_roc_curve(prob_all, label_all, roc_auc, model_name='SVM'):
    # Draw ROC curve
    fpr, tpr, thresholds = roc_curve(label_all, prob_all)
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1-Specificity', fontsize=22)
    plt.ylabel('Sensitivity', fontsize=22)
    plt.title(model_name, fontsize=28)
    plt.legend(loc="lower right")
    plt.show()


def model_all(model_name='LSTM'):
    # Build the model, train and test it
    data, label = interpolation_data()

    data = data.to_numpy()
    label = np.array(label)
    feature_orig = original_feature(data)
    feature_diff = original_diff(data)
    feature = np.concatenate([feature_orig, feature_diff], axis=1)

    w, h = np.shape(feature)
    feature_pd = pd.DataFrame(feature, columns=[f'feature_{i}' for i in range(0, h)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    recalls = []
    precisions = []
    f1scores = []
    roc_aucs = []
    sensitivities = []
    specificities = []

    if model_name=='LR':
        classifier = LogisticRegression(C=100, max_iter=2000, solver='newton-cholesky', random_state=42)
    elif model_name=='SVM':
        classifier = SVC(C=0.1, kernel='linear', gamma=0.1, probability=True)
    elif model_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=40)
    elif model_name == 'XGBoost':
        classifier = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    elif model_name == 'Random Forest':
        classifier = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt', random_state=42)
    elif model_name == 'Naive Bayes':
        classifier = GaussianNB(var_smoothing=1e-9)
    elif model_name == 'LDA':
        classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)

    prob_all = np.array([0, 1])
    label_all = np.array([0, 1])
    for train_index, test_index in kf.split(feature_pd):
        X_train, X_test = feature_pd.iloc[train_index], feature_pd.iloc[test_index]
        y_train, y_test = label[train_index], label[test_index]

        if model_name in ['LR', 'SVM', 'KNN', 'Naive Bayes', 'LDA', 'LSTM', 'Gaussian Process', 'LSTM']:
            # 使用随机欠采样进行数据平衡
            undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X_train_sampled, y_train_sampled = undersampler.fit_resample(X_train, y_train)
        elif model_name in ['XGBoost', 'Random Forest']:
            X_train_sampled, y_train_sampled = X_train, y_train

        X_train_filtered, X_test_filtered, significant_features = feature_select_ttest(X_train_sampled, X_test, y_train_sampled)

        scaler = StandardScaler()
        X_train_filtered = scaler.fit_transform(X_train_filtered)
        X_test_filtered = scaler.transform(X_test_filtered)

        if model_name in ['LR', 'SVM']:
            # 使用mutual_info_classif作为评价指标
            selector = SelectKBest(score_func=mutual_info_classif, k=7)
            X_train_filtered = selector.fit_transform(X_train_filtered, y_train_sampled)
            X_test_filtered = selector.transform(X_test_filtered)

        elif model_name in ['KNN', 'Naive Bayes', 'Gaussian Process', 'LDA', 'LSTM']:
            # # 使用f_classif作为评价指标
            selector = SelectKBest(score_func=f_classif, k=7)
            X_train_filtered = selector.fit_transform(X_train_filtered, y_train_sampled)
            X_test_filtered = selector.transform(X_test_filtered)

        if model_name in ['Gaussian Process']:
            # 创建高斯过程模型
            w, h = np.shape(X_train_filtered)
            kernel = GPy.kern.RBF(input_dim=h)  # 可以尝试不同的核函数
            classifier = GPy.models.GPClassification(X_train_filtered, y_train_sampled.reshape(-1, 1), kernel=kernel)
            classifier.optimize()
            y_prob, _ = classifier.predict(X_test_filtered)

        if model_name in ['LSTM']:
            w, h = np.shape(X_train_filtered)
            classifier = Sequential()
            classifier.add(LSTM(units=128, input_shape=(h, 1), activation='relu'))
            classifier.add(Dropout(0.3))
            classifier.add(Dense(units=1, activation='sigmoid'))
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
            classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            classifier.fit(X_train_filtered, y_train_sampled)
            y_prob = classifier.predict(X_test_filtered)

        if model_name in ['LR', 'SVM', 'KNN', 'Naive Bayes', 'LDA', 'XGBoost', 'Random Forest']:
            classifier.fit(X_train_filtered, y_train_sampled)
            y_prob = classifier.predict_proba(X_test_filtered)[:, 1]

        threshold = sum(y_train_sampled) / len(y_train_sampled)
        accuracy, recall, precision, f1score, roc_auc, sensitivity, specificity = evaluation(y_test, y_prob, threshold=threshold)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1scores.append(f1score)
        roc_aucs.append(roc_auc)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

        prob_all = np.concatenate([prob_all, np.squeeze(y_prob)])
        label_all = np.concatenate([label_all, y_test])

    # Calculate the average accuracy across 5 folds
    average_accuracy = sum(accuracies) / len(accuracies)
    average_recall = sum(recalls) / len(accuracies)
    average_precision = sum(precisions) / len(accuracies)
    average_f1score = sum(f1scores) / len(accuracies)
    average_roc_auc = sum(roc_aucs) / len(accuracies)
    sensitivity = sum(sensitivities) / len(accuracies)
    specificity = sum(specificities) / len(accuracies)

    prob_all, label_all = prob_all[2:], label_all[2:]
    plot_roc_curve(prob_all, label_all, average_roc_auc, model_name=model_name)

    print(
        f"accuracy:{average_accuracy}, roc_auc:{average_roc_auc}, sensitivity:{sensitivity}, specificity:{specificity}")


if '__main__' == __name__:
    # model_name = ['LR', 'SVM', 'XGBoost', 'Random Forest', 'KNN', 'Gaussian Process', 'LDA', 'LSTM']
    model_all(model_name='LR')
