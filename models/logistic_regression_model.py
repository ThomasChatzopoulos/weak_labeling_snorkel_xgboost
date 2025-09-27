import pandas as pd

from shutil import copy
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from yaml import load, FullLoader

from weak_labeling_snorkel_xgboost.utils.utils import createDir, printTimestamp
from weak_labeling_snorkel_xgboost.models.utils import readFiles, completeDataFrame
from weak_labeling_snorkel_xgboost.models.model_functions import (
    findBestNumOfFeatures,
    applyFeatureExtraction,
    applyFeatureSelection,
    applyBalancing,
    applyTextPreprocessing,
    calcPredictionStats,
)


settings = load(open("../../weak_labeling/settings.yaml"), Loader=FullLoader)

for year in range(settings["firstYear"], settings["lastYear"] + 1):
    print(f'\n\n\n{printTimestamp()}  **** year: {year}')
    workingPath = f"{settings['workingPath']}/snorkel_results_{year}_0_2_6"
    createDir(workingPath, f"reports_{settings['useAsTrainDataset']}")
    copy("../../weak_labeling/settings.yaml", workingPath)

    print(printTimestamp(), " -- Loading data")
    train, test, weakly_test, descr_ids = readFiles(year, f'{settings["filesPath"]}/Dataset_SI_old_{year}')

    train_snorkel = pd.read_csv(f"{workingPath}/label_matrix_train_{settings['useAsTrainDataset']}_{year}.csv")
    train_snorkel = completeDataFrame(train_snorkel, train, descr_ids)

    if settings["balanceDataset"]:
        train_snorkel = applyBalancing(train_snorkel, descr_ids, settings, year)

    if settings["datasetTextPreprocessing"]:
        train_snorkel, test = applyTextPreprocessing(
            train_snorkel, test, descr_ids, year, settings
        )

    if settings["predictNumOfFeatures"]:
        numOfFeatures = findBestNumOfFeatures(
            train_snorkel,
            descr_ids,
            settings["folds"],
            f"{workingPath}/reports",
            round(len(train_snorkel) / 20),
            OneVsRestClassifier(LogisticRegression()),
        )
    else:
        numOfFeatures = settings["numOfFeatures"]

    tfidf_train, vocabulary = applyFeatureExtraction(train_snorkel["text"], None)
    tfidf_train_featured, shared_vocabulary = applyFeatureSelection(
        train_snorkel, tfidf_train, vocabulary, descr_ids, numOfFeatures
    )
    tfidf_test_featured, v = applyFeatureExtraction(
        test["text"], vocabulary=shared_vocabulary
    )

    y_train = (train_snorkel[descr_ids]).values.tolist()
    y_test = (test[descr_ids]).values.tolist()

    print(printTimestamp(), " -- Applying logistic LogisticRegression model")
    clf = OneVsRestClassifier(LogisticRegression(penalty='l2',
                                                 dual=False,
                                                 tol=0.001,
                                                 C=10,
                                                 fit_intercept=True,
                                                 intercept_scaling=1,
                                                 class_weight=None,
                                                 random_state=None,
                                                 solver='liblinear',
                                                 max_iter=1000,
                                                 multi_class='auto',
                                                 verbose=0,
                                                 warm_start=False,
                                                 n_jobs=None,
                                                 l1_ratio=None))
    clf.fit(tfidf_train_featured, y_train)
    y_pred_test = clf.predict(tfidf_test_featured)

    calcPredictionStats(
        y_pred_test, y_test, descr_ids, settings, year, numOfFeatures, workingPath, "LR"
    )
