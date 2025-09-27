import string
import operator

import pandas as pd

from math import ceil
from itertools import compress
from random import shuffle, randint
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from os.path import exists

from weak_labeling_snorkel_xgboost.utils.utils import (
    save_evaluation_report,
    createDir,
    printTimestamp,
)
from weak_labeling_snorkel_xgboost.models.utils import getPositionOfInstances, mergeReports
from weak_labeling_snorkel_xgboost.snorkel_labeling.snorkel_model_functions import filtering


#################################################
#    Under-sampling / balancing the dataset     #
#################################################


def stats_on_instance_validity(input_data, target_labels):
    """

    :param input_data:
    :param target_labels:
    :return:
    """
    positive_instances = input_data[target_labels].sum()
    valid_instances = {}
    valid_negative_instances = {}
    for label in target_labels:
        valid_instances[label] = input_data["valid_labels"].str.contains(label).sum()
        valid_negative_instances[label] = (
            valid_instances[label] - positive_instances[label]
        )
    print("\tValid_instances per label :")
    print(valid_instances)
    print("\tValid_negative_instances per label :")
    print(valid_negative_instances)
    return valid_instances, positive_instances


def balance_dataset(input_data, target_labels, balance_n, filename):
    """
    Use this function to balance the dataset. The ratio which is adjusted
    negative_instances/positive_instances <= balance_n
    :param input_data:      The train dataset
                            (required columns: valid_labels,[target_labels])
                            type: pandas DataFrame
    :param target_labels:   List with descriptors.
                            type: (string) List
    :param balance_n:       The balance ration.
                            type: Integer
    :param filename:        The filename-path to store the balanced dataset.
                            type: String
    :return:
    """
    # TODO: Could we use RandomUnderSampler
    #  (sampling_strategy= ... instead of this custom implementation?
    # remove unwanted samples (e.g. for undersampling)

    seed = randint(0, 99)

    positive_instances = input_data[target_labels].sum()

    remove_pmids = []
    if balance_n is not None:
        # balance the dataset removing fully negative articles
        input_data["label_count"] = input_data.loc[:, target_labels].sum(axis=1)
        all_negative_instances = input_data.loc[input_data["label_count"] == 0].copy()
        print("\tlen(input_data) :", str(len(input_data)))
        print("\tlen(All negative_instances) :", str(len(all_negative_instances)))
        # print("\tall_negative_instances :", all_negative_instances)

        all_positive_instances = len(input_data) - len(all_negative_instances)
        print("\tlen(All positive_instances) :", str(all_positive_instances))

        valid_instances = {}
        valid_negative_instances = {}
        for label in target_labels:
            valid_instances[label] = (
                input_data["valid_labels"].str.contains(label).sum()
            )
            valid_negative_instances[label] = (
                valid_instances[label] - positive_instances[label]
            )
        print("\tvalid_instances per label :")
        print(valid_instances)
        print("\tValid_negative_instances per label :")
        print(valid_negative_instances)

        # Condition to meet
        # For each label, valid negative instances should count at most
        # balance_n times the positive ones
        condition_met = False

        initial_negative = len(all_negative_instances)
        counter = 0
        while not condition_met and not all_negative_instances.empty:
            # print progress every 100 steps
            counter = counter + 1
            if counter % 10000 == 0:
                print(
                    printTimestamp(),
                    "\t",
                    str(round(counter / initial_negative, 3)),
                    " % of ",
                    str(initial_negative),
                    " negative instances considered. ",
                    str(
                        round(1.0 - (len(all_negative_instances) / initial_negative), 3)
                    ),
                    " % negative reduction so far.",
                )
            # Run until the condition is met, or no more unecesary instances
            # are available to remove for meetting it
            #     random_index = random.randint(0, len(all_negative_instances)-1)
            random_index = all_negative_instances.sample(random_state=seed).index
            # position_index = test_data.index.get_loc(initial_index)
            # Each instance is randomly selected and checked only once
            random_negative_instance = all_negative_instances.loc[random_index]
            # print('random_negative_instance', random_negative_instance)
            # print('random_negative_instance[pmid]', random_negative_instance['pmid'])
            # print('random_negative_instance.index', random_negative_instance.index)
            # Check if necessary for at least on label
            necessary = False
            # All valid labels for this instance
            valid_labels = random_negative_instance["valid_labels"].values[0].split(" ")
            # keep only valid labels of focus. Valid labels for which we don't need
            # to learn a model are not considered.
            valid_labels = set.intersection(set(valid_labels), set(target_labels))
            # print('valid_labels ', valid_labels)

            # valid_labels = random_negative_instance['valid_labels'].split(" ")
            for valid_label in valid_labels:
                if (
                    valid_negative_instances[valid_label]
                    <= positive_instances[valid_label] * balance_n
                ):
                    # This instance is necessary for this label. Do nothing.
                    necessary = True
            if not necessary:
                # This instance is not necessary for any label, it can be removed
                # from the training dataset, remove a random negative
                remove_pmids.append(random_negative_instance["pmid"].values[0])
                # Update valid negative values to reflect new situation after the removal
                for valid_label in valid_labels:
                    valid_negative_instances[valid_label] = (
                        valid_negative_instances[valid_label] - 1
                    )
            # Check all labels to update the condition
            condition_met = True
            for label in target_labels:
                if (
                    valid_negative_instances[label]
                    > positive_instances[label] * balance_n
                ):
                    condition_met = False
            #   remove this instance from the dataframe as each instance is only checked once
            all_negative_instances.drop(random_index, inplace=True)

        print("\tremove_pmids (final) :", len(remove_pmids))
        # print('\tremove_pmids (final) :', remove_pmids)

    data = input_data[~input_data["pmid"].isin(remove_pmids)]
    stats_on_instance_validity(data, target_labels)
    data.to_csv(filename, index=False)
    return data


#################################################
#       Functions for text preprocessing        #
#################################################


def removePunctuation(data):
    """
    Remove punctuation from the text of the articles.

    :param data:    Text with punctuation
                    Type: pandas Dataframe
    :return:        Text without punctuation
                    Type: pandas Dataframe
    """
    punct = string.punctuation
    for index, value in data.items():
        rp_words = []
        words = value.split()
        for word in words:
            for p in punct:
                word = word.replace(p, " ")
            rp_words.append(word)
        data[index] = " ".join(rp_words)
    return data


def applyStemming(data):
    """
    Apply stemming on the text.

    :param data:    Text where words have endings
                    Type: pandas Dataframe
    :return:        Text where words have not endings
                    Type: pandas Dataframe
    """
    ps = PorterStemmer()
    for index, value in data.items():
        st_words = []
        words = value.split()
        for word in words:
            st_words.append(ps.stem(word))
        data[index] = " ".join(st_words)
    return data


def removeStopWordsFromString(text, stop_words, lowercase):
    """
    This is a secondary function which implements the basic function of removing the
    stopwords from a string. This function is part of "removeStopwords" function below.

    :param text:        The given text-string from which stopwords will be removed.
                        type: String
    :param stop_words:  Set containing stopwords.
                        type: Set
    :param lowercase:   If 'True', then check for stop-words in lowercase
                        type: Boolean
    :return:
    """
    word_tokens = word_tokenize(text)
    if lowercase:
        return " ".join([w for w in word_tokens if w.lower() not in stop_words])
    else:
        return " ".join([w for w in word_tokens if w not in stop_words])


def removeStopwords(data, lowercase):
    """
    Remove stop-words using nltk stopwords library.

    :param data:        Text containing stop-words
                        Type: pandas Dataframe
    :param lowercase:   If 'True', then check for stop-words in lowercase
                        Type: Boolean
    :return:            Text without stop-words
                        Type: pandas Dataframe
    """
    stop_words = set(stopwords.words("english"))
    for index, value in data.items():
        data[index] = removeStopWordsFromString(value, stop_words, lowercase)
    return data


def textPreprocessing(data, lowercase, punctuation, stemming, stop_words):
    """
    Use this function to preprocess data-text. The supported functions are the conversion
    of the text to lowercase, the removal of the punctuation, the application of stemming
    and the removal of stop_words.

    :param data:        The data-text for preprocessing
                        type: pandas Dataframe
    :param lowercase:   If True, then convert the data-text to lowercase
                        type: boolean
    :param punctuation: if True, then remove punctuation from data-text
                        type: boolean
    :param stemming:    if True, then remove suffixes from data-text
                        type: boolean
    :param stop_words:  if True, then remove stop-words from data-text
                        type: boolean
    :return:            The preprocessed data-text
                        type: pandas Dataframe
    """
    if lowercase:
        data = data.str.lower()
    if punctuation:
        data = removePunctuation(data)
    if stemming:
        data = applyStemming(data)
    if stop_words:
        data = removeStopwords(data, lowercase)
    return data


def preprocessText(
    dataset, lowercase, punctuation, stemming, stop_words, descr_ids, path
):
    """
    Use this function to automate the pre-processing of the datasets: articles text
    preprocessing and store file with results. Τhe pre-processing is possible to include
    text conversion to lowercase, punctuation removal from the text, stemming to the words
    of the text and stop-words removal.

    :param dataset:     The requested dataset (columns: 'text', 'pmid', 'valid_labels'
                        and list with [descr_ids])
                        type: pandas DataFrame
    :param lowercase:   If "True", the text will be converted to lowercase.
                        type: Boolean
    :param punctuation: If "True", the punctuation will be removed from the text.
                        type: Boolean
    :param stemming:    If "True", stemming will be applied to the words of the text.
                        type: Boolean
    :param stop_words:  If "True", the stop_words will be removed from the text.
                        type: Boolean
    :param descr_ids:   The descriptor IDs
                        type: list
    :param path:        The path-directory, where the pre-processed file will be stored.
                        type: String
    :return:            The pre-precessed dataset (columns: 'text', 'pmid', 'valid_labels'
                        and list with [descr_ids])
                        type: pandas DataFrame

    """
    preprocessed = textPreprocessing(
        dataset["text"], lowercase, punctuation, stemming, stop_words
    )
    new_dataset = pd.DataFrame(
        {
            "pmid": dataset["pmid"].values.tolist(),
            "text": preprocessed.values.tolist(),
            "valid_labels": dataset["valid_labels"].values.tolist(),
        }
    ).join(dataset[descr_ids].reset_index())
    new_dataset.to_csv(path, index=False)
    return new_dataset


#################################################
#  General Functions for dataset preprocessing  #
#################################################


def applyBalancing(dataset, descr_ids, settings, year):
    """
    An abstract function to apply balancing on dataset.

    :param dataset:     The train dataset for the balancing. The dataset will only be used
                        if the corresponding preprocessed file is not found stored.
                        type: pandas DataFrame
    :param descr_ids:   List with the descriptor IDs (target labels)
                        type: string list
    :param settings:    The settings of the algorithm.
                        type: yaml
    :param year:        The year of the dataset.
                        type: integer
    :return:            The balanced train dataset.
                        type: pandas DataFrame
    """
    print(printTimestamp(), " -- Balance train dataset")
    workingPath = f"{settings['workingPath']}/snorkel_results_{year}_0_2_6"
    blns_filename = f"{workingPath}/train_{year}_balanced_{settings['useAsTrainDataset']}.csv"
    if exists(blns_filename):
        print(printTimestamp(), " -- Load balanced train dataset")
        return pd.read_csv(blns_filename)
    else:
        return balance_dataset(
            dataset, descr_ids, settings["balanceRatio"], blns_filename
        )


def applyTextPreprocessing(train_dataset, test_dataset, descr_ids, year, settings):
    """
    An abstract function to apply text preprocessing on dataset.

    :param train_dataset:   The train dataset for the text preprocessing. The dataset will
                            only be used if the corresponding preprocessed file is not
                            found stored.
                            type: pandas DataFrame
    :param test_dataset:    The test dataset for the text preprocessing. The dataset will
                            only be used if the corresponding preprocessed file is not
                            found stored.
                            type: pandas DataFrame
    :param descr_ids:       List with the descriptor IDs (target labels)
                            type: string list
    :param year:            The year of the dataset.
                            type: integer
    :param settings:        The settings of the algorithm.
                            type: yaml
    :return:                The train and test dataset with the preprossesed text.
                            type: pandas DataFrame
    """
    print(printTimestamp(), " -- Preprocess datasets")
    train_preprocessed = pd.DataFrame()
    test_preprocessed = pd.DataFrame()
    conditions = (
        f'{str(settings["lowercase"])[0]}{str(settings["punctuation"])[0]}'
        f'{str(settings["stemming"])[0]}{str(settings["stop_words"])[0]}'
    )
    workingPath = f"{settings['workingPath']}/snorkel_results_{year}_0_2_6"
    pr_train = f'{workingPath}/train_preprocessed_{year}_{conditions}_{settings["useAsTrainDataset"]}.csv'
    pr_test = f'{workingPath}/test_preprocessed_{year}_{conditions}_{settings["useAsTrainDataset"]}.csv'
    if exists(pr_train):
        print(printTimestamp(), " -- Load preprocessed train data")
        train_preprocessed = pd.read_csv(pr_train)
    else:
        print(printTimestamp(), " -- Preprocessing train dataset ...")
        train_preprocessed = preprocessText(
            train_dataset,
            settings["lowercase"],
            settings["punctuation"],
            settings["stemming"],
            settings["stop_words"],
            descr_ids,
            pr_train,
        )
    if exists(pr_test):
        print(printTimestamp(), " -- Load preprocessed test data")
        test_preprocessed = pd.read_csv(pr_test)
    else:
        print(printTimestamp(), " -- Preprocessing test dataset ...")
        test_preprocessed = preprocessText(
            test_dataset,
            settings["lowercase"],
            settings["punctuation"],
            settings["stemming"],
            settings["stop_words"],
            descr_ids,
            pr_test,
        )
    return train_preprocessed, test_preprocessed


#################################################
#       Functions for feature extraction        #
#################################################


def tokenizeText(text, vocabulary=None):
    """
    Tokenize articles text; creates the document-term matrix.

    :param text:        DataFrame containing the article's text
                        type: pandas DataFrame
    :param vocabulary:  A specific vocabulary based on which it will be performed
                        the tokenization.
                        type:
    :return:            The document-term matrix
                        type: pandas Dataframe
                        The vocabulary of articles
                        type: List
    """
    if vocabulary is not None:
        vectorizer = CountVectorizer(vocabulary=vocabulary)
    else:
        vectorizer = CountVectorizer()
    doc_term_mat = vectorizer.fit_transform(text)

    vocabulary = {i: v for v, i in vectorizer.vocabulary_.items()}
    words = []
    for i in range(len(vocabulary)):
        words.insert(i, vocabulary.get(i))

    return doc_term_mat, words


def calc_tfidf(doc_term_mat):
    """
    Calculate the tf-idf of the article's text, given the tokenized text
    (the document-term matrix).

    :param doc_term_mat:    The document-term matrix
                            type: pandas Dataframe
    :return:                The tf-idf matrix in csr
                            type: scipy.sparse
    """
    transformer = TfidfTransformer()
    return transformer.fit_transform(doc_term_mat)


def applyFeatureExtraction(text, vocabulary):
    """
    Use this function to apply feature extraction at once. The feature extraction includes
    the tokenization of the text and the calculation of the tf-idf matrix.

    :param text:        DataFrame containing the article's text
                        type: pandas DataFrame
    :param vocabulary:  A specific vocabulary based on which it will be performed
                        the tokenization.
                        type:
    :return:    tf_idf_matrix:  The tf-idf matrix in csr
                                type: scipy.sparse
                words:          The vocabulary of articles
                                type: List
    """
    print(printTimestamp(), " -- Applying feature extraction on dataset")
    doc_term_mat, words = tokenizeText(text, vocabulary)
    tf_idf_matrix = calc_tfidf(doc_term_mat)
    return tf_idf_matrix, words


#################################################
#       Functions for feature selection         #
#################################################


def getNumOfFeatures(data, descr_ids, k):
    """
    Calculate the number of features for each descriptor as a percentage, based on k.
    The k/5 of the features are shared between the descriptors.
    The rest 4k/5 are distributed to the descriptors depending on the number of articles.

    :param data:        Document-descriptor dataframe containing 0/1
                        type: DataFrame
    :param descr_ids:   The descriptor IDs
                        type: list
    :param k:           The number of features
                        type: integer
    :return:            Dictionary: Descr_id -> num of features for the descr.
    """
    non_zero = (data[descr_ids] != 0).sum().to_dict()
    summary = sum(non_zero.values())
    for i in non_zero:
        non_zero[i] = ceil(
            k * 0.2 / len(descr_ids) + non_zero.get(i) * k * 0.8 / summary
        )

    diff = sum(non_zero.values()) - k
    if diff > 0:
        sorted_non_zero = dict(
            sorted(non_zero.items(), key=operator.itemgetter(1), reverse=True)
        )
        for key in sorted_non_zero:
            sorted_non_zero[key] -= 1
            diff -= 1
            if diff == 0:
                break
        non_zero = sorted_non_zero

    elif diff < 0:
        sorted_non_zero = dict(sorted(non_zero.items(), key=operator.itemgetter(1)))
        for key in sorted_non_zero:
            sorted_non_zero[key] += 1
            diff += 1
            if diff == 0:
                break
        non_zero = sorted_non_zero
    return non_zero


def selectFeatures(tfidf, y, k, old_mask, diff, times):
    """
    Get the top k features (words-terms) of the articles for a specific descriptor.

    :param tfidf:   The tf-idf (csr) matrix
                    type: scipy.sparse
    :param y:       The results of a descriptor in articles (1-D DataFrame containing 0/1)
                    type: pandas Dataframe
    :param k:       The number of features to receive
                    type: integer
    :param old_mask:A boolean list used as mask
                    type: (boolean) list
    :param diff:    Number indicating the additional number of features-tokens to complete
                    the actual number of k features
                    type: integer
    :param times:   It shows how many times the function has been used as retroactive for
                    a particular descriptor.
                    type: integer
    :return:        selected_data:  The tf-idf (csr) matrix with only the k features
                    type: scipy.sparse
                    mask:           boolean list for the features in the words-tokens
                    type: List
    """
    selector = SelectKBest(score_func=chi2, k=k + diff)
    selected_data = selector.fit_transform(tfidf, y)
    mask = selector.get_support()
    # check if any features have already been used
    new_diff = sum(old_mask) + k - sum(updateMask(old_mask, mask))
    if new_diff > 0 and times <= 10:
        times += 1
        selected_data, mask = selectFeatures(
            tfidf, y, k, old_mask, diff + new_diff, times
        )
    return selected_data, mask


def updateMask(first_mask, second_mask):
    """
    Merge 2 boolean lists-masks according to the logic of the function 'OR'.

    :param first_mask:  The first boolean list used as mask.
                        type: list (boolean)
    :param second_mask: The second boolean list used as mask.
                        type: list (boolean)
    :return:            The merged mask.
                        type: list (boolean)
    """
    new_mask = []
    for i in range(len(first_mask)):
        if first_mask[i] != second_mask[i]:
            new_mask.append(True)
        else:
            new_mask.append(first_mask[i])
    return new_mask


def applyFeatureSelection(x, tfidf_x, vocabulary, descriptor_ids, num_of_features):
    """
    Apply feature selection on the datasets x, y (train and test).

    :param x:               The desired dataset without modifications.
                            type: pandas DataFrame
    :param tfidf_x:         The tf-idf matrix of x.
                            type: scipy.sparse (csr)
    :param vocabulary:      A specific vocabulary based on which it will be performed
                            the tokenization.
                            type:
    :param descriptor_ids:  List with the descriptor ids.
                            type: list of strings
    :param num_of_features: The number of feature for the feature selection.
                            type: integer
    :return:    tfidf_x_transformed:    The processed text of x dataset in tf-idf form;
                                        the tfidf_x with only the top features resulting
                                        from feature selection.
                                        type: scipy.sparse (csr)
                shared_vocabulary:      The top tokens on which the feature selection
                                        was performed.
    """
    print(printTimestamp(), " -- Applying feature selection on train dataset")
    percentage_num_features = getNumOfFeatures(
        x[descriptor_ids], descriptor_ids, num_of_features
    )
    mask = [False] * len(vocabulary)
    for descr in descriptor_ids:
        X_new, feature_mask = selectFeatures(
            tfidf_x, x[descr], percentage_num_features[descr], mask, 0, 0
        )
        mask = updateMask(mask, feature_mask)

    shared_vocabulary = list(compress(vocabulary, mask))
    tfidf_x_transformed = tfidf_x[:, getPositionOfInstances(mask)]

    return tfidf_x_transformed, shared_vocabulary


#################################################################
#  Functions for Cross Validation focused on feature selection  #
#################################################################


def splitDataset(dataset, folds, fold, indexes):
    """
    It splits one dataset into 2 parts: The first is the train dataset and the second is
    the test dataset.
    We consider the train/test ratio is proportional to the folds
    (train/test=(folds-1)/folds).

    For example: for folds=5 the train dataset will have the 4/5 of the dataset
                         and the test dataset will have the 1/5 of the dataset.

    :param dataset: The desired dataset for splitting.
                    Type: pandas DataFrame
    :param folds:   The numbers of the folds in cross validation.
                    Type: integer
    :param fold:    The current fold.
                    Type: integer
    :param indexes: A list with shuffle integer numbers in the range of the length of the
                    dataset, in order to increase randomness.
                    Type: list of integers
    :return:    train:  The train dataset
                        type: pandas DataFrame
                test:   The test Dataset
                        type: pandas DataFrame
    """
    fold_range = round(len(dataset) / folds)
    if fold == folds - 1:
        last = len(dataset)
    else:
        last = (fold + 1) * fold_range
    train_indexes = indexes[0: fold * fold_range] + indexes[last:]
    test_indexes = indexes[fold * fold_range: last]

    train = dataset.iloc[train_indexes]
    test = dataset.iloc[test_indexes]
    return train, test


def mergeCVResults(path, folds, num_of_features):
    """
    Read results files from cross validation and create one file containing total f1 micro
    and macro score of each fold and the average of the f1 micro and macro scores of
    all folds.

    :param folds:           The number of folds in the cross validation.
                            Type: integer
    :param path:            The path in witch the files are stored.
                            Type: string
    :param num_of_features: The number of features used during the feature selection.
                            Type: integer
    :return:                f1 micro and macro score of each fold and the average of them
                            Type: pandas DataFrame
    """
    report = {}
    for fold in range(folds):
        k_report = pd.read_csv(
            f"{path}/report_{num_of_features}_{fold}.csv", index_col="label"
        )
        k_dict = k_report.to_dict()
        report[fold] = {
            "micro avg": k_dict["f1-score"].get("micro avg"),
            "macro avg": k_dict["f1-score"].get("macro avg"),
        }

    results = pd.DataFrame.from_dict(report, orient="index")
    results.loc["mean"] = [results["micro avg"].mean(), results["macro avg"].mean()]

    results.to_csv(f"{path}/report_{num_of_features}.csv", index=True)
    return results


def crossValidation(
    dataset, descriptor_ids, folds, path, num_of_features, indexes, classifier
):
    """
    It splits one dataset into 2 parts and applies cross validation on them, using as
    classifier the OneVsRestClassifier(LogisticRegression()), saving the results.

    :param dataset:         The desired dataset.
                            type: pandas DataFrame
    :param descriptor_ids:  List containing the descriptor IDs.
                            type: list (of strings)
    :param folds:           The number of folds for the cross validation.
                            type: int
    :param path:            The path to save the results.
                            type: string
    :param num_of_features: The number of token to keep during feature selection.
                            type: int
    :param indexes:         List with shuffle integer numbers in the range of the length
                            of the dataset.
                            Type: list of integers
    :param classifier:
    :return:                The mean f1 macro score between the folds.
    """
    for fold in range(folds):
        f_train, f_test = splitDataset(dataset, folds, fold, indexes)
        tfidf_train, vocabulary = applyFeatureExtraction(f_train["text"], None)
        tfidf_train_featured, shared_vocabulary = applyFeatureSelection(
            f_train, tfidf_train, vocabulary, descriptor_ids, num_of_features
        )
        tfidf_test_featured, v = applyFeatureExtraction(
            f_test["text"], vocabulary=shared_vocabulary
        )

        y_train = (f_train[descriptor_ids]).values.tolist()
        y_test = (f_test[descriptor_ids]).values.tolist()

        classifier.fit(tfidf_train_featured, y_train)
        y_pred_test = classifier.predict(tfidf_test_featured)

        k_report = classification_report(y_test, y_pred_test, output_dict=True)
        save_evaluation_report(
            k_report,
            descriptor_ids,
            None,
            f"{path}/report_{num_of_features}_{fold}.csv",
        )
    scores = mergeCVResults(path, folds, num_of_features)
    return scores["macro avg"]["mean"]


def findBestNumOfFeatures(
    dataset, descriptor_ids, folds, path, max_features, classifier
):
    """
    It finds the optimal number of features. Τhe value search range is between the number
    of the descriptors (as base) and a maximum value, given as parameter. The search is
    based on binary search and as a criterion for adjusting the median is the f1 macro
    score of edge (low/high) values, which are obtained by applying cross validation on
    the given dataset.

    :param dataset:         The desired dataset.
                            type: pandas DataFrame
    :param descriptor_ids:  List containing the descriptor IDs.
                            type: list (of strings)
    :param folds:           The number of folds for the cross validation.
                            type: int
    :param path:            The path to save the results.
                            type: string
    :param max_features:    The maximum number of token to keep during feature selection.
                            type: int
    :param classifier
    :return:
    """
    new_path = createDir(path, "feature_selection")
    indexes = [i for i in range(len(dataset))]
    shuffle(indexes)

    low = len(descriptor_ids)
    high = max_features
    number_of_features = [low, high]
    f1_macro_low = crossValidation(
        dataset, descriptor_ids, folds, new_path, low, indexes, classifier
    )
    f1_macro_high = crossValidation(
        dataset, descriptor_ids, folds, new_path, high, indexes, classifier
    )

    while low <= high:
        mid = (high + low) // 2
        print("\tnew mid:", mid)
        if f1_macro_low < f1_macro_high:
            low = mid + 1
            f1_macro_low = crossValidation(
                dataset, descriptor_ids, folds, new_path, low, indexes, classifier
            )
            number_of_features.append(low)
        elif f1_macro_low > f1_macro_high:
            high = mid - 1
            f1_macro_high = crossValidation(
                dataset, descriptor_ids, folds, new_path, high, indexes, classifier
            )
            number_of_features.append(high)
        else:
            break
    number_of_features.sort()
    mergeReports(number_of_features, new_path)
    file = open(f"{new_path}/f_val_f1_score.txt", "w+")
    file.write(f"Number of features = {low}\nf1 macro score = {f1_macro_low}")
    file.close()
    return low


def calcPredictionStats(
    predictions, y_test, descr_ids, settings, year, num_of_features, model
):
    """
    Calculate the model performance given the model predictions and store results.
    Metrics like f1-macro score.
    :param predictions:     The model predictions.
                            type:
    :param y_test:          The test data.
                            type: list
    :param descr_ids:       List containing the descriptor IDs.
                            type: list (of strings)
    :param settings:        The settings of the algorithm.
                            type: yaml
    :param year:            The working year/ the year of the datasets.
                            type: integer
    :param num_of_features: The number of features used (for text).
                            type: integer
    :param model:           A code name for the model used:
                            'LR' for Logistic Regression
                            'XG' for XGBoost
                            type: string
    :return:                -
    """
    weakly_test = pd.read_csv(
        f'{settings["filesPath"]}/Dataset_SI_old_{year}/weakly_labelled/test_{year}.csv'
    )
    y_pred_test_df = pd.DataFrame(
        predictions, index=weakly_test["pmid"], columns=descr_ids
    )
    y_pred_test_df.to_csv(f"{settings['workingPath']}/snorkel_results_{year}_0_2_6/reports_{settings['useAsTrainDataset']}/{model}_prediction_{num_of_features}.csv")
    y_pred_test_filtered = filtering(
        y_pred_test_df[descr_ids].to_numpy(), weakly_test, descr_ids
    )
    report = classification_report(y_test, y_pred_test_filtered, output_dict=True)
    print(report["macro avg"].get("f1-score"))
    save_evaluation_report(
        report,
        descr_ids,
        None,
        f"{settings['workingPath']}/snorkel_results_{year}_0_2_6/reports_{settings['useAsTrainDataset']}/{model}_report_{num_of_features}.csv",
    )
