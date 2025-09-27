import csv
import os
import yaml
import json

import pandas as pd
import numpy as np

from datetime import datetime
from itertools import combinations
from shutil import copy
from time import time

from weak_labeling_snorkel_xgboost.snorkel_labeling.snorkel_model_functions import (
    loadData,
    getDescriptorsInfo,
    applyLabelModel,
    applyMajorityVoter,
    applyMinorityVoter,
    updatePredictions,
    selectKBestLFs,
    filtering,
    classification_report,
    saveVoterPredictions,
    calcStatsForVoters,
    calcStatsForLFs,
)
from weak_labeling_snorkel_xgboost.utils.utils import (
    createDir,
    loadResultsFromJSON,
    saveResultsToJSON,
    createSummaryReport,
    printTimestamp,
)


######################################
#                                    #
#     for (n k) combinations         #
#                                    #
######################################


def createLabelMatrixFromResults(lf_results, lfs_to_use, index):
    label_matrix = lf_results.get(lfs_to_use[0])[index]
    for lf in lfs_to_use[1:]:
        label_matrix = np.vstack((label_matrix, lf_results.get(lf)[index]))

    return np.transpose(label_matrix)


def getCombinationsOfLFs(lfs):
    """
    Get a list with combinations of LFs containing at least 3 LFs.
    (n k) = n!/(k!(n-k)!), where n -> lfs, k -> in range(3, len(lfs))
    :param lfs: List with the LFs.
                type: (string) List
    :return:    List containing lists with combinations
    """
    com = []
    for lf in range(3, len(lfs)):
        for combination in list(combinations(lfs, lf)):
            com.append(list(combination))
    return com


def getCombID(lfs_to_use, lfs_comb):
    """
    Return the positions of the LFs of the lfs_comb in the lfs_to_use joined with '_'.
    Using this function you create a unique id for each lf combination.
    e.g.    lfs_to_use = ["occurrence_label", "exact", "synonyms"]
            lfs_comb = ["occurrence_label", "synonyms"]
            then CombID = "0_2"
    :param lfs_to_use:  List with all the label function names
                        type: (string) list
    :param lfs_comb:    List with the label function names of the combination
                        type: (string) list
    :return:            combination id
                        string
    """
    position = []
    for lf in lfs_comb:
        position.append(str(lfs_to_use.index(lf)))
    return "_".join(position)


def combRepoJSONtoCSV(path, year):
    with open(f"{path}/combination_report_{year}.json") as json_file:
        data = json.load(json_file)

    with open(
        f"{path}/combination_report_{year}.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        header_row = [
            "Label Functions",
            "LM_micro_f1",
            "LM_macro_f1",
            "Maj_micro_f1",
            "Maj_macro_f1",
            "Min_micro_f1",
            "Min_macro_f1",
        ]
        writer.writerow(header_row)
        for combination in data:
            # write the label functions
            comb_dict = data.get(combination)
            content_row = [comb_dict.get(next(iter(comb_dict))).get("combination")]
            for voter in comb_dict:
                # write fi micro/macro scores
                content_row = content_row + [
                    comb_dict.get(voter).get("micro_f1"),
                    comb_dict.get(voter).get("macro_f1"),
                ]
            writer.writerow(content_row)


combRepoJSONtoCSV("C:/Users/files", 2006)

settings_file = open("C:/Users/files/settings.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

# Parameters
workingPath = settings["workingPath"]  # The path where the results will be stored
filesPath = (
    f'{settings["filesPath"]}'  # The path where the files, like datasets, are stored
)
firstYear = settings["firstYear"]  # The first year to consider in the analysis
lastYear = settings["lastYear"]  # The last year to consider in the analysis
LFsToUse = settings["LFsToUse"]  # A list with the LFs you want to apply

os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings[
    "TF_CPP_MIN_LOG_LEVEL"
]  # Turn off TensorFlow logging messages
os.environ["PYTHONHASHSEED"] = settings["PYTHONHASHSEED"]  # For reproducibility

newWorkingPath = (
    workingPath
    + f'/working_with_results_{datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")}/'
)
os.mkdir(newWorkingPath)



for year in range(firstYear, lastYear + 1):
    lf_train_results = loadResultsFromJSON(workingPath, "lf_train_results", year)
    lf_test_results = loadResultsFromJSON(workingPath, "lf_test_results", year)

    # load data
    trainData, testData, weaklyTestData = loadData(filesPath, year)
    # get information about the descriptors
    descriptorIDs, descriptorNames, descriptorSynonyms = getDescriptorsInfo(
        filesPath, year
    )

    file = open(f"{newWorkingPath}/combination_report_{year}.json", "a")
    file.write("{")
    file.close()

    combs = getCombinationsOfLFs(LFsToUse)
    # pos = combs.index(['concept_occurrence_label',
    #                    'name_exact',
    #                    'name_exact_lowercase',
    #                    'name_exact_lowercase_tokens_no_punc',
    #                    'synonyms_exact',
    #                    'synonyms_lowercase',
    #                    'synonyms_lowercase_tokens_no_punc',
    #                    'tokens_lowercase_no_punc'])
    # for LFsComb in combs[pos:]:
    for LFsComb in combs:
        combID = getCombID(LFsToUse, LFsComb)
        print(
            printTimestamp() + f" ----> combination #{combs.index(LFsComb)} - {combID}"
        )
        voter_results = {}
        for d in descriptorIDs:
            print(printTimestamp() + " --> Descriptor: " + d)
            # The Label Functions to be used
            L_train = createLabelMatrixFromResults(
                lf_train_results, LFsComb, descriptorIDs.index(d)
            )
            L_test = createLabelMatrixFromResults(
                lf_test_results, LFsComb, descriptorIDs.index(d)
            )
            # Apply Voters
            X_pred_lm, Y_pred_lm = applyLabelModel(L_train, L_test, False)
            X_pred_maj, Y_pred_maj = applyMajorityVoter(L_train, L_test)
            X_pred_min, Y_pred_min = applyMinorityVoter(L_train, L_test)
            voter_results = updatePredictions(
                voter_results, Y_pred_lm, Y_pred_maj, Y_pred_min
            )

        # calculate statistics for voters and LFs and save results
        Y_golden = (testData[descriptorIDs]).values.tolist()

        voter_report = {}
        for voter in voter_results:
            metrics = {}
            print(printTimestamp() + f" --> Statistics for {voter}")
            filtered_data = filtering(
                np.transpose(np.array(voter_results.get(voter))).tolist(),
                weaklyTestData,
                descriptorIDs,
            )
            report = classification_report(Y_golden, filtered_data, output_dict=True)
            metrics["combination"] = LFsComb
            metrics["micro_f1"] = report.get("micro avg").get("f1-score")
            metrics["macro_f1"] = report.get("macro avg").get("f1-score")
            voter_report[f"{voter}_{combID}"] = metrics

        print(voter_report)

        file = open(f"{newWorkingPath}/combination_report_{year}.json", "a")
        file.write(f'"{combID}": ')
        json.dump(voter_report, file)
        file.write(",\n")
        file.close()
    file = open(f"{newWorkingPath}/combination_report_{year}.json", "rb+")
    file.seek(-1, os.SEEK_END)
    file.truncate()
    file.close()
    file = open(f"{newWorkingPath}/combination_report_{year}.json", "a")
    file.write("}")
    file.close()


######################################
#                                    #
#     for specific combination       #
#                                    #
######################################


# settings_file = open("../settings.yaml")
settings_file = open("../../settings.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

# Parameters
workingPath = settings["workingPath"]  # The path where the results will be stored
filesPath = settings["filesPath"]  # The path where the files, like datasets, are stored
firstYear = settings["firstYear"]  # The first year to consider in the analysis
lastYear = settings["lastYear"]  # The last year to consider in the analysis
LFsToUse = settings["LFsToUse"]  # A list with the LFs you want to apply

os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings[
    "TF_CPP_MIN_LOG_LEVEL"
]  # Turn off TensorFlow logging messages
os.environ["PYTHONHASHSEED"] = settings["PYTHONHASHSEED"]  # For reproducibility

newWorkingPath = createDir(settings["workingPath"], "snorkel_results_0_2_6")
copy("../../settings.yaml", newWorkingPath)


for year in range(firstYear, lastYear + 1):
    X_voter_results = {}
    Y_voter_results = {}
    selected_lfs = [0] * len(LFsToUse)

    lf_train_results = loadResultsFromJSON(workingPath, "lf_train_results", year)
    lf_test_results = loadResultsFromJSON(workingPath, "lf_test_results", year)

    # load data
    print("Loading data")
    trainData, testData, weaklyTestData = loadData(filesPath, year)
    # get information about the descriptors
    print("get information about the descriptors")
    descriptorIDs, descriptorNames, descriptorSynonyms = getDescriptorsInfo(
        filesPath, year
    )
    col_names = ["lf", "j", "Polarity", "Coverage", "Overlaps", "Conflicts"]
    lfs_train_stat = pd.DataFrame(
        index=range(len(len(LFsToUse) * descriptorIDs)), columns=col_names
    )
    lfs_test_stat = pd.DataFrame(
        index=range(len(len(LFsToUse) * descriptorIDs)), columns=col_names
    )

    voter_results = {}
    for d in descriptorIDs:
        print(printTimestamp() + " --> Descriptor: " + d)
        # The Label Functions to be used
        L_train = createLabelMatrixFromResults(
            lf_train_results, LFsToUse, descriptorIDs.index(d)
        )
        L_test = createLabelMatrixFromResults(
            lf_test_results, LFsToUse, descriptorIDs.index(d)
        )
        # Apply Voters
        X_pred_lm, Y_pred_lm = applyLabelModel(L_train, L_test, False)
        X_pred_maj, Y_pred_maj = applyMajorityVoter(L_train, L_test)
        X_pred_min, Y_pred_min = applyMinorityVoter(L_train, L_test)
        X_voter_results = updatePredictions(
            X_voter_results, X_pred_lm, X_pred_maj, X_pred_min
        )
        Y_voter_results = updatePredictions(
            Y_voter_results, Y_pred_lm, Y_pred_maj, Y_pred_min
        )

        # Data, selected_lfs = selectKBestLFs(L_test, (testData[d]).values.tolist(), 5, selected_lfs)

    # print(LFsToUse)
    # print(selected_lfs)

    saveVoterPredictions(
        X_voter_results, trainData, descriptorIDs, newWorkingPath, "train", year
    )
    saveVoterPredictions(
        Y_voter_results, testData, descriptorIDs, newWorkingPath, "test", year
    )

    saveResultsToJSON(X_voter_results, newWorkingPath, "voters_X_results", year)
    saveResultsToJSON(Y_voter_results, newWorkingPath, "voters_Y_results", year)

    saveResultsToJSON(lf_train_results, newWorkingPath, "lf_train_results", year)
    saveResultsToJSON(lf_test_results, newWorkingPath, "lf_test_results", year)

    # calculate statistics for voters and LFs and save results

    Y_golden = (testData[descriptorIDs]).values.tolist()
    calcStatsForVoters(
        Y_voter_results, Y_golden, weaklyTestData, descriptorIDs, newWorkingPath, year
    )
    calcStatsForLFs(
        LFsToUse,
        lf_test_results,
        Y_golden,
        weaklyTestData,
        descriptorIDs,
        newWorkingPath,
        year,
    )

    # save statistics for LFs like Polarity, Coverage, Overlaps, Conflicts
    lfs_train_stat.to_csv(newWorkingPath + "/lf_statistics_train.csv", index=False)
    lfs_test_stat.to_csv(newWorkingPath + "/lf_statistics_test.csv", index=False)

    # create a summary excel (.xlsx) file
    createSummaryReport(newWorkingPath, LFsToUse, year)
