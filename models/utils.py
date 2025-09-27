import pandas as pd
import numpy as np


def readFiles(year, path):
    """
    Given the year, it returns the train and the test datasets and the descriptor's ids
    as pandas DataFrame.

    :param year:    The desired year
                    type: integer
    :param path:    The path where the datasets and the file with the descriptors
                    are stored.
                    type: string
    :return:        The train dataset, the test dataset and the descriptor ids
                    type: DataFrame
    """
    train = pd.DataFrame()
    test = pd.DataFrame()
    descr_ids = pd.DataFrame()
    weakly_test = pd.DataFrame()
    try:
        train = pd.read_csv(path + f"/train_{year}.csv")
        test = pd.read_csv(path + f"/test_{year}.csv")
        weakly_test = pd.read_csv(path + "/weakly_labelled/test_" + str(year) + ".csv")
        descr_ids = pd.read_csv(path + f"/UseCasesSelected_{year}.csv")[
            "Descr. UI"
        ].values.tolist()
    except Exception as e:
        print(e)
    return train, test, weakly_test, descr_ids


def completeDataFrame(df_to_check, full_df, descr_ids):
    """
    Due to large file size, not all information, like "text" and "valid_labels" are stored
    to all the files.
    Use this function to complete these information from a full dataset.

    :param df_to_check: The dataset with the missing data.
                        type: pandas DataFrame
    :param full_df:     The dataset with the full data.
                        type: pandas DataFrame
    :param descr_ids:   List with descriptor IDs (target_labels).
                        type: (string) List
    :return:            The complete df_to_check dataset.
                        type: pandas DataFrame
    """
    return pd.merge(
        full_df[full_df["pmid"].isin(df_to_check["pmid"])][
            ["pmid", "text", "valid_labels"]
        ],
        df_to_check,
        on="pmid",
    )


def getPositionOfInstances(mask):
    """
    Given a boolean (1-D) list, it returns an integer list,
    containing the positions of the 'True' values.
    :param mask:    boolean list
    :return:        integer list
    """
    positions = []
    for i in range(len(mask)):
        if mask[i]:
            positions.append(i)
    return np.array(positions)


def mergeReports(k_values, path):
    """

    :param k_values:
    :param path:
    :return:
    """
    report = {}
    for k in k_values:
        k_report = pd.read_csv(f"{path}/report_{k}.csv", index_col=0)
        k_dict = k_report.to_dict()
        report[k] = {
            "micro avg": k_dict["micro avg"].get("mean"),
            "macro avg": k_dict["macro avg"].get("mean"),
        }

    d = pd.DataFrame.from_dict(report, orient="index")
    d.to_csv(f"{path}/report.csv", index=True)


def checkMultilabel(y):
    """
    Count the number of articles annotated at least with 2 descriptors in the
    articles-descriptors matrix.

    :param y:   The articles-descriptors matrix
                type:
    :return:    The number of articles annotated at least with 2 descriptors
                type: integer
    """
    S = np.sum(y, axis=1)
    ml = 0
    for n in S:
        if n >= 2:
            ml += 1
    return ml
