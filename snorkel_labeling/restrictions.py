import pandas as pd


def mergeRestrictions(path, first_year, last_year):
    """
    Create statistics for the restrictions for descriptor selection.
    It requires all the "newDescriptorsFull_{year}.csv" files for the given range of years (first_year, last_year).
    It creates a csv file with the restriction levels as columns. The value of each column is the number of the
    descriptors that meet the conditions both the given constraint and the previous ones (the columns on the left).
    :param path:        Both the working path and the path where required files are stored.
                        type: string
    :param first_year:  The first year to consider in the analysis.
                        type: integer
    :param last_year:   The last year to consider in the analysis.
                        type: integer
    :return: -
    """
    # The restrictions that have been applied to descriptors.
    restrictions = [
        "0: No (old concept) subdivision",
        "2: Cs > 1",
        "3: not a leaf",
        "4: testGold < 10",
        "5: train <  10",
        "6: train > 1000000",
        "7: trainWS < 10",
        "OK",
    ]
    # More general categories of restrictions.
    restrictions_merged = [
        "Total",
        "0: No (old concept) subdivision",
        "2: Cs > 1",
        "3: not a leaf",
        "4: right article num",
    ]

    # CSV file creation with detailed information about the restrictions
    results = {}
    for year in range(first_year, last_year + 1):
        counts = (
            pd.DataFrame(
                pd.DataFrame(
                    pd.read_csv(path + f"/newDescriptorsFull_{year}.csv")[
                        "Rest. Level"
                    ].value_counts()
                ).T,
                columns=restrictions,
            )
            .fillna(0)
            .rename(index={"Rest. Level": year})
            .T
        )
        d = counts.astype("int").to_dict()
        results[year] = d[year]

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(f"{path}/restrictions.csv")

    merged_results = pd.DataFrame(
        index=restrictions_merged,
        columns=[f"#{i}" for i in range(first_year, last_year + 1)] + ["Total"],
    )

    # CSV file creation with merged information about the restrictions
    for year in range(2006, 2020):
        merged_results.at["Total", year] = results_df[year].sum()
        merged_results.at["0: No (old concept) subdivision", year] = (
            merged_results.at["Total", year]
            - results_df.at["0: No (old concept) subdivision", year]
        )
        merged_results.at["2: Cs > 1", year] = (
            merged_results.at["0: No (old concept) subdivision", year]
            - results_df.at["2: Cs > 1", year]
        )
        merged_results.at["3: not a leaf", year] = (
            merged_results.at["2: Cs > 1", year] - results_df.at["3: not a leaf", year]
        )
        merged_results.at["4: right article num", year] = (
            merged_results.at["3: not a leaf", year]
            - results_df.at["4: testGold < 10", year]
            - results_df.at["5: train <  10", year]
            - results_df.at["6: train > 1000000", year]
            - results_df.at["7: trainWS < 10", year]
        )

    merged_results = merged_results.T
    merged_results.at["Total", "Total"] = merged_results["Total"].sum()
    merged_results.at["Total", "0: No (old concept) subdivision"] = merged_results[
        "0: No (old concept) subdivision"
    ].sum()
    merged_results.at["Total", "2: Cs > 1"] = merged_results["2: Cs > 1"].sum()
    merged_results.at["Total", "3: not a leaf"] = merged_results["3: not a leaf"].sum()
    merged_results.at["Total", "4: right article num"] = merged_results[
        "4: right article num"
    ].sum()

    merged_results.to_csv(f"{path}/restrictions_merged.csv")


document_path = "/test/original_old_datasets_04_03_2022/statistics_and_more"
firstYear = 2006
lastYear = 2019
mergeRestrictions(document_path, firstYear, lastYear)
