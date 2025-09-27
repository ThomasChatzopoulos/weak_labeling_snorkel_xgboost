import os
import time
from random import shuffle, randint
from datetime import datetime
import pandas as pd

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + "Start")
# Parameters-variables from settings file
workingPath = "/home/chatzothomas/projects/models"
filesPath = "/home/chatzothomas/projects/models/Dataset_SI_old_2006"
year = 2006

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"


def stats_on_instace_validity(input_data, target_labels):
    positive_instances = input_data[target_labels].sum()
    valid_instances = {}
    valid_negative_instances = {}
    for label in target_labels:
        valid_instances[label] = input_data['valid_labels'].str.contains(label).sum()
        valid_negative_instances[label] = valid_instances[label] - positive_instances[label]
    print("\tValid_instances per label :")
    print(valid_instances)
    print("\tValid_negative_instances per label :")
    print(valid_negative_instances)
    return valid_instances, positive_instances


def balance_dataset(input_data, target_labels, balance_n):
    # TODO: Could we use RandomUnderSampler(sampling_strategy= ... instead of this custom implementation?
    # remove unwanted samples (e.g. for undersampling)

    seed = randint(0, 99)

    positive_instances = input_data[target_labels].sum()

    remove_pmids = []
    if balance_n is not None:
        # balance the dataset removing fully negative articles
        input_data["label_count"] = input_data.loc[:, target_labels].sum(axis=1)
        all_negative_instances = input_data.loc[input_data['label_count'] == 0].copy()
        print("\tlen(input_data) :", str(len(input_data)))
        print("\tlen(All negative_instances) :", str(len(all_negative_instances)))
        # print("\tall_negative_instances :", all_negative_instances)

        all_positive_instances = len(input_data) - len(all_negative_instances)
        print("\tlen(All positive_instances) :", str(all_positive_instances))

        valid_instances = {}
        valid_negative_instances = {}
        for label in target_labels:
            valid_instances[label] = input_data['valid_labels'].str.contains(label).sum()
            valid_negative_instances[label] = valid_instances[label] - positive_instances[label]
        print("\tvalid_instances per label :")
        print(valid_instances)
        print("\tValid_negative_instances per label :")
        print(valid_negative_instances)

        # Condition to meet
        # For each label, valid negative instances should count at most balance_n times the positive ones
        condition_met = False

        initial_negative = len(all_negative_instances)
        counter = 0
        while not condition_met and not all_negative_instances.empty:
            # print progress every 100 steps
            counter = counter + 1
            if counter % 10000 == 0:
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), '\t',
                      str(round(counter / initial_negative, 3)), " % of ", str(initial_negative), " negative instances considered. ",
                      str(round(1.0 - (len(all_negative_instances) / initial_negative), 3)), " % negative reduction so far.")
            # Run until the condition is met, or no more unecesary instances are available to remove for meetting it
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
            valid_labels = random_negative_instance['valid_labels'].values[0].split(" ")  # All valid labels for this instance
            # keep only valid labels of focus. Valid labels for which we don't need to learn a model are not considered.
            valid_labels = set.intersection(set(valid_labels), set(target_labels))
            # print('valid_labels ', valid_labels)

            # valid_labels = random_negative_instance['valid_labels'].split(" ")
            for valid_label in valid_labels:
                if valid_negative_instances[valid_label] <= positive_instances[valid_label] * balance_n:
                    # This instance is necessary for this label. Do nothing.
                    necessary = True
            if not necessary:
                # This instance is not necessary for any label, it can be removed from the training dataset
                # remove a random negative
                remove_pmids.append(random_negative_instance['pmid'].values[0])
                # Update valid negative values to reflect new situation after the removal
                for valid_label in valid_labels:
                    valid_negative_instances[valid_label] = valid_negative_instances[valid_label] - 1
            # Check all labels to update the condition
            condition_met = True
            for label in target_labels:
                if valid_negative_instances[label] > positive_instances[label] * balance_n:
                    condition_met = False
            #   remove this instance from the dataframe as each instance is only checked once
            all_negative_instances.drop(random_index, inplace=True)

        print('\tremove_pmids (final) :', len(remove_pmids))
        # print('\tremove_pmids (final) :', remove_pmids)

    data = input_data[~input_data['pmid'].isin(remove_pmids)]
    stats_on_instace_validity(data, target_labels)
    return data


# load data
trainData = pd.read_csv(f'{filesPath}/train_{year}.csv')
# get information about the descriptors
descriptorIDs = pd.read_csv(f'{filesPath}/UseCasesSelected_{year}.csv')["Descr. UI"].values.tolist()

# trainData = pd.read_csv(f"{settings['workingPath']}/train_balanced.csv").iloc[:, :-1]
trainData = balance_dataset(trainData, descriptorIDs, 5)
trainData.to_csv(f"{workingPath}/train_{year}_balanced.csv", index=False)
