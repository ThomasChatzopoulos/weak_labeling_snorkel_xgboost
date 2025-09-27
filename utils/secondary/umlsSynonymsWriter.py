import json
import mysql.connector
import pandas as pd
import yaml


def loadDescriptors(path, year):
    """
    Use this function to load the descriptors of a year.
    :param path:    The path where the datasets are stored.
    :param year:    The desired year.
    :return:        Pandas Dataframes with train and test datasets.
    """
    UCSelected = pd.DataFrame()
    try:
        UCSelected = pd.read_csv(path + '/UseCasesSelected_' + str(year) + '.csv')
    except Exception as e:
        print(e)

    return UCSelected["Descr. UI"].values.tolist()


def connectToServer():
    """
    Connect to umls server.
    :return: The umls connection
    """
    umls_settings = settings["umls"]
    config = {
        'user': umls_settings["dbuser"],
        'password': umls_settings["dbpass"],
        'host': "192.168.11.135",
        'database': "umls_mesh2019",
    }
    try:
        conn = mysql.connector.connect(**config)
    except mysql.connector.Error as e:
        print(e)

    return conn


def convertResults(results):
    """
    The contents contain a mixture of list and tuples.
    Convert the results to a sting list.
    :param results: list of tuple of strings
    :return:        list of string
    """
    synomyms = []
    for r in results:
        for syn in r:
            synomyms.append(syn)
    return synomyms


def getDescrSynonyms(descriptor_id):
    """
    Get synonyms of descriptors from umls.
    :param descriptor_id:   List with descriptor IDs
    :param server_con:      The connection to the server where umls is stored
    :return:                Dicrionary: descriptor_id -> list of synonyms
    """
    descriptor_synonyms = {}
    conn = connectToServer()
    cursor = conn.cursor()
    query = "select STR from MRCONSO where SDUI=\""+descriptor_id+"\""
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    descriptor_synonyms[descriptor_id] = convertResults(results)
    return descriptor_synonyms


def writeJSONFile(path, data):
    try:
        with open(path + '/synonyms.json', 'w') as fp:
            fp.write(
                '{"documents":[\n' +
                ',\n'.join(json.dumps(i) for i in data) +
                ']}\n')
    except Exception as e:
        print(e)


# The main algorithm
# Read the settings
settings_file = open("../settings.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

synonyms_list = []
for year in range(settings["firstYear"], settings["lastYear"] + 1):
    # load data
    descriptor_ids = loadDescriptors(settings["filesPath"], year)

    for d in descriptor_ids:
        synonyms_list.append(getDescrSynonyms(d))

writeJSONFile(settings["workingPath"], synonyms_list)
