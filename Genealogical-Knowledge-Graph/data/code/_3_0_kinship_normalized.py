import re
import nltk
import spacy
from tqdm import tqdm


nlp = spacy.load("en_core_web_sm")

normalized_list = [["spouse","married to"],
["great-grandchild","great grand-child"],
["great-grandchild-in-law","great grandchild-in-law"],
["great-granddaughter","great granddaughter"],
["great-grandson","great grandson"],
["great-grandnephew","great grand-nephew"],
["great-grandniece","great grand-niece"],
["great-great-grandchild","great-great grand-child"],
["great-great-granddaughter","great-great granddaughter"],
["great-great-grandson","great-great grandson"],
["parent","born to","son of","daughter of"],
["granduncle","grand uncle"],
["grandaunt","grand aunt"],
["grandchild","grand child"],
['grandchild-in-law',  'grand child-in-law']
]


def count_kinsip_num():
    with open("../ontology.csv", "r") as f:
        raw_data = f.readlines()
    kinship_list = []
    for i in raw_data:
        if len(i) > 1:
            items = i[:-1].split(":")
            for j in items:
                a = re.sub("[[\]]", "", j).split(",")
                for k in a:
                    kinship_list.append(k)
    print(len(kinship_list))
    print(kinship_list)

def seg_token(sentence):
    # token_list = nltk.word_tokenize(sentence)
    token_list = [str(i) for i in nlp(sentence)]
    return token_list

def normalized_kinship():
    replace_list = []
    dic_replace = {}
    for i in normalized_list:
        for j in i:
            replace_list.append(j)
        for key in i:
            dic_replace.update({key:i[0]})
    # print(len(replace_list))
    # print(len(dic_replace))
    # print(dic_replace)
    # print(replace_list)

    with open("../sentence_level_corpus_all_information.csv", "r") as f:
        raw_data = f.readlines()
    print("raw_data len : ", len(raw_data))

    new_list = []
    for i in tqdm(raw_data[1:]):
        items = i.split("|")
        temp_str = items[-1]
        if len(items[-1]) > 1:
            for kinship_pair in eval(items[-1]):
                if kinship_pair[0] in replace_list:
                    temp_str = re.sub(kinship_pair[0], dic_replace[kinship_pair[0]], temp_str)

            new_items = str(items[0]) + "|" + str(items[1]) + "|" + str(items[2]) + "|" + str(items[3]) + "|" + \
                                       str(items[4]) + "|" + str(items[5]) + "|" + str(items[6]) + "|" + str(temp_str)
            new_list.append(new_items)
        else:
            new_list.append(i)

    print("new_data len : ", len(new_list))

    with open("../sentence_level_corpus_all_information_normalized.csv", "w") as f:
        f.writelines("Obituary_ID|Sentence|Death_Name|Death_Date|Birth_Date|Age_ID|Location_ID|relationship\n")
        for i in new_list:
            f.writelines(i)

normalized_kinship()
