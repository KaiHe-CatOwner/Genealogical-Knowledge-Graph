import spacy
import pandas as pd
import re
import os
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
global error
error = 0



# Death_Name | Death_Date | Birth_Date | Age_ID | Location_ID | relationship
# we need match annotated span with token-level tag
def generated_tagged_index(token_list, ann_item_span, sentence):
    index_list = []
    sentence_making = ""
    pos_start, pos_end = ann_item_span.split("~")
    pos_start = int(pos_start)
    pos_end = int(pos_end)

    for i in range(len(token_list)):
        start_counter = len(sentence_making)

        # temp1 = 423
        # if temp1-1 < int(pos_start) < temp1+1 and temp1-1 < int(start_counter) < temp1+1:
        #     print(1)

        if sentence_making + token_list[i] + " " == sentence[: len(sentence_making)+len(token_list[i])+1]:
            sentence_making = sentence_making + token_list[i] + " "
            end_counter = len(sentence_making)-2
        elif sentence_making + token_list[i] == sentence[: len(sentence_making) + len(token_list[i])]:
            sentence_making = sentence_making + token_list[i]
            end_counter = len(sentence_making)-1
        else:
            raise NameError("sentence making error!")

        if start_counter >= int(pos_start) and end_counter <= int(pos_end) and (token_list[i] in sentence[pos_start: pos_end]):
            index_list.append(i)

    assert sentence_making == sentence
    return index_list

# [('great-grandchild', '305~311'), ('great-grandchild', '316~321')]
def generated_kinship(ann_item, temp_list, sentence):

    for annoted_pair in ann_item:
        kin_type = "Person_" + annoted_pair[0]
        ann_item_span = annoted_pair[1]
        name_propation = annoted_pair[2]
        parenethess = annoted_pair[3] if (annoted_pair[3] != "nickname") and (annoted_pair[3] != "middlename") else "None"
        tagged_index_list = generated_tagged_index(token_list, ann_item_span, sentence)

        #use relation pair form EntityType_BIOES_kinship
        temp_list = generated_BIOES(tagged_index_list, kin_type, temp_list)
        for index in tagged_index_list:
            # temp_list[index] = temp_list[index]+"_"+kin_type
            temp_list[index] = temp_list[index]+"_"+name_propation
            temp_list[index] = temp_list[index]+"_"+parenethess

    return temp_list

def generated_location(ann_item, temp_list, sentence):

    for annoted_pair in ann_item:
        entity_type = annoted_pair[0]
        ann_item_span = annoted_pair[1]
        tagged_index_list = generated_tagged_index(token_list, ann_item_span, sentence)
        temp_list = generated_BIOES(tagged_index_list, entity_type, temp_list)

    return temp_list

def generated_BIOES(tagged_index_list, entity_type, temp_list):
    if len(tagged_index_list) == 0:
        temp_list = temp_list
    if len(tagged_index_list) == 1:
        temp_list[tagged_index_list[0]] = entity_type + "_S"
    if len(tagged_index_list) == 2:
        temp_list[tagged_index_list[0]] = entity_type + "_B"
        temp_list[tagged_index_list[1]] = entity_type + "_E"
    if len(tagged_index_list) > 2:
        temp_list[tagged_index_list[0]] = entity_type + "_B"
        for i in tagged_index_list[1:-1]:
            temp_list[i] = entity_type + "_I"
        temp_list[tagged_index_list[-1]] = entity_type + "_E"
    return temp_list


def return_taged_list(token_list, Annotation_span_list, sentence):
    temp_list = ["O"] * len(token_list)
    global error
    counter = 0
    for ann_item in Annotation_span_list:
        counter += 1
        if type(ann_item) is str:
            if type(eval(ann_item)[0]) == str and counter < 5:
                entity_type = re.sub(" ", "_", eval(ann_item)[0])

                if entity_type == "death":
                    entity_type = "DeathName"
                if entity_type == "Death_Date":
                    entity_type = "DeathDate"
                if entity_type == "Birth_Date":
                    entity_type = "BirthDate"

                ann_item_span = eval(ann_item)[1]
                tagged_index_list = generated_tagged_index(token_list, ann_item_span, sentence)
                temp_list = generated_BIOES(tagged_index_list, entity_type, temp_list)

            if counter == 5 and (type(ann_item) is str):
                temp_list = generated_location(eval(ann_item), temp_list, sentence)

            if counter == 6 and (type(ann_item) is str):
                temp_list = generated_kinship(eval(ann_item), temp_list, sentence)

    return temp_list


if __name__ == '__main__':
    raw_data = pd.read_csv("../sentence_level_corpus_all_information_normalized_improve_location.csv", sep="|")
    result_file = "../data_BIES/BIES.json"

    if os.path.exists(result_file):
        os.remove(result_file)
        print(result_file+" has deleted!!")
    with tqdm(total=len(raw_data)) as pbar:
        for i, v in raw_data.iterrows():
            pbar.update(1)
            result_list = []
            raw_data_list = []
            token_list = [str(i) for i in nlp(v["Sentence"])]
            raw_data_list.append(v["Obituary_ID"])
            raw_data_list.append(token_list)
            # Death_Name | Death_Date | Birth_Date | Age_ID | Location_ID | relationship
            tag_list = return_taged_list(token_list, v[2:], v["Sentence"])
            raw_data_list.append(tag_list)
            result_list.append(str(raw_data_list))
            if len(tag_list) > 0:
                with open(result_file, "a") as f:
                    for i in result_list:
                        f.write(i+'\n')