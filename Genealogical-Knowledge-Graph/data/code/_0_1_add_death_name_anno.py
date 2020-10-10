import nltk
import nltk.data
import lxml.etree as etree
import os
import spacy
import re
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")


def parse_xml_text(file):
    doc = etree.parse(file)
    text = doc.xpath("//TEXT")[0].text
    return text


def find_death_in_sent(name, sent, anno_list, sentence_span_begin):
    sentence_text = sent.attrib['sentence_text'].lower()
    name_token_list = name.lower().split(" ")
    name_token_list = [i for i in name_token_list if len(i) > 0]

    anno_span_list=[]
    for anno in anno_list:
        ann_span = anno.attrib["spans"].split("~")
        anno_span_list.append(ann_span)

    add_list = []
    add_index_list = []
    for index, name_token in enumerate(name_token_list):
        add_flag = True
        name_token = name_token.replace(".", "\.")
        name_token = name_token.replace("(", "\(")
        name_token = name_token.replace(")", "\)")
        for find_name_token in re.finditer(name_token, sentence_text):
            for ann_pair in anno_span_list:
                if (find_name_token.start()+int(sentence_span_begin) >= int(ann_pair[0])) and (find_name_token.end()+int(sentence_span_begin) <= int(ann_pair[1])) :
                    add_flag = False
            if add_flag:
                # print(find_name_token)
                add_list.append(find_name_token)
                add_index_list.append(index)

    if (0 not in add_index_list) and (len(name_token_list) not in add_index_list) :
        add_list = []

    total_list = []
    for i in add_list:
        total_list.append((i.start(), i.end()))

    if len(total_list)>0:
        total_list_new = split_num_l(total_list)
    else:
        total_list_new = []
    # print(sentence_text)
    # print(total_list)
    # print(total_list_new)
    # print("==========")

    return total_list_new


def split_num_l(sort_lst):
    total_list_new = []
    temp_list = []
    for index in range(len(sort_lst)):
        if index < len(sort_lst)-1:
            if sort_lst[index][1]+1 == sort_lst[index+1][0]:
                temp_list.append(sort_lst[index])
            else:
                if len(temp_list) ==0:
                    total_list_new.append([sort_lst[index]])
                else:
                    total_list_new.append(temp_list)
                    temp_list = []
        else:
            if len(temp_list)>0:
                temp_list.append(sort_lst[index])
                total_list_new.append(temp_list)
            else:
                total_list_new.append([sort_lst[index]])

    for one_name_index in range(len(total_list_new)):
        total_list_new[one_name_index] = (total_list_new[one_name_index][0][0], total_list_new[one_name_index][-1][-1])

    return total_list_new



if __name__ == '__main__':

    dir_0 = "../sentence_level_corpus"
    target_data_file_path = '../sentence_level_corpus_add_death'
    file_list = []
    for file_tuple in os.walk(dir_0):
        for file in file_tuple[2]:
            file = os.path.join(dir_0, file)
            file_list.append(file)

    print("There are %s file !" % str(len(file_list)))

    total_add = 0
    file_count = 0
    for file in tqdm(file_list):
        # file = os.path.join(dir_0, "127.txt.xml")
        # print(file)

        file_count += 1
        doc = etree.parse(file)

        root = etree.Element("Obituary_Annotation")

        Title_name = doc.xpath("//Titled_Name")[0]
        death_name = Title_name.attrib["text"]
        etree.SubElement(root, Title_name.tag, attrib=Title_name.attrib)

        Location_Name_Link_list = doc.xpath("//Location-Name")
        for i in Location_Name_Link_list:
            etree.SubElement(root, i.tag, attrib=i.attrib)

        sentence_list = doc.xpath("//sentence")

        anno_add = 0
        name_ID = 0
        for sentence in sentence_list:
            sent_item = etree.SubElement(root, sentence.tag, attrib=sentence.attrib)
            anno_list = sentence.getchildren()
            sentence_span_begin = sentence.attrib['sentence_span'].split("~")[0]
            # only add the deceased when there are annotations
            if len(anno_list) > 0:
                for i in anno_list:
                    new_tag = "annotation" + str(int(i.tag.split("annotation")[1])+anno_add)
                    if "Name_ID" not in i.attrib['id']:
                        etree.SubElement(sent_item, new_tag, attrib=i.attrib)
                    else:
                        i.attrib["id"] = "Name_ID_"+str(name_ID)
                        etree.SubElement(sent_item, new_tag, attrib=i.attrib)
                        name_ID+=1

                add_list = find_death_in_sent(death_name, sentence, anno_list, sentence_span_begin)
                token_list =[str(i) for i in nlp(sentence.attrib["sentence_text"])]
                for token_span in add_list:
                    begin, end = token_span
                    text = sentence.attrib["sentence_text"][begin: end]
                    new_attrib = {'id':'Name_ID_'+str(name_ID), 'spans':str(begin+int(sentence_span_begin))+"~"+str(end+int(sentence_span_begin)), 'text':text, 'Type':"death"}
                    name_ID += 1

                    new_tag = "annotation" + str(int(i.tag.split("annotation")[1])+anno_add+1)
                    etree.SubElement(sent_item, new_tag, attrib=new_attrib)
                    anno_add +=1
                    total_add+=1


        tree = etree.ElementTree(root)
        target_data_file = os.path.join(target_data_file_path, str(file.split('/')[-1]))
        tree.write(target_data_file, pretty_print=True, xml_declaration=True, encoding='utf-8')
        # break

    print("anno_add for the deseaced:", total_add)
