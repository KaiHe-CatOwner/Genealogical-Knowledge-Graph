import nltk
import nltk.data
import lxml.etree as etree
import os
import spacy
import re
from tqdm import tqdm
#this file generate sentence level annotation file


nlp = spacy.load('en_core_web_sm')

def parse_xml_text(file):
    doc = etree.parse(file)
    text = doc.xpath("//TEXT")[0].text
    return text


def nltk_split_sentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

# def spacy_split_sentence(doc):
#     nlp = spacy.load('en_core_web_sm')
#     nlp.add_pipe(spacy_custom_token_sentence, before='parser')
#     doc = nlp(doc)
#     sentences_list = [sent.text for sent in doc.sents]
#     return sentences_list
#
#
# def spacy_custom_token_sentence(doc):
#     for token in doc[:-1]:
#         if token.text == '...':
#             doc[token.i+1].is_sent_start = True
#     return doc


def return_sentence_pos(sentences, text):
    #return start and end position for each sentence
    sentence_start_end_list = []

    for sentence in sentences:
        # try:
        sentence = re.sub("\(", "\\(", sentence)
        sentence = re.sub("\)", "\\)", sentence)
        sentence = re.sub("\?", "\\?", sentence)
        sentence = re.sub("\+", "\\+", sentence)
        sentence = re.sub("\$", "\\$", sentence)
        sentence = re.sub("\[", "\\[", sentence)
        sentence = re.sub("\]", "\\]", sentence)
        sentence = re.sub("\*", "\\*", sentence)
        sentence = re.sub("\.", "\\.", sentence)

        temp_finditer = list(re.finditer(sentence, text))
        if len(temp_finditer) == 0:
            raise NameError("no find sentence in text")

        temp_count = 0
        for m in temp_finditer:
            if len(temp_finditer) == 1:
                sentence_start_end_list.append((m.start(), m.end()))
            if len(temp_finditer) > 1 and temp_count <= len(temp_finditer):
                sentence_start_end_list.append(temp_finditer[temp_count].span())
                temp_count += 1
                break

        # except:
        #     print("re.finder error~~~~~~~~~~~~~~~~~~~~~~")

    return sentence_start_end_list


def joint_wrong_token_sentence(sentences, sentence_start_end_list):
    assert len(sentence_start_end_list) == len(sentences)

    for i in range(len(sentences) - 1, -1, -1):
        # deal with ";" token
        if sentences[i][0] == ";" or ((re.match(r"[a-z]", sentences[i]) is not None) and (re.match(r".[a-z]*[A-Z]+", sentences[i-1][::-1]) is not None)):
            blank_count = int(sentence_start_end_list[i][0]) - int(sentence_start_end_list[i - 1][1])
            sentences[i - 1] = sentences[i - 1] + " "*blank_count + sentences[i]
            sentences.remove(sentences[i])

            sentence_start_end_list[i - 1] = (sentence_start_end_list[i -1 ][0], sentence_start_end_list[i][1])
            sentence_start_end_list.remove(sentence_start_end_list[i])

    assert len(sentence_start_end_list) == len(sentences)
    return sentences, sentence_start_end_list


def generate_sentence_level(sentence_start_end_list, sentences, file, target_data_file_path):
    doc = etree.parse(file)
    TAG = doc.xpath("//TAGS")[0]
    # print(file)
    target_data_file = os.path.join(target_data_file_path, str(file.split('/')[-1]))
    # print(target_data_file)
    sentences, sentence_start_end_list = joint_wrong_token_sentence(sentences, sentence_start_end_list)
    try:
        dic_sentence_start_end = {}
        for i in range(len(sentence_start_end_list)):
            dic_sentence_start_end[sentences[i]] = sentence_start_end_list[i]
    except:
        print("error~~~~~~~~~")
        pass

    TAG_list = []
    for child in TAG.iterchildren():
        TAG_list.append(child)
    #print('TAG_list =' ,TAG_list)
    #print('TAG_list[1].attrib =' ,TAG_list[1].attrib)

    root = etree.Element("Obituary_Annotation")
    Title_name = doc.xpath("//Titled_Name")[0]
    etree.SubElement(root, Title_name.tag, attrib=Title_name.attrib)
    Location_Name_Link_list = doc.xpath("//TAGS/Location-Name")
    for i in Location_Name_Link_list:
        etree.SubElement(root, i.tag, attrib=i.attrib)

    count = 0
    sen_ID = 0
    for sentence in sentences:
        start, end = dic_sentence_start_end[sentence]
        sentence_item = etree.SubElement(root, "sentence", attrib={'sentence_ID': str(sen_ID), 'sentence_text': sentence, 'sentence_span': str(start)+"~"+str(end)})
        #print('sentence_item.attrib =' ,sentence_item.attrib)
        sen_ID += 1

        for i in TAG_list:
            try:
                if i.tag != "Titled_Name" and i.tag != "Location-Name":
                    #print('!!',start,end)
                    #print('!' ,int(i.attrib['spans'].split("~")[0]),int(i.attrib['spans'].split("~")[1]))
                    if start <= int(i.attrib['spans'].split("~")[0]) and end >= int(i.attrib['spans'].split("~")[1]):
                        x = etree.SubElement(sentence_item, 'annotation'+str(count), attrib=i.attrib)
                        #print('annotation'+str(count)+'=' ,etree.SubElement(sentence_item, 'annotation'+str(count), attrib=i.attrib).attrib)
                        count += 1
            except:
                print("generate error!")

    tree = etree.ElementTree(root)
    tree.write(target_data_file, pretty_print=True, xml_declaration=True, encoding='utf-8')


if __name__ == '__main__':

    dir_0 = "../processed_data_Location_ID_corrected"
    target_data_file_path = '../sentence_level_corpus'
    file_list = []
    for file_tuple in os.walk(dir_0):
        for file in file_tuple[2]:
            file = os.path.join(dir_0, file)
            file_list.append(file)

    print("There are %s file !" % str(len(file_list)))

    file_count = 0
    for file in tqdm(file_list):
        # print(file.split('/')[-1])
        # file = os.path.join(dir_0, "2943.txt.xml")
        # print(file)
        # print(file_count)
        file_count += 1
        text = parse_xml_text(file)
        # sentences = spacy_split_sentence(text)
        sentences = nltk_split_sentence(text)
        #print(sentences)
        sentences = [i.strip() for i in sentences]
        sentence_start_end_list = return_sentence_pos(sentences, text)
        #print(sentence_start_end_list)
        generate_sentence_level(sentence_start_end_list, sentences, file, target_data_file_path)
        # break

