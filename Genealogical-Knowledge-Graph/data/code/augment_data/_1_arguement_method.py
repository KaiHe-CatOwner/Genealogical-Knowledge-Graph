import random
from nltk.corpus import wordnet
import nltk.tokenize
import re
from fuzzywuzzy import fuzz, process

ERRor_count = 0


def if_in_list(word, list):
    list_1 = [i.lower() for i in list]
    if word.lower() in list_1:
        return True
    else:
        return False


def replace_char(string, replace, position):
    part1 = string[0:int(position[0])]
    part2 = string[int(position[1]):]
    return part1 + replace + part2


def stop_word():
    with open("stop_word.csv", "r") as f:
        stop_list = f.readlines()
    return stop_list


# synonyms is list to get order 1
def get_synonyms(word):
    # word = "I"
    synonyms = []
    word = "".join([char for char in word if char in ' qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'])
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ")
            synonym = "".join(
                [char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'])
            synonyms.append(synonym)

    if word.lower() in synonyms:
        try:
            synonyms.remove(word)
            synonyms.remove(word.lower())
        except:
            pass
    return list(synonyms)


# including the same word
def get_sim_word(word, word_list):
    word = word.lower()
    word_list = [i.lower() for i in word_list]
    if word in word_list:
        return word
    else:
        need_word = process.extractOne(word, word_list)[0]
        return need_word


def synonym_replacement(new_sentence, words_list, n_sr, annotation_list):
    replace_list = []
    for _ in range(n_sr):
        random_synonym, random_word, random_index = commom_insertion_replace(words_list, annotation_list)

        if random_word == " ":
            continue

        try:
            if random_word not in replace_list:
                need_span = re_search_fn(random_word, new_sentence).span()
                new_sentence = new_sentence[:need_span[0]] + random_synonym + new_sentence[need_span[1]:]
                replace_list.append(random_word)
                del(words_list[random_index])
                words_list.insert(random_index, random_synonym)
        except:
            try:
                if random_word not in replace_list:
                    random_word1 = get_sim_word(random_word, words_list)
                    need_span = re_search_fn(random_word1, new_sentence).span()
                    new_sentence = new_sentence[:need_span[0]] + random_synonym + new_sentence[need_span[1]:]
                    replace_list.append(random_word1)
                    del (words_list[random_index])
                    words_list.insert(random_index, random_synonym)
            except:
                print("synonym_replacement opera error!")
    return new_sentence


# def random_trans(words):
#     # 实例化
#     content_1 = " ".join(words)
#     translator = Translator(service_urls=['translate.google.cn'])
#     content_2 = translator.translate(content_1, dest='zh-CN').text
#     content_3 = translator.translate(content_2, dest='en').text
#     content_3 = content_3.split(" ")
#     # print(content_3)
#     return content_3


def commom_insertion_replace(words_list, annotation_list):
    synonyms = []
    counter = 0
    counter_1 = 0
    while len(synonyms) < 1:
        randon_index = random.randint(0, len(words_list) - 1)
        random_word = words_list[randon_index]
        while (len(random_word) < 3) or if_in_list(random_word, annotation_list):
            randon_index = random.randint(0, len(words_list) - 1)
            random_word = words_list[randon_index]
            counter_1 += 1
            if counter_1 >= 15:
                random_word = " "
                randon_index = -1
                break

        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 15:
            synonyms = [" "]
            break
    try:
        random_synonym = synonyms[0]
    except:
        random_synonym = " "

    return random_synonym, random_word, randon_index


def re_search_fn(need_word, sentence):
    random_word_pattern = "(?<=[^a-zA-Z])?" + need_word + "(?=[^a-zA-Z])"
    need_model = re.search(random_word_pattern, sentence)
    return need_model


# insert one synonyms word, only for tokens which tags are "O", annotation_list = [names]
def random_insertion(new_sentence, words_list, n_ri, annotation_list):
    for _ in range(n_ri):
        random_synonym, random_word, random_index = commom_insertion_replace(words_list, annotation_list)
        if random_word == " ":
            continue

        # only insert in the right or left, so it can't insert into names(annotation_list)
        before_flag = random.choice([True, False])
        try:
            need_span = re_search_fn(random_word, new_sentence).span()
            if before_flag:
                insert_pos = need_span[0]
                new_sentence = new_sentence[:insert_pos] + random_synonym + " " + new_sentence[insert_pos:]
                words_list.insert(random_index, random_synonym)
            else:
                insert_pos = need_span[1]
                new_sentence = new_sentence[:insert_pos] + " " + random_synonym + new_sentence[insert_pos:]
                words_list.insert(random_index+1, random_synonym)
        except:
            print("insert opera error!")

    return new_sentence


def return_swap_need_index(new_words, new_kinship, annotation_list, kinship_list):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random.randint(0, len(new_words) - 1)
    random_word_1 = new_words[random_idx_1]
    random_word_2 = new_words[random_idx_2]
    counter = 0
    while (random_word_1 == new_kinship) or (random_word_1 in annotation_list) or (random_word_1 in kinship_list) or \
            (re.search("\w+", random_word_1) is None):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_word_1 = new_words[random_idx_1]
        counter += 1
        if counter > 20:
            random_idx_1 = -1
            break

    counter = 0
    while (random_idx_2 == random_idx_1) or \
            (random_word_2 == new_kinship) or (random_word_2 in annotation_list) or (
            random_word_2 in kinship_list) or \
            (re.search("\w+", random_word_2) is None):
        random_idx_2 = random.randint(0, len(new_words) - 1)
        random_word_2 = new_words[random_idx_2]
        counter += 1
        if counter > 20:
            random_idx_2 = -1
            break
    return random_idx_1, random_idx_2


# only swap no matter tokens
def random_swap(new_sentence, new_words, n_rs, annotation_list, kinship_list, new_kinship):
    # print("========================")
    # print(new_sentence)
    for _ in range(n_rs):
        random_idx_1, random_idx_2 = return_swap_need_index(new_words, new_kinship, annotation_list, kinship_list)
        # print(new_sentence)
        # print(new_words[random_idx_1])
        # print(new_words[random_idx_2])
        # print("~~~")
        if random_idx_1 == -1 or random_idx_2 == -1:
            return new_sentence
        else:
            try:
                word_1_B, word_1_E = re_search_fn(new_words[random_idx_1], new_sentence).span()
                word_2_B, word_2_E = re_search_fn(new_words[random_idx_2], new_sentence).span()

                span_list = sorted([word_1_B, word_1_E, word_2_B, word_2_E])
                seg_1 = new_sentence[:span_list[0]]
                word1 = new_sentence[span_list[2]:span_list[3]]
                seg_2 = new_sentence[span_list[1]:span_list[2]]
                word2 = new_sentence[span_list[0]: span_list[1]]
                seg_3 = new_sentence[span_list[3]:]
                new_sentence = seg_1 + word1 + seg_2 + word2 + seg_3
                # print("seg_1:", seg_1)
                # print("word1:", word1)
                # print("seg_2:", seg_2)
                # print("word2:", word2)
                # print("seg_3:", seg_3)
                # print("~~~")
                # print(span_list)
                # print(word_1_B)
                # print(word_1_E)
                # print(word_2_B)
                # print(word_2_E)
                # print(new_words[random_idx_1])
                # print(new_words[random_idx_2])
                # print("~~~")
            except:
                print("swap error!")
    # print("========================")
    return new_sentence


def random_deletion(new_sentence, new_words, n_rd, annotation_list, kinship_list, new_kinship):
    # obviously, if there's too few words, don't delete
    if len(new_words) < 15:
        return new_sentence

    random_delete_list = []
    while_count = 0
    # randomly delete words
    while len(random_delete_list) < n_rd:
        while_count += 1
        if while_count > 15:
            break
        random_idx = random.randint(0, len(new_words) - 1)
        pick_word = new_words[random_idx]
        if (random_idx not in random_delete_list) and \
                (pick_word not in annotation_list) and (pick_word not in kinship_list) \
                    and (pick_word != new_kinship) and (re.search("\w+", pick_word) is not None):
            random_delete_list.append(random_idx)
            # print("index is", random_idx, "-- word is", new_words[random_idx])

    delete_span_list = []
    for i in random_delete_list:
        try:
            B, E = re_search_fn(new_words[i], new_sentence).span()
            delete_span_list.append((B, E))
        except:
            print("delete error!")

    delete_span_list = sorted(delete_span_list)
    # print(delete_span_list)
    # print("raw_sentence", new_sentence)
    # print("~~~")

    aggreated_span = 0
    for i in delete_span_list:
        blank_flag = False
        if i[0]-aggreated_span-1 > 0:
            start = i[0]-aggreated_span-1
            blank_flag = True
        else:
            start = i[0] - aggreated_span

        if blank_flag:
            end = i[1]-aggreated_span
        else:
            end = i[1]-aggreated_span+1

        # print(start)
        # print(end)
        # print(new_sentence[start: end])

        new_sentence1 = new_sentence[: start]
        new_sentence2 = new_sentence[end:]
        new_sentence = new_sentence1 + new_sentence2

        aggreated_span += i[1] - i[0]

        # print("new_sentence1: ", new_sentence1)
        # print("new_sentence2: ", new_sentence2)
        # print("new_sentence: ", new_sentence)
        # print("~~~")
    return new_sentence


def random_opera(new_sentence, n, annotation_list, kinship_list, new_kinship):
    alpha_sr, alpha_ri, alpha_rs, alpha_rd = n
    words_list = nltk.word_tokenize(new_sentence)
    words_list = [word for word in words_list if word is not '']
    num_words = len(words_list)

    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))
    n_rd = max(1, int(alpha_rd * num_words))

    random_opera = random.randint(0, 4)
    # random_opera = 0

    # if random_opera == 0:
    #     new_sentence = random_trans(words)
    if random_opera == 0:
        new_sentence = random_insertion(new_sentence, words_list, n_ri, annotation_list)
    if random_opera == 1:
        new_sentence = synonym_replacement(new_sentence, words_list, n_sr, annotation_list)
    if random_opera == 2:
        new_sentence = random_swap(new_sentence, words_list, n_rs, annotation_list, kinship_list, new_kinship)
    if random_opera == 3:
        new_sentence = random_deletion(new_sentence, words_list, n_rd, annotation_list, kinship_list, new_kinship)

    return new_sentence


# raw_data_triple=(raw_sentence, [(kinship, position_1)], [(name, position_2)] ),  must have kinship
def data_inteface_1(data_list, new_kinship):
    global ERRor_count
    raw_sentence = data_list[1]
    words = nltk.word_tokenize(raw_sentence)
    kinship_pos_list = []

    for i in eval(data_list[-1]):
        try:
            #sometime the raw word of sentence not in annotation
            kinship_find = get_sim_word(i[0], words)
            kinship_POS_1, kinship_POS_2 = re.search(kinship_find, raw_sentence.lower()).span()
        except:
            # some kinship are not the same format in the sentence
            ERRor_count += 1
            print("ERRor~" + str(ERRor_count))

            kinship_POS_1, kinship_POS_2 = [0, 1]

        kinship_pos_list.append((i[0], [kinship_POS_1, kinship_POS_2]))

    name_pos_list = []
    for i in eval(data_list[-1]):
        index_1, index_2 = i[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))

    if data_list[2] != "":
        index_1, index_2 = eval(data_list[2])[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))

    if data_list[3] != "":
        index_1, index_2 = eval(data_list[3])[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))

    if data_list[4] != "":
        index_1, index_2 = eval(data_list[4])[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))

    if data_list[5] != "":
        index_1, index_2 = eval(data_list[5])[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))

    for i in eval(data_list[-1]):
        index_1, index_2 = i[1].split("~")
        name_pos_list.append((raw_sentence[int(index_1):int(index_2)], [int(index_1), int(index_2)]))


    raw_data_triple = (raw_sentence, kinship_pos_list, name_pos_list)

    return raw_data_triple


# raw_data_triple=(raw_sentence, [(kinship, position)], [(name, position)] ),  must have kinship
def generated_data(new_kinship, data_list, n):
    #(raw_sentence, kinship_pos_list, name_pos_list)
    raw_data_triple = data_inteface_1(data_list, new_kinship)
    raw_sentence = raw_data_triple[0]
    kinship_list = raw_data_triple[1]
    name_pos_list = raw_data_triple[2]

    name_tokenize_list = []
    for i in name_pos_list:
        name_tokenize_list = name_tokenize_list + nltk.word_tokenize(i[0])

    random_kinship_index = random.randint(0, len(kinship_list) - 1)
    selected_triple = kinship_list[random_kinship_index]

    # random pick one kinship to replace, if choose the same one, then choose the next one
    select_count = 0
    while select_count < 10:
        if new_kinship != selected_triple[0]:
            # replace one kind of kinship
            new_sentence = replace_char(raw_sentence, new_kinship, selected_triple[1])
            break
        else:
            select_count += 1
            random_kinship_index = random.randint(0, len(kinship_list) - 1)
            selected_triple = kinship_list[random_kinship_index]

    if select_count == 10:
        print("random selected kinship wrong@")
        print(new_kinship + "_____________________" + raw_sentence)
        new_sentence = raw_sentence

    kinship_seg_list = []
    for i in [i[0] for i in kinship_list]:
        kinship_seg_list = kinship_seg_list + nltk.word_tokenize(i)

    kinship_seg_list = list(set(kinship_seg_list))
    new_sentence = random_opera(new_sentence, n, name_tokenize_list, kinship_seg_list, new_kinship)
    # new_sentence = random_opera(new_sentence, n, annotation_list, kinship_seg_list, new_kinship)

    return new_sentence.strip(), selected_triple


if __name__ == "__main__":
    # alpha_sr, alpha_ri, alpha_rs, alpha_rd
    n = [0.1, 0.1, 0.1, 0.1]

    # input = (sentencem [kinships], [annotation_list])
    data_list = "899|He enjoyed woodworking, fishing, dancing, and especially spending time with his family.Eddie is survived by special friend, Sandy Frank of Canton; children, Danny of Canton, Kevin (Della) of Decorah, Randy (Elaine St. Mary) of Lime Springs and Brenda (Allen C.) Holthaus of Cresco; stepdaughter, Elaine (Jeff) Geottel; grandchildren, Adam Whalen, Richard Hollenbeck, Rusty Whalen, Erika Whalen, Cassidy Whalen, Mikel Hansen, Rachel Vsetecka, Sarah Halweg, Devin Holthaus, Carter, Olivia, and Wyatt St. Mary, Patrick, Aislinn, Gabriel, and Eireann Geottel; 16 great-grandchildren; a brother, LaVerne (Neldah) Whalen of Rochester; sister-in-law, Shirley Whalen of Rochester; and many nieces and nephews.He was preceded in death by his parents; daughter-in-law, Joan Whalen; grandson, Mark Whalen; great-grandson, Bryson Holthaus; and brothers and sisters, Harold, Quintin, Paul, Delone, Joe, Larry, Louis, and Lucy Whalen.Funeral service will be at 11 a.m. July 27, at St. Agnes Plymouth Rock Catholic Church, with Fr.|('death', '87~92')||||[('Location-Danny', '166~172'), ('Location-Decorah', '191~198'), ('Location-Lime Springs', '227~239'), ('Location-Brenda (Allen C.) Holthaus', '274~280'), ('Location-LaVerne (Neldah) Whalen', '618~627'), ('Location-Shirley Whalen', '662~671')]|[('brother', '591~614', 'no', 'coupled-name'), ('sister-in-law', '644~658', 'no', 'None'), ('sibling', '854~860', 'yes', 'None'), ('sibling', '862~869', 'yes', 'None'), ('sibling', '871~875', 'yes', 'None'), ('sibling', '877~883', 'yes', 'None'), ('sibling', '885~888', 'yes', 'None'), ('sibling', '890~895', 'yes', 'None'), ('sibling', '897~902', 'yes', 'None'), ('sibling', '908~919', 'yes', 'None'), ('child', '157~162', 'no', 'None'), ('child', '174~187', 'no', 'coupled-name'), ('child', '200~223', 'no', 'coupled-name'), ('child', '244~270', 'no', 'coupled-name'), ('step-daughter', '296~317', 'no', 'coupled-name'), ('daughter-in-law', '759~770', 'no', 'None'), ('grandchild', '334~345', 'no', 'None'), ('grandchild', '347~365', 'no', 'None'), ('grandchild', '367~379', 'no', 'None'), ('grandchild', '381~393', 'no', 'None'), ('grandchild', '395~409', 'no', 'None'), ('grandchild', '411~423', 'no', 'None'), ('grandchild', '425~440', 'no', 'None'), ('grandchild', '442~454', 'no', 'None'), ('grandchild', '456~470', 'no', 'None'), ('grandchild', '472~478', 'yes', 'None'), ('grandchild', '480~486', 'yes', 'None'), ('grandchild', '492~506', 'yes', 'None'), ('grandchild', '508~515', 'yes', 'None'), ('grandchild', '517~524', 'yes', 'None'), ('grandchild', '526~533', 'yes', 'None'), ('grandchild', '539~554', 'yes', 'None'), ('grandson', '782~793', 'no', 'None'), ('great-grandson', '811~826', 'no', 'None')]"
    data_list = data_list.split("|")
    replace_kinship = "wife"
    print("new kinship is", replace_kinship)
    # print(data_list[1][115:125])

    new_sentence, selected_triple = generated_data(replace_kinship, data_list, n)
    print(new_sentence)
