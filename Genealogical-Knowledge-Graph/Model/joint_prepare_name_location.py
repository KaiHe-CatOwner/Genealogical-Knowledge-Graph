import torch
from TP_FN_FP import get_triple_O_seg, improved_result, formatted_outpus
import re


# sentence = sentence_index
# return = [person_name, Person_common_encoder_example, task4: tags_sentence[i - 1]]
def extracted_entity_list(pick_str, sentence, tokens_sentence, tags_sentence, common_encoder, train_flag=False,
                          tokenizer=None):
    entity = ""
    entity_list = []
    entity_common_encoder = torch.zeros(common_encoder[0][0].shape).cuda()
    entity_common_encoder_list = []
    gold_location_name_pair_list = []
    word = 0
    split_name_count = 0
    for i in range(len(tags_sentence)):
        if pick_str in tags_sentence[i]:
            entity = entity + deal_str(tokens_sentence[i], tokenizer)
            entity_label = tags_sentence[i].lower()  # .replace(" ", "")
            entity_common_encoder = entity_common_encoder + common_encoder[sentence][i]
            word += 1
        elif tags_sentence[i] == "O":
            if entity != "":
                if pick_str == "Location" and (entity[-1] == "." or entity[-1] == ","):
                    entity = entity[:-1]
                entity_list.append(deal_str(entity))

                if pick_str == "Location" and train_flag:
                    if " and " in entity_label.split("ocation-")[1]:
                        entity_label = entity_label.split("ocation-")[1].split(" and ")
                        split_name_count += len(entity_label) - 1
                        for name_split in entity_label:
                            gold_location_name_pair_list.append(deal_str(entity + "-" + deal_str(name_split)))
                    else:
                        gold_location_name_pair_list.append(deal_str(entity + "-" + entity_label.split("ocation-")[1]))

                entity_common_encoder_list.append([entity, entity_common_encoder / word, tags_sentence[i - 1]])
                entity_common_encoder = torch.zeros(common_encoder[0][0].shape).cuda()
                entity = ""

        if tokens_sentence[i] == "<PAD>" or tokens_sentence[i] == "[PAD]":
            break


    return list(set(entity_list)), entity_common_encoder_list, list(set(gold_location_name_pair_list)), split_name_count


def Make_task5_to_same_length(Batch_task5_pair, Batch_task5_common_encoder, max_len, common_encoder_shape):
    for index, (one_sentence_pair, one_sentence_common_encoder) in enumerate(
            zip(Batch_task5_pair, Batch_task5_common_encoder)):
        if len(one_sentence_pair) < max_len:
            for i in range(max_len - len(one_sentence_pair)):
                one_sentence_pair.append('<PAD>')
                one_sentence_common_encoder.append(torch.zeros(common_encoder_shape * 3).cuda())

        Batch_task5_pair[index] = one_sentence_pair
        Batch_task5_common_encoder[index] = one_sentence_common_encoder
    return Batch_task5_pair, Batch_task5_common_encoder


# gold_location_name_pair_list : ['cincinnati,oh-Evelyn and Reilly Murphy', 'rochester,mn-Joseph, Benjamin, Matthew, and Jonathan Devlin']
def prepared_task5_gold_dic(person_names, Location_names, gold_location_name_pair_list, task5_TAGS):
    sentence_gold_dic = {}
    person_names = list(set(person_names))
    Location_names = list(set(Location_names))
    if len(person_names) == 0 or len(Location_names) == 0:
        sentence_gold_dic["<PAD>"] = task5_TAGS["<PAD>"]
        return sentence_gold_dic
    else:
        for person_number in range(len(person_names)):
            for Location_number in range(len(Location_names)):
                Location_Person = Location_names[Location_number] + "-" + person_names[person_number]
                if Location_Person in gold_location_name_pair_list:
                    sentence_gold_dic[Location_Person] = task5_TAGS["yes"]
                else:
                    sentence_gold_dic[Location_Person] = task5_TAGS["no"]

        return sentence_gold_dic


# only for valid, compute loss
def get_task_5_predict_to_gold(batch_gold_list, batch_predict_list, task5_TAGS):
    task_5_predict_gold = []
    for sentence_index in range(len(batch_predict_list)):
        one_sentene_task_5_predict_gold = []
        for pair in batch_predict_list[sentence_index]:
            if pair[0] == "<PAD>":
                one_sentene_task_5_predict_gold.append(task5_TAGS["<PAD>"])
            elif deal_str(pair[0]) in [deal_str(i) for i in batch_gold_list[sentence_index].keys()]:
                one_sentene_task_5_predict_gold.append(batch_gold_list[sentence_index][pair[0]])
            else:
                one_sentene_task_5_predict_gold.append(task5_TAGS["no"])

        task_5_predict_gold.append(one_sentene_task_5_predict_gold)

    task_5_predict_gold = torch.tensor(task_5_predict_gold, dtype=torch.long).cuda()

    return task_5_predict_gold


def deal_str(str, tokenizer=None):
    if tokenizer is None:
        reg = "[^0-9A-Za-z\(\)-]"
        return re.sub(reg, '', str).lower()
    else:
        str = tokenizer.ids_to_tokens[str]
        reg = "[^0-9A-Za-z\(\)-]"
        return re.sub(reg, '', str).lower()


# input is model pred, out put is pair (name_location : 1/0)
def make_task5_final_list(Batch_task5_pair, prediction_person_location):
    # [ [("L_N_pair ", pred), ()], [], [] ]
    task5_prediction_list = []
    for index, (one_sentence_pair, one_sentence_pred) in enumerate(zip(Batch_task5_pair, prediction_person_location)):
        TASk5_prediction_one_sentence_list = []
        for number, (L_N, pred) in enumerate(zip(one_sentence_pair, one_sentence_pred)):
            TASk5_prediction_one_sentence_list.append((L_N, int(pred)))
        task5_prediction_list.append(TASk5_prediction_one_sentence_list)
    return task5_prediction_list


def prepare_train_batch_data(common_embbeding, TOEKNS_5, TAGS_5, TAGS_5_NEW, batch_data, bert_flag=False,
                             tokenizer=None):
    batch_gold_list = []
    Batch_task5_pair_total = []
    Batch_task5_common_encoder_total = []
    max_len = 0
    split_name_count_total = 0
    for sentence in range(common_embbeding.shape[0]):
        # train and valid have different name_location pair
        if bert_flag:
            tokens_person_location_sentence = batch_data.tokens[sentence].cpu().numpy().tolist()
        else:
            tokens_person_location_sentence = [TOEKNS_5.vocab.itos[i] for i in
                                               batch_data.tokens[sentence].cpu().numpy().tolist()]

        tags_person_location_sentence = [TAGS_5.vocab.itos[i] for i in
                                         batch_data.tags[sentence].cpu().numpy().tolist()]

        gold_names_list, names_common_encoder, _, _ = extracted_entity_list("Person", sentence,
                                                                            tokens_person_location_sentence,
                                                                            tags_person_location_sentence,
                                                                            common_embbeding, tokenizer=tokenizer)

        gold_Location_list, Location_common_encoder, gold_location_name_pair_list, split_name_count = extracted_entity_list(
            "Location", sentence, tokens_person_location_sentence, tags_person_location_sentence,
            common_embbeding, train_flag=True, tokenizer=tokenizer)

        split_name_count_total += split_name_count

        sentence_gold_dic = prepared_task5_gold_dic(gold_names_list, gold_Location_list, gold_location_name_pair_list,
                                                    TAGS_5_NEW)

        batch_gold_list.append(sentence_gold_dic)

        # form embedding of pair
        Batch_task5_pair_total, Batch_task5_common_encoder_total, max_len = prepared_task5_batch_list(
            names_common_encoder, Location_common_encoder, max_len, Batch_task5_pair_total, \
            common_embbeding.shape[2], Batch_task5_common_encoder_total)

    for i in Batch_task5_pair_total:
        pad_num = 0
        if i == ['<PAD>']:
            pad_num += 1

    if pad_num == len(Batch_task5_pair_total):
        return [], [], [], 0

    Batch_task5_pair_total, Batch_task5_common_encoder = Make_task5_to_same_length(Batch_task5_pair_total,
                                                                                   Batch_task5_common_encoder_total,
                                                                                   max_len, common_embbeding.shape[2])
    Batch_task5_common_encoder = make_task5_batch(Batch_task5_common_encoder)

    return Batch_task5_pair_total, Batch_task5_common_encoder, batch_gold_list, split_name_count_total


def format_improved_task5_input(sentence, improve_flag):
    sen_len = 0
    for i in sentence:
        if i != "<PAD>":
            sen_len += 1

    predicted_triple_list = get_triple_O_seg(sentence[:sen_len])
    # print("predicted_triple_list", predicted_triple_list)
    if improve_flag:
        predicted_triple_list = improved_result(predicted_triple_list)
    else:
        predicted_triple_list = formatted_outpus(predicted_triple_list)

    new_sentece = ["O"] * sen_len
    for temp_list in predicted_triple_list:
        for i in temp_list:
            new_sentece[i[0]] = i[1]

    return new_sentece


def prepare_test_batch_data(common_embbeding, TOEKNS_5, TAGS_5, TAGS_5_NEW, TAGS_12, entity_extracted_in_12,
                            batch_data, improve_flag, bert_flag=False, tokenizer=None):
    batch_gold_list = []
    Batch_task5_pair_total = []
    Batch_task5_common_encoder_total = []
    max_len = 0
    split_name_count_total = 0
    for sentence in range(batch_data.batch_size):
        # train and valid have different name_location pair

        if bert_flag:
            # tokens_person_location_sentence = [TOEKNS_5(i) for i in
            #                                    batch_data.tokens[sentence].cpu().numpy().tolist()]
            tokens_person_location_sentence = batch_data.tokens[sentence].cpu().numpy().tolist()
        else:
            tokens_person_location_sentence = [TOEKNS_5.vocab.itos[i] for i in
                                               batch_data.tokens[sentence].cpu().numpy().tolist()]
        tags_person_location_sentence = [TAGS_5.vocab.itos[i] for i in
                                         batch_data.tags[sentence].cpu().numpy().tolist()]

        gold_names_list, _, _, _ = extracted_entity_list("Person", sentence, tokens_person_location_sentence,
                                                         tags_person_location_sentence,
                                                         common_embbeding, tokenizer=tokenizer)

        gold_Location_list, _, gold_location_name_pair_list, split_name_count = extracted_entity_list("Location",
                                                                                                      sentence,
                                                                                                      tokens_person_location_sentence,
                                                                                                      tags_person_location_sentence,
                                                                                                      common_embbeding,
                                                                                                      train_flag=True,
                                                                                                      tokenizer=tokenizer)

        split_name_count_total += split_name_count

        sentence_gold_dic = prepared_task5_gold_dic(gold_names_list, gold_Location_list,
                                                    gold_location_name_pair_list, TAGS_5_NEW)
        batch_gold_list.append(sentence_gold_dic)

        # finish sentence_gold_dic, do valid need
        task12_tags_one_sentence = [TAGS_12.vocab.itos[i] for i in
                                    entity_extracted_in_12[sentence].cpu().numpy().tolist()]

        task12_tags_one_sentence = format_improved_task5_input(task12_tags_one_sentence, improve_flag)

        _, names_common_encoder, _, _ = extracted_entity_list("Person", sentence,
                                                              tokens_person_location_sentence,
                                                              task12_tags_one_sentence,
                                                              common_embbeding, tokenizer=tokenizer)

        _, Location_common_encoder, _, _ = extracted_entity_list("Location", sentence,
                                                                 tokens_person_location_sentence,
                                                                 task12_tags_one_sentence,
                                                                 common_embbeding, tokenizer=tokenizer)

        # form embedding of pair
        Batch_task5_pair_total, Batch_task5_common_encoder_total, max_len = prepared_task5_batch_list(
            names_common_encoder, Location_common_encoder, \
            max_len, Batch_task5_pair_total, \
            common_embbeding.shape[2], Batch_task5_common_encoder_total)

    # for i in Batch_task5_pair_total:
    #     pad_num = 0
    #     if i == ['<PAD>']:
    #         pad_num += 1
    #
    # if pad_num == len(Batch_task5_pair_total):
    #     return [], [], [], 0

    Batch_task5_pair_total, Batch_task5_common_encoder = Make_task5_to_same_length(Batch_task5_pair_total,
                                                                                   Batch_task5_common_encoder_total,
                                                                                   max_len, common_embbeding.shape[2])
    Batch_task5_common_encoder = make_task5_batch(Batch_task5_common_encoder)
    return Batch_task5_pair_total, Batch_task5_common_encoder, batch_gold_list, split_name_count_total


def prepared_task5_batch_list(task5_Person_common_encoder_total, Location_common_encoder_total,
                              max_len, Batch_task5_pair, common_encoder_shape, Batch_task5_common_encoder):
    if len(task5_Person_common_encoder_total) == 0 or len(Location_common_encoder_total) == 0:
        Batch_task5_pair.append(["<PAD>"])
        Batch_task5_common_encoder.append([torch.zeros(common_encoder_shape * 3).cuda()])
        max_len = 1 if max_len < 1 else max_len
        return Batch_task5_pair, Batch_task5_common_encoder, max_len
    else:

        Location_Name_in_one_sentence_pair = []
        Location_Name_in_one_sentence_common_encoder = []
        for Person_number in range(len(task5_Person_common_encoder_total)):
            for Location_number in range(len(Location_common_encoder_total)):
                Person_Location_camp = torch.cat((task5_Person_common_encoder_total[Person_number][1],
                                                  Location_common_encoder_total[Location_number][1],
                                                  task5_Person_common_encoder_total[Person_number][1].mul(
                                                      Location_common_encoder_total[Location_number][1])))

                Person_Location_tokens_camp = Location_common_encoder_total[Location_number][0] + "-" + \
                                              task5_Person_common_encoder_total[Person_number][0]

                Location_Name_in_one_sentence_pair.append(Person_Location_tokens_camp)

                Location_Name_in_one_sentence_common_encoder.append(Person_Location_camp)

        Batch_task5_pair.append(Location_Name_in_one_sentence_pair)
        max_len = len(Location_Name_in_one_sentence_pair) if max_len < len(
            Location_Name_in_one_sentence_pair) else max_len
        Batch_task5_common_encoder.append(Location_Name_in_one_sentence_common_encoder)
        return Batch_task5_pair, Batch_task5_common_encoder, max_len


# get task 5 model input
def make_task5_batch(Batch_task5_common_encoder):
    Batch_task5_common_encoder_list = []
    for sentence_index in range(len(Batch_task5_common_encoder)):
        one_sentence_pair_list = []
        for pair_index in range(len(Batch_task5_common_encoder[sentence_index])):
            one_sentence_pair_list.append(Batch_task5_common_encoder[sentence_index][pair_index])

        one_sentence_pair_list = torch.stack(one_sentence_pair_list)
        Batch_task5_common_encoder_list.append(one_sentence_pair_list)

    Batch_task5_final_common_encoder = torch.stack(Batch_task5_common_encoder_list)
    # return size = [batch, time_step, dim]
    return Batch_task5_final_common_encoder


def prepared_task4_batch_list(task4_Person_common_encoder_total, max_len, common_encoder_shape, task4_TAGS,
                              batch_names, batch_tags, batch_common_encoder):
    if len(task4_Person_common_encoder_total) == 0:
        batch_tags.append(task4_TAGS.vocab.stoi["<PAD>"])
        batch_common_encoder.append([torch.zeros(common_encoder_shape).cuda()])
        batch_names.append(["<PAD>"])
        max_len = 1 if max_len < 1 else max_len
        return batch_tags, batch_common_encoder, batch_names, max_len
    else:
        max_len = len(task4_Person_common_encoder_total) if max_len < len(
            task4_Person_common_encoder_total) else max_len
        one_sentence_names = []
        one_sentence_tags = []
        one_sentence_common_encoder = []
        for person in task4_Person_common_encoder_total:
            # person = [name, tag_str, common_encoder]
            one_sentence_names.append(person[0])
            one_sentence_tags.append(task4_TAGS.vocab.stoi[person[1]])
            one_sentence_common_encoder.append(person[2])

        batch_names.append(one_sentence_names)
        batch_tags.append(one_sentence_tags)
        batch_common_encoder.append(one_sentence_common_encoder)
        return batch_tags, batch_common_encoder, batch_names, max_len


def Make_task4_to_same_length(batch_tags, batch_common_encoder, batch_names, max_len, task4_TAGS, common_encoder_shape):
    # task4_TAGS = {"nickname":index, "couple-name":index, "previous-last-name":index, "none":index, "O":index, "<PAD>":index}
    for index, (one_sentence_names, one_sentence_tags, one_sentence_common_encoder) in enumerate(
            zip(batch_names, batch_tags, batch_common_encoder)):
        if len(one_sentence_names) < max_len:
            for i in range(max_len - len(one_sentence_names)):
                one_sentence_names.append('<PAD>')
                one_sentence_tags.append(task4_TAGS.vocab.stoi["<PAD>"])
                one_sentence_common_encoder.append(torch.zeros(common_encoder_shape).cuda())
        batch_names[index] = one_sentence_names
        batch_tags[index] = one_sentence_tags
        batch_common_encoder[index] = one_sentence_common_encoder
    return batch_names, batch_tags, batch_common_encoder


# get task 4 model input, and tags(tensor)
def make_task4_batch_and_tags(batch_common_encoder, batch_tags):
    Batch_common_encoder_list = []
    Batch_tags_list = []
    for i in range(len(batch_common_encoder)):
        one_sentence_tensor_list = []
        one_sentence_tags_list = []
        for j in range(len(batch_common_encoder[i])):
            one_sentence_tensor_list.append(batch_common_encoder[i][j])
            one_sentence_tags_list.append(torch.tensor(batch_tags[i][j]))
        one_sentence_tensor_list = torch.stack(one_sentence_tensor_list)
        one_sentence_tags_list = torch.stack(one_sentence_tags_list)
        Batch_common_encoder_list.append(one_sentence_tensor_list)
        Batch_tags_list.append(one_sentence_tags_list)

    Batch_common_encoder = torch.stack(Batch_common_encoder_list)
    Batch_tags = torch.stack(Batch_tags_list)
    assert Batch_common_encoder.shape[1] == Batch_tags.shape[1]
    # return size = [batch, time_step, dim]
    return Batch_common_encoder, Batch_tags


if __name__ == "__main__":
    pick_str = "Location"
    sentence = 0
    tokens_sentence = ['survivors', 'include', 'her', 'two', 'sons', ',', 'daniel', 'and', 'timothy', 'shanahan', 'of',
                       'austin', ',', 'mn', ';', 'and', 'her', 'daughter', ',', 'cynthia', 'perzynski', 'of', 'rose',
                       'creek', ',', 'mn', ';', 'three', 'grandchildren', ',', 'stacy', 'ann', '(', 'justin', ')',
                       'deno', 'of', 'mantorville', ',', 'tj', 'perzynski', 'of', 'rochester', ',', 'and', 'jami',
                       'weatherly', 'of', 'austin', '.']
    tags_sentence = ['O', 'O', 'O', 'O', 'O', 'O', 'Person', 'O', 'Person', 'Person', 'O',
                     'Location-Daniel and Timothy Shanahan', 'Location-Daniel and Timothy Shanahan',
                     'Location-Daniel and Timothy Shanahan', 'O', 'O', 'O', 'O', 'O', 'Person', 'Person', 'O',
                     'Location-Cynthia Perzynski', 'Location-Cynthia Perzynski', 'Location-Cynthia Perzynski',
                     'Location-Cynthia Perzynski', 'O', 'O', 'O', 'O', 'Person', 'Person', 'Person', 'Person', 'Person',
                     'Person', 'O', 'Location-Stacy Ann (Justin) Deno', 'O', 'Person', 'Person', 'O',
                     'Location-TJ Perzynski', 'O', 'O', 'Person', 'Person', 'O', 'Location-Jami Weatherly', 'O']
    common_encoder = torch.randn(1, len(tags_sentence), 3).cuda()
    entity_list, entity_common_encoder_list, gold_location_name_pair_list, split_name_count = extracted_entity_list(
        pick_str, sentence, tokens_sentence, tags_sentence, common_encoder, train_flag=True)
    print(entity_list)
    print(gold_location_name_pair_list)
