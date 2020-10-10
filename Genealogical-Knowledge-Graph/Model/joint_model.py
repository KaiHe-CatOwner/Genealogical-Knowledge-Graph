#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

# this import is useful
import torch.optim as optim
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import warnings
from utils import prepared_data, filter_content, establish_file, MyThread, make_dic_performance, \
    calculate_PRF, write_dic_performance, record_each_performance, \
    record_detail_result
from TP_FN_FP import make_task5_TP_FN_FP_sentence
import os
import time
import argparse
from collections import Counter
import sys
from joint_prepare_name_location import make_task5_final_list, get_task_5_predict_to_gold, prepare_test_batch_data, prepare_train_batch_data, extracted_entity_list


warnings.filterwarnings("ignore")
# Common Encoder
parser = argparse.ArgumentParser(description="LSTM Model")
parser.add_argument('--My_Test_data_flag', default=True, type=bool)
parser.add_argument('--Word_embedding_size', default=50, type=int)
parser.add_argument('--EPOCH', default=10, type=int)
parser.add_argument('--NUM_FOLD', default=1, type=int)
parser.add_argument('--EARLY_STOP_NUM', default=10, type=int)
parser.add_argument('--EARLY_STOP_threshold', default=0.1, type=float)
parser.add_argument('--NUM_Layer_Common_Encoder', default=1, type=int)
parser.add_argument('--Task_list', default="[12]", type=str, help="[12,3,4,5]")
parser.add_argument('--Loss_weight_list', default="[0.6, 0.2, 0.2, 0.5]", type=str, help="[0.6, 0.2, 0.2, 0.5]")
parser.add_argument('--BATCH_SIZE', default=32, type=int)
parser.add_argument('--Data_ID', default="999", type=str)
parser.add_argument('--MIN_EPOCH', default=50, type=int)
parser.add_argument('--Start_Decay', default=50, type=int)
parser.add_argument('--Decay_Cycle', default=5, type=int)
parser.add_argument('--GPU', default="3", type=str)
# Task 0
parser.add_argument('--Hidden_Size_Common_Encoder', default=300, type=int)
parser.add_argument('--LR_common_encoder', default=5e-3, type=float)
parser.add_argument('--LR_0_Decay_Rate', default=0.5, type=float)
parser.add_argument('--LN', default=True, type=bool)
parser.add_argument('--L2_Weight_Decay', default=1e-5, type=float)
parser.add_argument('--DROP_OUT', default=0.5, type=float)
parser.add_argument('--IF_Early_Stop', default=True, type=bool)
parser.add_argument('--Optim', default="RMSprop", type=str)
parser.add_argument('--Weight_Min', default=3, type=int)
parser.add_argument('--Weight_Max', default=10, type=int)
# Task 12
parser.add_argument('--HIDDEN_SIZE_12', default=300, type=int)
parser.add_argument('--NUM_LAYER_12', default=2, type=int)
parser.add_argument('--LR_12', default=5e-4, type=float)
parser.add_argument('--LR_12_Decay_Rate', default=0.5, type=float)
parser.add_argument('--Simply_num', default=50, type=int, help="# 0 will contain all kinship /10 / 50")
parser.add_argument('--Augment_Data', default=False, type=bool)
parser.add_argument('--IF_Entity_Weight', action='store_true', default=False)
parser.add_argument('--IF_Task12_Adap_Loss', action='store_true', default=False)
parser.add_argument('--Entity_tag_Weight', default=1, type=float)
parser.add_argument('--Weight_Lambda_12', default=0.01, type=float)
parser.add_argument('--Improve_Flag', default=False, type=bool)
# Task 3
parser.add_argument('--HIDDEN_SIZE_3', default=50, type=int)
parser.add_argument('--NUM_LAYER_3', default=1, type=int)
parser.add_argument('--LR_3', default=5e-4, type=float)
parser.add_argument('--LR_3_Decay_Rate', default=0.5, type=float)
parser.add_argument('--IF_NamePropagation_Weight', action='store_true', default=True)
parser.add_argument('--IF_Task3_Adap_Loss', action='store_true', default=True)
parser.add_argument('--Name_propagation_tag_Weight', default=1, type=float)
parser.add_argument('--Weight_Lambda_3', default=40, type=float)
# Task 4
parser.add_argument('--HIDDEN_SIZE_4', default=50, type=int)
parser.add_argument('--NUM_LAYER_4', default=1, type=int)
parser.add_argument('--LR_4', default=5e-4, type=float)
parser.add_argument('--LR_4_Decay_Rate', default=0.5, type=float)
parser.add_argument('--IF_Parentheses_Weight', action='store_true', default=True)
parser.add_argument('--IF_Task4_Adap_Loss', action='store_true', default=True)
parser.add_argument('--Parentheses_tag_Weight', default=1, type=float)
parser.add_argument('--Weight_Lambda_4', default=0.5, type=float)
# Task 5
parser.add_argument('--NUM_LAYER_5', default=1, type=int)
parser.add_argument('--HIDDEN_SIZE_5', default=50, type=int)
parser.add_argument('--LR_5', default=5e-4, type=float)
parser.add_argument('--LR_5_Decay_Rate', default=0.5, type=float)
parser.add_argument('--IF_Person_location_Weight', action='store_true', default=False)
parser.add_argument('--IF_Task5_Adap_Loss', action='store_true', default=False)
parser.add_argument('--Weight_Lambda_5', default=0.8, type=float)
parser.add_argument('--Person_location_tag_Weight', default=1, type=float)
args = parser.parse_args()

print("GPU", args.GPU)
print("IF_Early_Stop", args.IF_Early_Stop)
print("Simply_num", args.Simply_num)
print("Data_ID", args.Data_ID)
print("Word_embedding_size", args.Word_embedding_size)
print("My_Test_data_flag", args.My_Test_data_flag)
print("Task_list", args.Task_list)
print("Loss_weight_list", args.Loss_weight_list)
print()

print("args.IF_Entity_Weight", args.IF_Entity_Weight)
if args.IF_Entity_Weight:
    if args.IF_Task5_Adap_Loss:
        print("args.IF_Task12_Adap_Loss", args.IF_Task5_Adap_Loss)
        print("args.Weight_Lambda_12", args.Weight_Lambda_5)
    else:
        print("args.IF_Task12_Adap_Loss", args.IF_Task5_Adap_Loss)
        print("args.Entity_tag_Weight", args.Person_location_tag_Weight)

print()
print("args.IF_Person_location_Weight", args.IF_Person_location_Weight)
if args.IF_Person_location_Weight:
    if args.IF_Task5_Adap_Loss:
        print("args.IF_Task5_Adap_Loss", args.IF_Task5_Adap_Loss)
        print("args.Weight_Lambda_5", args.Weight_Lambda_5)
    else:
        print("args.IF_Task5_Adap_Loss", args.IF_Task5_Adap_Loss)
        print("args.Person_location_tag_Weight", args.Person_location_tag_Weight)

OPTIMIZER = eval("optim." + args.Optim)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
LR_Decay_List = np.arange(args.Start_Decay, args.EPOCH, args.Decay_Cycle).tolist()

if args.Word_embedding_size==300:
    args.Test_50_embeding_flag= False
elif args.Word_embedding_size==50:
    args.Test_50_embeding_flag = True
else:
    raise Exception("Word_embedding_size dim wrong!")

Embedding_requires_grad = False
BATCH_FIRST = True
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_performance_total = '../result/detail_performance/LSTM/total_performance.txt'
file_performance = '../result/detail_performance/LSTM/performance_' + str(sys.argv[1:]) + '.txt'
file_result = '../result/detail_results/LSTM/result_' + str(sys.argv[1:]) + '.txt'
file_model_save = "../result/save_model/LSTM/" + str(sys.argv[1:])
file_traning_performerance ='../result/detail_training/LSTM/training_'+str(sys.argv[1:])+'.txt'


# Common encoder
class My_Common_Encoder(nn.Module):
    def __init__(self, embedding_dim, TOEKNS):
        super(My_Common_Encoder, self).__init__()
        self.to(device)
        self.TOEKNS = TOEKNS
        self.vocab_size = len(TOEKNS.vocab)
        self.embedding_dim = embedding_dim
        self.drop_out = nn.Dropout(args.DROP_OUT)
        self.LN = nn.LayerNorm(self.embedding_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim).to(device)
        pretrained_embeddings = self.TOEKNS.vocab.vectors
        self.embedding.weight.data.copy_(pretrained_embeddings)
        for param in self.embedding.parameters():
            param.requires_grad = Embedding_requires_grad

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=args.Hidden_Size_Common_Encoder,
            num_layers=args.NUM_Layer_Common_Encoder,
            batch_first=BATCH_FIRST,
            bidirectional=True,
        )

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        embedded = self.embedding(x)
        if args.LN: embedded = self.LN(embedded)
        embedded = self.drop_out(embedded)
        out_put_lstm, encoder_lstm_h_c = self.lstm(embedded)
        return out_put_lstm


# TASK 1, 2
class My_Model_12(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model_12, self).__init__()
        self.to(device)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=args.HIDDEN_SIZE_12,
            num_layers=args.NUM_LAYER_12,
            batch_first=BATCH_FIRST,
            bidirectional=True,
        )

        self.fc = nn.Linear(args.HIDDEN_SIZE_12*2, self.output_dim, bias=False)

        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_embbeding):
        out_put_lstm, encoder_lstm_h_c = self.lstm(input_embbeding)
        return self.fc(out_put_lstm)


# TASK 3
class My_Model_3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model_3, self).__init__()
        self.to(device)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.name_propagation_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=args.HIDDEN_SIZE_3,
            num_layers=args.NUM_LAYER_3,
            batch_first=BATCH_FIRST,
            bidirectional=True,
        )

        self.fc_name_propagation = nn.Linear(args.HIDDEN_SIZE_3 * 2, self.output_dim, bias=False)

    def forward(self, input_embbeding):
        out_put_name_propagation, h_c = self.name_propagation_lstm(input_embbeding)
        return self.fc_name_propagation(out_put_name_propagation)


# TASK 4
class My_Model_4(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(My_Model_4, self).__init__()
        self.to(device)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.parentheses_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=args.HIDDEN_SIZE_4,
            num_layers=args.NUM_LAYER_4,
            batch_first=BATCH_FIRST,
            bidirectional=True,
        )

        self.fc_parentheses = nn.Linear(args.HIDDEN_SIZE_4 * 2, self.output_dim, bias=False)

    def forward(self, input_embbeding):
        out_put_parentheses, parentheses_h_c = self.parentheses_lstm(input_embbeding)
        return self.fc_parentheses(out_put_parentheses)


# TASK 5
class My_Model_5(nn.Module):
    def __init__(self, input_dim, output_dim, TOEKNS_5, TAGS_5, TAGS_5_NEW):
        super(My_Model_5, self).__init__()
        self.TAGS_5_NEW = TAGS_5_NEW
        self.TOEKNS_5 = TOEKNS_5
        self.TAGS_5 = TAGS_5
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model_5 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=args.HIDDEN_SIZE_5,
            num_layers=args.NUM_LAYER_5,
            batch_first=BATCH_FIRST,
            bidirectional=True,
        )
        self.fc = nn.Linear(args.HIDDEN_SIZE_5 * 2, self.output_dim, bias=False)

    def forward(self, common_embedding, batch_data):
        Batch_task5_pair, Batch_task5_common_encoder, batch_gold_list, split_name_count_total = \
            prepare_train_batch_data(common_embedding, self.TOEKNS_5, self.TAGS_5, self.TAGS_5_NEW, batch_data)

        if Batch_task5_pair == []:
            return [], [], [], 0

        out_put_lstm, encoder_lstm_h_c = self.model_5(Batch_task5_common_encoder)
        prediction_person_location = self.fc(out_put_lstm)

        return Batch_task5_pair, batch_gold_list, prediction_person_location, split_name_count_total

    @staticmethod
    def only_for_task5_valid(my_model_5, common_embedding, batch_data,
                             TOEKNS_5, TAGS_5, TAGS_5_NEW):
        Batch_task5_pair, Batch_task5_common_encoder, batch_gold_list, split_name_count_total = \
            prepare_train_batch_data(common_embedding, TOEKNS_5, TAGS_5, TAGS_5_NEW, batch_data)

        if Batch_task5_pair == []:
            return [], [], [], 0

        out_put_lstm, encoder_lstm_h_c = my_model_5.model_5(Batch_task5_common_encoder)
        prediction_person_location = my_model_5.fc(out_put_lstm)

        return Batch_task5_pair, batch_gold_list, prediction_person_location, split_name_count_total

    @staticmethod
    def my_valid(my_model_5, common_embbeding, batch_data,
                 TOEKNS_5, TAGS_5, TAGS_5_NEW, TAGS_12, entity_extracted_in_12):

        Batch_task5_pair, Batch_task5_common_encoder, batch_gold_list, split_name_count_total  = \
            prepare_test_batch_data(common_embbeding, TOEKNS_5, TAGS_5, TAGS_5_NEW, TAGS_12, entity_extracted_in_12, batch_data, args.Improve_Flag)

        if Batch_task5_pair == [] :
            return [], [], [], 0

        out_put_lstm, _ = my_model_5.model_5(Batch_task5_common_encoder)
        prediction_person_location = my_model_5.fc(out_put_lstm)

        return Batch_task5_pair, batch_gold_list, prediction_person_location, split_name_count_total


class Train_joint_model():
    def __init__(self, dic_init, TAGS_5_NEW, train_test_iterator_dic, TOEKNS_TAGS_dic):
        self.TOEKNS_TAGS_dic = TOEKNS_TAGS_dic
        self.TAGS_5_NEW = TAGS_5_NEW

        self.common_encoder = My_Common_Encoder(args.Word_embedding_size, self.TOEKNS_TAGS_dic[0][0]).to(device)
        self.train_iterator_0 = train_test_iterator_dic[0][0]
        self.optimizer_0 = OPTIMIZER(
            params=filter(lambda p: p.requires_grad, self.common_encoder.parameters()),
            lr=args.LR_common_encoder, weight_decay=args.L2_Weight_Decay)

        self.scheduler_0 = MultiStepLR(self.optimizer_0, milestones=LR_Decay_List,
                                        gamma=args.LR_0_Decay_Rate)

        if 12 in eval(args.Task_list):
            self.model_12 = My_Model_12(*dic_init[12]).to(device)
            self.train_iterator_12 = train_test_iterator_dic[12][0]
            self.optimizer_12 = OPTIMIZER(
                params=filter(lambda p: p.requires_grad, self.model_12.parameters()),
                lr=args.LR_12, weight_decay=args.L2_Weight_Decay)

            self.scheduler_12 = MultiStepLR(self.optimizer_12, milestones=LR_Decay_List,
                                       gamma=args.LR_12_Decay_Rate)

        if 3 in eval(args.Task_list):
            self.model_3 = My_Model_3(*dic_init[3]).to(device)
            self.train_iterator_3 = train_test_iterator_dic[3][0]
            self.optimizer_3 = OPTIMIZER(
                params=filter(lambda p: p.requires_grad, self.model_3.parameters()),
                lr=args.LR_3, weight_decay=args.L2_Weight_Decay)
            self.scheduler_3 = MultiStepLR(self.optimizer_3, milestones=LR_Decay_List,
                                            gamma=args.LR_3_Decay_Rate)

        if 4 in eval(args.Task_list):
            self.model_4 = My_Model_4(*dic_init[4]).to(device)
            self.train_iterator_4 = train_test_iterator_dic[4][0]
            self.optimizer_4 = OPTIMIZER(
                params=filter(lambda p: p.requires_grad, self.model_4.parameters()),
                lr=args.LR_4, weight_decay=args.L2_Weight_Decay)
            self.scheduler_4 = MultiStepLR(self.optimizer_4, milestones=LR_Decay_List,
                                           gamma=args.LR_4_Decay_Rate)

        if 5 in eval(args.Task_list):
            self.model_5 = My_Model_5(*dic_init[5], self.TOEKNS_TAGS_dic[5][0], self.TOEKNS_TAGS_dic[5][1], self.TAGS_5_NEW).to(device)
            self.train_iterator_5 = train_test_iterator_dic[5][0]
            self.optimizer_5 = OPTIMIZER(
                params=filter(lambda p: p.requires_grad, self.model_5.parameters()),
                lr=args.LR_5, weight_decay=args.L2_Weight_Decay)
            self.scheduler_5 = MultiStepLR(self.optimizer_5, milestones=LR_Decay_List,
                                           gamma=args.LR_5_Decay_Rate)

    def joint_train(self, tasks_batch_data_dic):
        loss_list = []
        prediction_dic = {}

        common_embedding = self.common_encoder(tasks_batch_data_dic[0].tokens)

        if 12 in eval(args.Task_list):
            thread_12 = MyThread(self.model_12.forward, args=[common_embedding])
            thread_12.start()
        if 3 in eval(args.Task_list):
            thread_3 = MyThread(self.model_3.forward, args=[common_embedding])
            thread_3.start()
        if 4 in eval(args.Task_list):
            thread_4 = MyThread(self.model_4.forward, args=[common_embedding])
            thread_4.start()
        if 5 in eval(args.Task_list):
            thread_5 = MyThread(self.model_5.forward, args=[common_embedding, tasks_batch_data_dic[5]])
            thread_5.start()

        if 12 in eval(args.Task_list):
            thread_12.join()
            predictions_12 = thread_12.get_result().permute(0, 2, 1)
            criterion_12 = get_loss(args.IF_Entity_Weight, args.IF_Task12_Adap_Loss, args.Weight_Lambda_12,
                                         tasks_batch_data_dic[12].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, entity_flag=True)
            loss_12 = criterion_12(predictions_12, tasks_batch_data_dic[12].tags)
            loss_list.append(loss_12)
            prediction_dic[12] = predictions_12
        if 3 in eval(args.Task_list):
            thread_3.join()
            predictions_3 = thread_3.get_result().permute(0, 2, 1)
            criterion_3 = get_loss(args.IF_NamePropagation_Weight, args.IF_Task3_Adap_Loss, args.Weight_Lambda_3,
                                        tasks_batch_data_dic[3].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, name_propagation_flag=True)
            loss_3 = criterion_3(predictions_3, tasks_batch_data_dic[3].tags)
            loss_list.append(loss_3)
            prediction_dic[3] = predictions_3
        if 4 in eval(args.Task_list):
            thread_4.join()
            predictions_4 = thread_4.get_result().permute(0, 2, 1)
            criterion_4 = get_loss(args.IF_Parentheses_Weight, args.IF_Task4_Adap_Loss, args.Weight_Lambda_4,
                                        tasks_batch_data_dic[4].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, parentheses_flag=True)
            loss_4 = criterion_4(predictions_4, tasks_batch_data_dic[4].tags)
            loss_list.append(loss_4)
            prediction_dic[4] = predictions_4
        if 5 in eval(args.Task_list):
            thread_5.join()
            batch_task5_pair, task_5_gold_batch, prediction_5, split_name_count_total = thread_5.get_result()

            if len(prediction_5) == 0:
                # loss_5 = torch.zeros(1, requires_grad = True)
                loss_5 = 0
                loss_list.append(loss_5)
                task5_results = [[], [], []]
                prediction_dic[5] = task5_results
            else:
                prediction_person_location_argmax = torch.argmax(prediction_5, 2)

                # input is model pred, out put is pair (name_location : 1/0)
                task_5_predict_batch = make_task5_final_list(batch_task5_pair, prediction_person_location_argmax)

                task_5_predict_gold_batch = get_task_5_predict_to_gold(task_5_gold_batch, task_5_predict_batch, self.TAGS_5_NEW)
                assert len(task_5_predict_gold_batch) == len(task_5_gold_batch)
                assert len(task_5_predict_gold_batch) == len(task_5_predict_batch)
                criterion = get_loss(args.IF_Person_location_Weight, args.IF_Task5_Adap_Loss, args.Weight_Lambda_5,
                                          task_5_predict_gold_batch, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW,
                                     Location_person_flag=True)

                prediction_5 = prediction_5.permute(0, 2, 1)
                loss_5 = criterion(prediction_5, task_5_predict_gold_batch)
                loss_list.append(loss_5)

                task5_results = [task_5_gold_batch, task_5_predict_batch, prediction_5]
                prediction_dic[5] = task5_results

        # get following task's result
        joint_loss = 0
        for i in range(len(loss_list)):
            joint_loss += eval(args.Loss_weight_list)[i] * loss_list[i]

        # for print
        if 5 not in eval(args.Task_list):
            loss_5 = "No task 5~"
            split_name_count_total = 0

        return joint_loss, prediction_dic, loss_5, split_name_count_total

    def need_train_batch(self):
        need_list = []
        need_zip_list = []

        need_list.append(0)
        need_zip_list.append(self.train_iterator_0)
        if 12 in eval(args.Task_list):
            need_list.append(12)
            need_zip_list.append(self.train_iterator_12)
        if 3 in eval(args.Task_list):
            need_list.append(3)
            need_zip_list.append(self.train_iterator_3)
        if 4 in eval(args.Task_list):
            need_list.append(4)
            need_zip_list.append(self.train_iterator_4)
        if 5 in eval(args.Task_list):
            need_list.append(5)
            need_zip_list.append(self.train_iterator_5)
        return need_list, need_zip_list

    def save_model(self):
        self.model_state_dic = {}
        self.model_state_dic['model_common_encoder_state_dict'] = self.common_encoder.state_dict()
        if 12 in eval(args.Task_list):
            self.model_state_dic['model_12_state_dict'] = self.model_12.state_dict()
        if 3 in eval(args.Task_list):
            self.model_state_dic[
                'model_3_state_dict'] = self.model_3.state_dict()
        if 4 in eval(args.Task_list):
            self.model_state_dic['model_4_state_dict'] = self.model_4.state_dict()
        if 5 in eval(args.Task_list):
            self.model_state_dic['model_5_state_dict'] = self.model_5.state_dict()
        torch.save(self.model_state_dic, file_model_save)

    def train_model(self, fold_num):
        save_epoch = 0
        early_stop_num = args.EARLY_STOP_NUM
        max_F = 0
        self.model_state_dic = {}
        print("start training...")
        for epoch in range(args.EPOCH):
            self.common_encoder.train()
            self.scheduler_0.step()
            if 12 in eval(args.Task_list):
                self.model_12.train()
                self.scheduler_12.step()
            if 3 in eval(args.Task_list):
                self.model_3.train()
                self.scheduler_3.step()
            if 4 in eval(args.Task_list):
                self.model_4.train()
                self.scheduler_4.step()
            if 5 in eval(args.Task_list):
                self.model_5.train()
                self.scheduler_5.step()

            total_loss = 0
            total_loss_5 = 0
            step = 0

            need_list, need_zip_list = self.need_train_batch()

            # batch level opera, so need accmulate value
            P_macro_batch_all_task = []
            R_macro_batch_all_task = []
            F_macro_batch_all_task = []
            task5_P_macro_list = []
            task5_R_macro_list = []
            task5_F_macro_list = []
            split_name_count_total = 0

            total_TP_FN_FP_dic = {}
            for task in eval(args.Task_list):
                total_TP_FN_FP_dic[task] = {}

            for batch_item in zip(*need_zip_list):
                tasks_batch_data_dic = {}
                for index in range(len(need_list)):
                    tasks_batch_data_dic[need_list[index]] = batch_item[index]

                self.optimizer_0.zero_grad()
                if 12 in eval(args.Task_list):
                    self.optimizer_12.zero_grad()
                if 3 in eval(args.Task_list):
                    self.optimizer_3.zero_grad()
                if 4 in eval(args.Task_list):
                    self.optimizer_4.zero_grad()
                if 5 in eval(args.Task_list):
                    self.optimizer_5.zero_grad()

                joint_loss, prediction_dic, loss_person_location, split_name_count_batch = self.joint_train(tasks_batch_data_dic)
                split_name_count_total += split_name_count_batch
                joint_loss.backward()

                self.optimizer_0.step()
                if 12 in eval(args.Task_list):
                    self.optimizer_12.step()
                if 3 in eval(args.Task_list):
                    self.optimizer_3.step()
                if 4 in eval(args.Task_list):
                    self.optimizer_4.step()
                if 5 in eval(args.Task_list):
                    self.optimizer_5.step()

                # tasks_result_dic[i] = [task12_micro_result, task12_macro_result, task12_TP_FN_FP]
                batch_TP_FN_FP_dic, _, _, _ = get_batch_TP_FN_FP(prediction_dic, tasks_batch_data_dic, self.TOEKNS_TAGS_dic)

                for task in eval(args.Task_list):
                    for tag, TP_FN_FP_tuple in batch_TP_FN_FP_dic[task].items():
                        if tag in total_TP_FN_FP_dic[task].keys():
                            total_TP_FN_FP_dic[task][tag] = tuple(np.array(total_TP_FN_FP_dic[task][tag]) + np.array(TP_FN_FP_tuple))
                        else:
                            total_TP_FN_FP_dic[task][tag] = TP_FN_FP_tuple

                step += 1
                total_loss += joint_loss
                if 5 in eval(args.Task_list):
                    total_loss_5 += loss_person_location

            # dic_reslut[5] = [task5_micro_result, task5_macro_result, task5_TP_FN_FP]
            dic_reslut = get_epoch_P_R_F(total_TP_FN_FP_dic)

            # # tasks_result_dic[i] = [task12_micro_result, task12_macro_result, task12_TP_FN_FP]
            for key, value in dic_reslut.items():
                if not np.isnan(dic_reslut[key][1][0]):
                    P_macro_batch_all_task.append(dic_reslut[key][1][0])
                    R_macro_batch_all_task.append(dic_reslut[key][1][1])
                    F_macro_batch_all_task.append(dic_reslut[key][1][2])

                    if key == 5:
                        task5_P_macro_list.append(dic_reslut[key][1][0])
                        task5_R_macro_list.append(dic_reslut[key][1][1])
                        task5_F_macro_list.append(dic_reslut[key][1][2])


            epoch_P_macro = np.average(P_macro_batch_all_task)
            epoch_R_macro = np.average(R_macro_batch_all_task)
            epoch_F_macro = np.average(F_macro_batch_all_task)

            task5_P_macro_list = np.average(task5_P_macro_list)
            task5_R_macro_list = np.average(task5_R_macro_list)
            task5_F_macro_list = np.average(task5_F_macro_list)

            print('training...  Data_ID: %2d, Epoch: %2d, Train Loss: %.3f, P: %.3f, R: %.3f, F: %.3f' %
                  (int(args.Data_ID), epoch, total_loss / step, epoch_P_macro, epoch_R_macro, epoch_F_macro))
            if 5 in eval(args.Task_list):
                print("                                                                                         task 5: Loss: %.3f, P: %.3f, R: %.3f, F: %.3f"%
                  (total_loss_5/ step, task5_P_macro_list, task5_R_macro_list, task5_F_macro_list))

            if fold_num == 0 and epoch == 0:
                write_flag = "w"
            else:
                write_flag = "a"

            with open(file_traning_performerance, write_flag) as f:
                f.writelines('Data_ID: %2d, Fold_num: %2d,  Epoch: %2d, Train Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \n' %
                  (int(args.Data_ID), fold_num, epoch, total_loss / step,  epoch_P_macro, epoch_R_macro, epoch_F_macro))
                if 5 in eval(args.Task_list):
                    f.writelines("                                                                                        task 5: Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \n" %
                        (total_loss_5/ step, task5_P_macro_list, task5_R_macro_list, task5_F_macro_list))

            # one epoch training over
            if args.IF_Early_Stop:
                if epoch_F_macro > max_F + args.EARLY_STOP_threshold:
                    early_stop_num = args.EARLY_STOP_NUM
                    max_F = epoch_F_macro
                    save_epoch = epoch
                elif epoch > args.MIN_EPOCH:
                    early_stop_num -= 1

                if early_stop_num == 0:
                    print("early stop, in epoch: %d, starting validation ...."%(int(save_epoch)))
                    self.save_model()
                    return epoch, split_name_count_total
            else:
                if epoch == args.EPOCH:
                    print("finshed max epoch: %d, starting validation ...."%(int(epoch)))
                    self.save_model()
                    return epoch, split_name_count_total

        print("early stop failed, reach max epoch: %d, starting validation ...."%(int(epoch)))
        self.save_model()
        return epoch, split_name_count_total


class Valid_joint_model(nn.Module):
    def __init__(self, dic_init, TAGS_5_NEW, train_test_iterator_dic, TOEKNS_TAGS_dic):
        super(Valid_joint_model, self).__init__()
        self.TOEKNS_TAGS_dic = TOEKNS_TAGS_dic
        self.TAGS_5_NEW = TAGS_5_NEW
        self.train_test_iterator_dic = train_test_iterator_dic

        self.valid_flag = True
        checkpoint = torch.load(file_model_save)

        self.common_encoder = My_Common_Encoder(args.Word_embedding_size, self.TOEKNS_TAGS_dic[0][0]).to(device)
        self.common_encoder.load_state_dict(checkpoint['model_common_encoder_state_dict'])
        self.test_iterator_0 = self.train_test_iterator_dic[0][1]

        if 12 in eval(args.Task_list):
            self.model_12 = My_Model_12(*dic_init[12]).to(device)
            self.model_12.load_state_dict(checkpoint['model_12_state_dict'])
            self.model_12.eval()
            self.test_iterator_12 = self.train_test_iterator_dic[12][1]

        if 3 in eval(args.Task_list):
            self.model_3 = My_Model_3(*dic_init[3]).to(device)
            self.model_3.load_state_dict(checkpoint['model_3_state_dict'])
            self.model_3.eval()
            self.test_iterator_3 = self.train_test_iterator_dic[3][1]

        if 4 in eval(args.Task_list):
            self.model_4 = My_Model_4(*dic_init[4]).to(device)
            self.model_4.load_state_dict(checkpoint['model_4_state_dict'])
            self.model_4.eval()
            self.test_iterator_4 = self.train_test_iterator_dic[4][1]

        if 5 in eval(args.Task_list):
            self.model_5 = My_Model_5(*dic_init[5], self.TOEKNS_TAGS_dic[5][0], self.TOEKNS_TAGS_dic[5][1], self.TAGS_5_NEW).to(device)
            self.model_5.load_state_dict(checkpoint['model_5_state_dict'])
            self.model_5.eval()
            self.test_iterator_5 = self.train_test_iterator_dic[5][1]

    def joint_valid(self, tasks_batch_data_dic):
        loss_list = []
        prediction_dic = {}

        common_embbeding = self.common_encoder(tasks_batch_data_dic[0].tokens)

        if 12 in eval(args.Task_list):
            thread_12 = MyThread(self.model_12.forward, args=[common_embbeding])
            thread_12.start()
        if 3 in eval(args.Task_list):
            thread_3 = MyThread(self.model_3.forward, args=[common_embbeding])
            thread_3.start()
        if 4 in eval(args.Task_list):
            thread_4 = MyThread(self.model_4.forward, args=[common_embbeding])
            thread_4.start()

        if 12 in eval(args.Task_list):
            thread_12.join()
            predictions_12 = thread_12.get_result().permute(0, 2, 1)
            criterion_12 = get_loss(args.IF_Entity_Weight, args.IF_Task12_Adap_Loss, args.Weight_Lambda_12,
                                         tasks_batch_data_dic[12].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, entity_flag=True)
            loss_12 = criterion_12(predictions_12, tasks_batch_data_dic[12].tags)
            loss_list.append(loss_12)
            prediction_dic[12] = predictions_12

        if 3 in eval(args.Task_list):
            thread_3.join()
            predictions_3 = thread_3.get_result().permute(0, 2, 1)
            criterion_3 = get_loss(args.IF_NamePropagation_Weight, args.IF_Task3_Adap_Loss, args.Weight_Lambda_3,
                                         tasks_batch_data_dic[3].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, name_propagation_flag=True)
            loss_3 = criterion_3(predictions_3, tasks_batch_data_dic[3].tags)
            loss_list.append(loss_3)
            prediction_dic[3] = predictions_3
        if 4 in eval(args.Task_list):
            thread_4.join()
            predictions_4 = thread_4.get_result().permute(0, 2, 1)
            criterion_4 = get_loss(args.IF_Parentheses_Weight, args.IF_Task4_Adap_Loss, args.Weight_Lambda_4,
                                        tasks_batch_data_dic[4].tags, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, parentheses_flag=True)
            loss_4 = criterion_4(predictions_4, tasks_batch_data_dic[4].tags)
            loss_list.append(loss_4)
            prediction_dic[4] = predictions_4
        if 5 in eval(args.Task_list):
            if 12 in eval(args.Task_list):
                thread_5 = MyThread(My_Model_5.my_valid, args=[self.model_5, common_embbeding, tasks_batch_data_dic[5],
                                                             self.TOEKNS_TAGS_dic[5][0], self.TOEKNS_TAGS_dic[5][1],
                                                             self.TAGS_5_NEW, self.TOEKNS_TAGS_dic[12][1], torch.argmax(predictions_12, 1)])
            else:
                thread_5 = MyThread(My_Model_5.only_for_task5_valid,
                                    args=[self.model_5, common_embbeding, tasks_batch_data_dic[5], self.TOEKNS_TAGS_dic[5][0], self.TOEKNS_TAGS_dic[5][1], self.TAGS_5_NEW])

            thread_5.start()
            thread_5.join()
            batch_task5_pair, task_5_gold_batch, prediction_5, split_name_count_batch = thread_5.get_result()

            if len(prediction_5) == 0:
                loss_5 = 0
                loss_list.append(loss_5)
                task5_results = [[], [], []]
                prediction_dic[5] = task5_results
            else:
                prediction_person_location_argmax = torch.argmax(prediction_5, 2)
                # input is model pred, out put is pair (name_location : 1/0)
                task_5_predict_batch = make_task5_final_list(batch_task5_pair, prediction_person_location_argmax)

                task_5_predict_gold_batch = get_task_5_predict_to_gold(task_5_gold_batch, task_5_predict_batch, self.TAGS_5_NEW)
                criterion = get_loss(args.IF_Person_location_Weight, args.IF_Task5_Adap_Loss, args.Weight_Lambda_5,
                                                   task_5_predict_gold_batch, self.TOEKNS_TAGS_dic, self.TAGS_5_NEW, Location_person_flag=True)

                prediction_5 = prediction_5.permute(0, 2, 1)
                loss_5 = criterion(prediction_5, task_5_predict_gold_batch)
                loss_list.append(loss_5)

                task5_results = [task_5_gold_batch, task_5_predict_batch, prediction_5]
                prediction_dic[5] = task5_results

        # get following task's result
        joint_loss = 0
        for i in range(len(loss_list)):
            joint_loss += eval(args.Loss_weight_list)[i] * loss_list[i]

        if 5 not in eval(args.Task_list):
            loss_5 = "No task 5~"
            split_name_count_batch = 0

        return joint_loss, prediction_dic, loss_5, split_name_count_batch

    def need_test_batch(self):
        need_list = []
        need_zip_list = []

        need_list.append(0)
        need_zip_list.append(self.test_iterator_0)

        if 12 in eval(args.Task_list):
            need_list.append(12)
            need_zip_list.append(self.test_iterator_12)
        if 3 in eval(args.Task_list):
            need_list.append(3)
            need_zip_list.append(self.test_iterator_3)
        if 4 in eval(args.Task_list):
            need_list.append(4)
            need_zip_list.append(self.test_iterator_4)
        if 5 in eval(args.Task_list):
            need_list.append(5)
            need_zip_list.append(self.test_iterator_5)

        return need_list, need_zip_list

    def valid_model(self, fold_num, epoch):
        torch.cuda.empty_cache()
        with torch.no_grad():

            total_loss = 0
            total_loss_5 = 0
            step = 0

            need_task_list, need_task_zip_list = self.need_test_batch()

            P_macro_batch_all_task = []
            R_macro_batch_all_task = []
            F_macro_batch_all_task = []

            task5_P_macro_list = []
            task5_R_macro_list = []
            task5_F_macro_list = []

            P_micro_batch_all_task = []
            R_micro_batch_all_task = []
            F_micro_batch_all_task = []

            split_name_count_total_valid = 0

            total_TP_FN_FP_dic = {}
            for task in eval(args.Task_list):
                total_TP_FN_FP_dic[task] = {}

            test_data_count = 0
            for batch_num, item in enumerate(zip(*need_task_zip_list)):
                # tasks_batch_data_dic : dic[taskID]=data_set (test_set)
                tasks_batch_data_dic = {}
                for index in range(len(need_task_list)):
                    tasks_batch_data_dic[need_task_list[index]] = item[index]

                joint_loss, prediction_dic, loss_person_location, split_name_count_batch_valid = self.joint_valid(tasks_batch_data_dic)
                split_name_count_total_valid += split_name_count_batch_valid

                batch_TP_FN_FP_dic, tokens_total, targets_total, predictions_total = get_batch_TP_FN_FP(prediction_dic, tasks_batch_data_dic, self.TOEKNS_TAGS_dic,  valid_flag=True)
                for task in eval(args.Task_list):
                    for tag, TP_FN_FP_tuple in batch_TP_FN_FP_dic[task].items():
                        if tag in total_TP_FN_FP_dic[task].keys():
                            total_TP_FN_FP_dic[task][tag] = tuple(
                                np.array(total_TP_FN_FP_dic[task][tag]) + np.array(TP_FN_FP_tuple))
                        else:
                            total_TP_FN_FP_dic[task][tag] = TP_FN_FP_tuple

                step += 1
                total_loss += joint_loss
                if 5 in eval(args.Task_list):
                    total_loss_5 += loss_person_location

                test_data_count = record_detail_result(file_result, tokens_total, targets_total, predictions_total, self.TOEKNS_TAGS_dic, eval(args.Task_list), batch_num, fold_num, test_data_count)

            dic_reslut = get_epoch_P_R_F(total_TP_FN_FP_dic)

            # dic_reslut[i] = [task12_micro_result, task12_macro_result, task12_TP_FN_FP]
            for key, value in dic_reslut.items():
                P_micro_batch_all_task.append(dic_reslut[key][0][0])
                R_micro_batch_all_task.append(dic_reslut[key][0][1])
                F_micro_batch_all_task.append(dic_reslut[key][0][2])

                P_macro_batch_all_task.append(dic_reslut[key][1][0])
                R_macro_batch_all_task.append(dic_reslut[key][1][1])
                F_macro_batch_all_task.append(dic_reslut[key][1][2])

                if key == 5:
                    task5_P_macro_list.append(dic_reslut[key][1][0])
                    task5_R_macro_list.append(dic_reslut[key][1][1])
                    task5_F_macro_list.append(dic_reslut[key][1][2])

            epoch_P_micro = np.average(P_micro_batch_all_task)
            epoch_R_micro = np.average(R_micro_batch_all_task)
            epoch_F_micro = np.average(F_micro_batch_all_task)

            epoch_P_macro = np.average(P_macro_batch_all_task)
            epoch_R_macro = np.average(R_macro_batch_all_task)
            epoch_F_macro = np.average(F_macro_batch_all_task)

            task5_epoch_P_macro = np.average(task5_P_macro_list)
            task5_epoch_R_macro = np.average(task5_R_macro_list)
            task5_epoch_F_macro = np.average(task5_F_macro_list)

            print()
            print('valid result:   Data_ID: %2d,  Valid_loss: %.3f, P: %.3f, R: %.3f, F: %.3f' %
                  (int(args.Data_ID), total_loss / step, epoch_P_macro, epoch_R_macro, epoch_F_macro))

            if 5 in eval(args.Task_list):
                print("                                                                                         task 5: Loss: %.3f, P: %.3f, R: %.3f, F: %.3f"%
                  (total_loss_5/ step, task5_epoch_P_macro, task5_epoch_R_macro, task5_epoch_F_macro))

                with open(file_traning_performerance, "a") as f:
                    f.writelines('')
                    f.writelines('valid...')
                    f.writelines('Data_ID: %2d, Fold_num: %2d,  Epoch: %2d, Train Loss: %.3f, P: %.3f, R: %.3f, F: %.3f \n' %
                      (int(args.Data_ID), fold_num, epoch, total_loss / step,  task5_epoch_P_macro, task5_epoch_R_macro, task5_epoch_F_macro))

            total_micro_result = [epoch_P_micro, epoch_R_micro, epoch_F_micro]
            total_macro_result = [epoch_P_macro, epoch_R_macro, epoch_F_macro]
            write_dic_performance(file_performance, total_micro_result, total_macro_result, dic_reslut, eval(args.Task_list), fold_num, epoch)

            torch.cuda.empty_cache()
            return epoch_P_macro, epoch_R_macro, epoch_F_macro, \
                   epoch_P_micro, epoch_R_micro, epoch_F_micro, total_loss / step, dic_reslut, split_name_count_total_valid


# batch_leve;
def get_batch_TP_FN_FP(prediction_dic, item_dic, TOEKNS_TAGS_dic, valid_flag=False):

        tokens_total = item_dic[0].tokens
        for_sen_len_list = item_dic[0].tags
        TAGS_for_sen_len_list = TOEKNS_TAGS_dic[0][1]
        TP_FN_FP_dic = {}

        targets_total = {}
        predictions_total = {}

        if 12 in eval(args.Task_list):
            each_kinship_TP_FN_FP = {}
            TAGS_12 = TOEKNS_TAGS_dic[12][1]
            predictions_entity = torch.argmax(prediction_dic[12], 1)
            targets_entity = item_dic[12].tags
            targets_total[12] = targets_entity
            predictions_total[12] = predictions_entity
            assert len(predictions_entity) == len(targets_entity)
        if 3 in eval(args.Task_list):
            each_name_propagation_TP_FN_FP = {}
            TAGS_3 = TOEKNS_TAGS_dic[3][1]
            predictions_name_propagation = torch.argmax(prediction_dic[3], 1)
            targets_name_propagation = item_dic[3].tags
            targets_total[3] = targets_name_propagation
            predictions_total[3] = predictions_name_propagation
            assert len(predictions_name_propagation) == len(targets_name_propagation)
        if 4 in eval(args.Task_list):
            each_parentheses_TP_FN_FP = {}
            TAGS_4 = TOEKNS_TAGS_dic[4][1]
            predictions_parentheses = torch.argmax(prediction_dic[4], 1)
            targets_parentheses = item_dic[4].tags
            targets_total[4] = targets_parentheses
            predictions_total[4] = predictions_parentheses
            assert len(predictions_parentheses) == len(targets_parentheses)
        if 5 in eval(args.Task_list):
            each_person_location_TP_FN_FP = {"L-N": (0, 0, 0)}
            # [task_5_gold_batch, task_5_predict_batch, prediction_5]
            targets_person_location = prediction_dic[5][0]
            task_5_predict_batch_key_value = prediction_dic[5][1]
            targets_total[5] = targets_person_location
            predictions_total[5] = task_5_predict_batch_key_value
            if not valid_flag:
                assert len(task_5_predict_batch_key_value) == len(targets_person_location)

        # get each_kinship_TP_FN_FP
        for sentence in range(tokens_total.shape[0]):
            sentence_length = 0
            for tokens in for_sen_len_list[sentence]:
                if tokens != TAGS_for_sen_len_list.vocab.stoi["<PAD>"]:
                    sentence_length += 1

            # first return is current result, second return is total results
            if 12 in eval(args.Task_list):
                each_kinship_TP_FN_FP = make_dic_performance(
                    targets_entity, predictions_entity, sentence, sentence_length, TAGS_12,
                    each_kinship_TP_FN_FP, args.Improve_Flag, True)
                TP_FN_FP_dic[12] = each_kinship_TP_FN_FP
            if 3 in eval(args.Task_list):
                each_name_propagation_TP_FN_FP = make_dic_performance(
                                targets_name_propagation, predictions_name_propagation, sentence, sentence_length,
                                TAGS_3, each_name_propagation_TP_FN_FP, args.Improve_Flag, False)
                TP_FN_FP_dic[3] = each_name_propagation_TP_FN_FP
            if 4 in eval(args.Task_list):
                each_parentheses_TP_FN_FP = make_dic_performance(targets_parentheses, predictions_parentheses,
                                                                sentence, sentence_length, TAGS_4,
                                                                each_parentheses_TP_FN_FP, args.Improve_Flag, False)
                TP_FN_FP_dic[4] = each_parentheses_TP_FN_FP
            if 5 in eval(args.Task_list):
                try:
                    each_person_location_TP_FN_FP = make_task5_TP_FN_FP_sentence(targets_person_location,
                                                                             task_5_predict_batch_key_value,
                                                                             sentence,
                                                                             each_person_location_TP_FN_FP)
                    TP_FN_FP_dic[5] = each_person_location_TP_FN_FP
                except:
                    # sentence out of boundary
                    pass

        return TP_FN_FP_dic, tokens_total, targets_total, predictions_total


# batch_leve;
def get_epoch_P_R_F(total_TP_FN_FP_dic):
    tasks_result_dic = {}

    if 12 in eval(args.Task_list):
        P_entity_micro, R_entity_micro, F_entity_micro, \
        micro_entity_TP_total, micro_entity_FN_total, micro_entity_FP_total, \
        p_entity_list, r_entity_list, f_entity_list, \
        P_entity_macro, R_entity_macro, F_entity_macro \
            = calculate_PRF(total_TP_FN_FP_dic[12])

        task12_micro_result = [P_entity_micro, R_entity_micro, F_entity_micro]
        task12_macro_result = [P_entity_macro, R_entity_macro, F_entity_macro]
        task12_TP_FN_FP = [micro_entity_TP_total, micro_entity_FN_total, micro_entity_FP_total]

        task12_result_list = [task12_micro_result, task12_macro_result, task12_TP_FN_FP]
        tasks_result_dic[12] = task12_result_list

    if 3 in eval(args.Task_list):
        P_name_propagation_micro, R_name_propagation_micro, F_name_propagation_micro, \
        micro_name_propagation_TP_total, micro_name_propagation_FN_total, micro_name_propagation_FP_total, \
        p_name_propagation_list, r_name_propagation_list, f_name_propagation_list, \
        P_name_propagation_macro, R_name_propagation_macro, F_name_propagation_macro \
            = calculate_PRF(total_TP_FN_FP_dic[3])

        task3_micro_result = [P_name_propagation_micro, R_name_propagation_micro, F_name_propagation_micro]
        task3_macro_result = [P_name_propagation_macro, R_name_propagation_macro, F_name_propagation_macro]
        task3_TP_FN_FP = [micro_name_propagation_TP_total, micro_name_propagation_FN_total,
                          micro_name_propagation_FP_total]

        task3_result_list = [task3_micro_result, task3_macro_result, task3_TP_FN_FP]
        tasks_result_dic[3] = task3_result_list

    if 4 in eval(args.Task_list):
        P_parentheses_micro, R_parentheses_micro, F_parentheses_micro, \
        micro_parentheses_TP_total, micro_parentheses_FN_total, micro_parentheses_FP_total, \
        p_parentheses_list, r_parentheses_list, f_parentheses_list, \
        P_parentheses_macro, R_parentheses_macro, F_parentheses_macro \
            = calculate_PRF(total_TP_FN_FP_dic[4])

        task4_micro_result = [P_parentheses_micro, R_parentheses_micro, F_parentheses_micro]
        task4_macro_result = [P_parentheses_macro, R_parentheses_macro, F_parentheses_macro]
        task4_TP_FN_FP = [micro_parentheses_TP_total, micro_parentheses_FN_total, micro_parentheses_FP_total]

        task4_result_list = [task4_micro_result, task4_macro_result, task4_TP_FN_FP]
        tasks_result_dic[4] = task4_result_list

    if 5 in eval(args.Task_list):

        if total_TP_FN_FP_dic[5] == {'L-N': (0, 0, 0)}:
            task5_micro_result = [np.nan, np.nan, np.nan]
            task5_macro_result = [np.nan, np.nan, np.nan]
            task5_TP_FN_FP = [np.nan, np.nan, np.nan]
        else:
            P_Name_Location_micro, R_Name_Location_micro, F_Name_Location_micro, \
            micro_Name_Location_TP_total, micro_Name_Location_FN_total, micro_Name_Location_FP_total, \
            p_Name_Location_list, r_Name_Location_list, f_Name_Location_list, \
            P_Name_Location_macro, R_Name_Location_macro, F_Name_Location_macro \
                = calculate_PRF(total_TP_FN_FP_dic[5])

            task5_micro_result = [P_Name_Location_micro, R_Name_Location_micro, F_Name_Location_micro]
            task5_macro_result = [P_Name_Location_macro, R_Name_Location_macro, F_Name_Location_macro]
            task5_TP_FN_FP = [micro_Name_Location_TP_total, micro_Name_Location_FN_total, micro_Name_Location_FP_total]

        task5_result_list = [task5_micro_result, task5_macro_result, task5_TP_FN_FP]
        tasks_result_dic[5] = task5_result_list

    return tasks_result_dic


def get_loss(IF_Weight_loss, IF_Adap_loss, Weight_Lambda, targets, TOEKNS_TAGS_dic, TAGS_5_NEW, entity_flag=False,
             name_propagation_flag=False, parentheses_flag=False, Location_person_flag=False):
    # Tag O's index = 1
    if entity_flag:
        TAGS = TOEKNS_TAGS_dic[12][1].vocab.stoi
        relation_tag_weight = args.Entity_tag_Weight
    elif name_propagation_flag:
        TAGS = TOEKNS_TAGS_dic[3][1].vocab.stoi
        relation_tag_weight = args.Name_propagation_tag_Weight
    elif parentheses_flag:
        TAGS = TOEKNS_TAGS_dic[4][1].vocab.stoi
        relation_tag_weight = args.Parentheses_tag_Weight
    elif Location_person_flag:
        TAGS = TAGS_5_NEW
        relation_tag_weight = args.Person_location_tag_Weight
    else:
        raise Exception("No usb-task chosen !")

    if IF_Weight_loss:
        # auto weights or fix weights
        if IF_Adap_loss:
            if Location_person_flag:
                relation_tag_weight_tensor = get_task5_adaptive_weight_tensor(targets, TAGS)
            else:
                relation_tag_weight_tensor = get_adaptive_weight_tensor(targets, TAGS, Weight_Lambda)

            criterion = nn.CrossEntropyLoss(ignore_index=TAGS['<PAD>'],
                                            weight=relation_tag_weight_tensor)
        else:
            list_weight = [relation_tag_weight] * len(TAGS)
            list_weight[TAGS['O']] = 1
            list_weight[TAGS['<PAD>']] = 1
            relation_tag_weight_tensor = torch.Tensor(list_weight).cuda()
            criterion = nn.CrossEntropyLoss(ignore_index=TAGS['<PAD>'],
                                            weight=relation_tag_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=TAGS['<PAD>'])

    return criterion


def get_adaptive_weight_tensor(targets, TAGS, Weight_Lambda):
    class_dic = dict(Counter([int(i) for i in list(targets.view(-1))]))
    total_num = 0
    init_num = 0
    for key, values in class_dic.items():
        if key != TAGS["<PAD>"]:
            total_num += values
            if key != TAGS["O"]:
                init_num += values

    denominator = (len(class_dic) - 2) if (len(class_dic) - 2) != 0 else 1
    list_weight0 = [init_num / denominator] * len(TAGS)
    for key, values in class_dic.items():
        weight = total_num / class_dic[key]
        list_weight0[key] = weight

    list_weight1 = [i / list_weight0[TAGS["O"]] for i in list_weight0]
    list_weight2 = [i * Weight_Lambda for i in list_weight1]

    list_weight2[TAGS["<PAD>"]] = 1
    list_weight2[TAGS["O"]] = 1
    relation_tag_weight_tensor = torch.Tensor(list_weight2).cuda()
    return relation_tag_weight_tensor


def get_task5_adaptive_weight_tensor(targets, TAGS):
    class_dic = dict(Counter(sum(targets.tolist(), [])))
    if TAGS["yes"] not in class_dic.keys():
        return torch.Tensor([1] * len(TAGS)).cuda()

    total_num = 0
    yes_num = 0
    no_num = 0
    for key, values in class_dic.items():
        if key != TAGS["<PAD>"]:
            total_num += values
            if key == TAGS["yes"]:
                yes_num += values
            elif key == TAGS["no"]:
                no_num += values
            else:
                raise Exception("get_adaptive_weight_tensor wrong")

    list_weight = [1] * len(TAGS)
    max_num = max(no_num, yes_num)
    yes_weight = (max_num / yes_num) if yes_num != 0 else 1
    no_weight = (max_num / no_num) if no_num != 0 else 1

    list_weight[TAGS["<PAD>"]] = 1
    list_weight[TAGS["yes"]] = yes_weight * args.Weight_Lambda_5
    list_weight[TAGS["no"]] = no_weight * args.Weight_Lambda_5
    relation_tag_weight_tensor = torch.Tensor(list_weight).cuda()
    return relation_tag_weight_tensor


def print_execute_time(func):
    if type(func) is tuple:
        opera_time = (func[1] - func[0])
        if opera_time > 60:
            if opera_time / 60 > 60:
                opera_time = round(opera_time / 3600, 3)
                print(f'execute time: {opera_time} hour')
            else:
                opera_time = round(opera_time / 60, 3)
                print(f'execute time: {opera_time} minute')
        else:
            print(f'execute time: {round(opera_time, 3)} s')
    else:
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            opera_time = (end - start)
            if opera_time > 60:
                if opera_time/ 60 > 60:
                    opera_time = round(opera_time / 3600, 3)
                    print(f'execute time: {opera_time} hour')
                else:
                    opera_time = round(opera_time / 60, 3)
                    print(f'execute time: {opera_time} minute')
            else:
                print(f'execute time: {round(opera_time,3)} s')
        return wrapper

@print_execute_time
def get_valid_performance():

    final_result_list_P_macro = []
    final_result_list_R_macro = []
    final_result_list_F_macro = []
    final_result_list_P_micro = []
    final_result_list_R_micro = []
    final_result_list_F_micro = []
    each_fold_average_list = []

    for i in range(args.NUM_FOLD):  # NUM_FOLD = 10
        print("==========================" + str(i) + "=================================================")
        # In this paper, we take augment data rather simply_num
        start = time.clock()

        TOEKNS_TAGS_dic = {}
        dic_init = {}
        train_test_iterator_dic = {}

        establish_file(i, args.Simply_num, args.Data_ID, args.My_Test_data_flag)

        train_set_0, test_set_0, TOEKNS_0, TAGS_0 = prepared_data(args.Data_ID,
                                                                      args.Word_embedding_size,
                                                                      test_50_embeding_flag=args.Test_50_embeding_flag)
        train_iterator_0, test_iterator_0, TOEKNS_0, TAGS_0 = filter_content(
            train_set_0, test_set_0, TOEKNS_0, TAGS_0, args.BATCH_SIZE, device,
            entity_only_flag=True)

        TOEKNS_TAGS_dic[0] = (TOEKNS_0, TAGS_0)
        train_test_iterator_dic[0] = (train_iterator_0, test_iterator_0)

        if 12 in eval(args.Task_list):
            train_set_12, test_set_12, TOEKNS_12, TAGS_12 = prepared_data(args.Data_ID,
                                                                          args.Word_embedding_size,
                                                                          test_50_embeding_flag=args.Test_50_embeding_flag)
            train_iterator_12, test_iterator_12, TOEKNS_12, TAGS_12 = filter_content(
                train_set_12, test_set_12, TOEKNS_12, TAGS_12, args.BATCH_SIZE, device,
                entity_only_flag=True)

            dic_init[12] = args.Hidden_Size_Common_Encoder*2, len(TAGS_12.vocab)
            TOEKNS_TAGS_dic[12] = (TOEKNS_12, TAGS_12)
            train_test_iterator_dic[12] = (train_iterator_12, test_iterator_12)

        if 3 in eval(args.Task_list):
            train_set_3, test_set_3, TOEKNS_3, TAGS_3 = prepared_data(args.Data_ID,
                                                                      args.Word_embedding_size,
                                                                      test_50_embeding_flag=args.Test_50_embeding_flag)
            train_iterator_3, test_iterator_3, TOEKNS_3, TAGS_3 = filter_content(
                train_set_3, test_set_3, TOEKNS_3, TAGS_3, args.BATCH_SIZE, device,
                name_propagation_flag=True)

            dic_init[3] = args.Hidden_Size_Common_Encoder*2, len(TAGS_3.vocab)
            TOEKNS_TAGS_dic[3] = (TOEKNS_3, TAGS_3)
            train_test_iterator_dic[3] = (train_iterator_3, test_iterator_3)

        if 4 in eval(args.Task_list):
            train_set_4, test_set_4, TOEKNS_4, TAGS_4 = prepared_data(args.Data_ID,
                                                                      args.Word_embedding_size,
                                                                      test_50_embeding_flag=args.Test_50_embeding_flag)
            train_iterator_4, test_iterator_4, TOEKNS_4, TAGS_3= filter_content(
                train_set_4, test_set_4, TOEKNS_4, TAGS_4, args.BATCH_SIZE, device,
                parentheses_flag=True)

            dic_init[4] = args.Hidden_Size_Common_Encoder*2, len(TAGS_4.vocab)
            TOEKNS_TAGS_dic[4] = (TOEKNS_4, TAGS_4)
            train_test_iterator_dic[4] = (train_iterator_4, test_iterator_4)

        if 5 in eval(args.Task_list):
            TAGS_5_NEW = {"no": 0, "yes": 1, "<PAD>": 2}
        else:
            TAGS_5_NEW = {}

        if 5 in eval(args.Task_list):
            train_set_5, test_set_5, TOEKNS_5, TAGS_5 = prepared_data(args.Data_ID,
                                                                      args.Word_embedding_size,
                                                                      test_50_embeding_flag=args.Test_50_embeding_flag)
            train_iterator_5, test_iterator_5, TOEKNS_5, TAGS_5 = filter_content(
                train_set_5, test_set_5, TOEKNS_5, TAGS_5, args.BATCH_SIZE, device,
                person_location_flag=True)

            dic_init[5] = args.Hidden_Size_Common_Encoder*2*3, len(TAGS_5_NEW)
            TOEKNS_TAGS_dic[5] = (TOEKNS_5, TAGS_5)
            train_test_iterator_dic[5] = (train_iterator_5, test_iterator_5)

        joint_train = Train_joint_model(dic_init, TAGS_5_NEW, train_test_iterator_dic, TOEKNS_TAGS_dic)
        stop_epoch, split_name_count_total_train = joint_train.train_model(i)

        valid_joint_model = Valid_joint_model(dic_init, TAGS_5_NEW, train_test_iterator_dic, TOEKNS_TAGS_dic)
        maxP_macro, maxR_macro, maxF_macro, maxP_micro, maxR_micro, maxF_micro, total_loss, tasks_result_dic, split_name_count_total_valid \
            = valid_joint_model.valid_model(i, stop_epoch)

        split_name_count_total = split_name_count_total_train + split_name_count_total_valid
        print("split_name_count_total", split_name_count_total)
        each_fold_average_list.append(tasks_result_dic)

        final_result_list_P_macro.append(maxP_macro)
        final_result_list_R_macro.append(maxR_macro)
        final_result_list_F_macro.append(maxF_macro)

        final_result_list_P_micro.append(maxP_micro)
        final_result_list_R_micro.append(maxR_micro)
        final_result_list_F_micro.append(maxF_micro)

        end = time.clock()
        print_execute_time((start, end))

    record_each_performance(file_performance_total, final_result_list_P_macro, final_result_list_R_macro, final_result_list_F_macro,
                            final_result_list_P_micro, final_result_list_R_micro, final_result_list_F_micro,
                            each_fold_average_list, eval(args.Task_list), "LSTM")


if __name__ == "__main__":
    get_valid_performance()




