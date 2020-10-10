# formal experiment

#python joint_model_bert.py  --Data_ID=3  --GPU=1  --NUM_FOLD=10 --Task_list="[12]"  --Loss_weight_list="[1]"
#python joint_model_bert.py  --Data_ID=2  --GPU=3  --NUM_FOLD=10 --Task_list="[12,3]"  --Loss_weight_list="[0.6, 0.4]"
nohup python joint_model_bert.py  --Data_ID=0  --GPU=1  --NUM_FOLD=10 --Task_list="[12,3,4]"  --Loss_weight_list="[0.6, 0.2, 0.2]"  >Task_list=[12,3,4] 2>&1 &
#python joint_model_bert.py  --Data_ID=0  --GPU=0  --Task_list="[12,3,4,5]"  --Loss_weight_list="[0.6, 0.2, 0.2, 0.5]"
nohup python joint_model_bert.py  --Data_ID=2  --GPU=2  --Task_list="[3]"  --Loss_weight_list="[1]" >Task_list=[3] 2>&1 &
#python joint_model_bert.py  --Data_ID=3  --GPU=3  --Task_list="[4]"  --Loss_weight_list="[1]"
#python joint_model_bert.py  --Data_ID=3  --GPU=0  --Task_list="[12, 5]"  --Loss_weight_list="[0.6, 0.4]"


#nohup python joint_model_bert.py  --Data_ID=1  --GPU=0  --NUM_FOLD=10  --Augment_Data="BIES_bert_base_numberic_agument_data_0.2_200.json" >agument_data_0.2 2>&1 &
#nohup python joint_model_bert.py  --Data_ID=2  --GPU=2  --NUM_FOLD=10 --Augment_Data="BIES_bert_base_numberic_agument_data_0.3_200.json" >agument_data_0.3 2>&1 &
#nohup python joint_model_bert.py  --Data_ID=3  --GPU=2  --NUM_FOLD=10  --Augment_Data="BIES_bert_base_numberic_agument_data_0.4_200.json" >agument_data_0.4 2>&1 &
#nohup python joint_model_bert.py  --Data_ID=4  --GPU=3  --NUM_FOLD=10  --Augment_Data="BIES_bert_base_numberic_agument_data_0.5_200.json" >agument_data_0.5 2>&1 &
#nohup python joint_model_bert.py  --Data_ID=5  --GPU=1  --NUM_FOLD=10  --Augment_Data="BIES_bert_base_numberic_agument_data_0.6_200.json" >agument_data_0.6 2>&1 &