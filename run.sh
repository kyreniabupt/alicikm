cd baseline_code
# 用1中选好的prompt （Qwen2.5-0.5B/1.5B）过一遍训练集，
# 如果能直接答对的话，这个训练数据是简单的，就可以扔掉了
# qc
python run.py --input_path ./data/train_QC.txt \
               --output_path ./myresults/train_QC1.txt \
               --model_path ./model/Qwen2.5-1.5B \
               --template_name prompt_template_qc \
               --filter_i 0



# 对2过滤后的训练集，用Qwen2.5-70B，每个回答8次，如果能答对一次，就表明数据是正确的，可以保留下来数据，作为数据-2
python run.py --input_path ./filter_dataset/dataset-1.txt \
               --output_path ./myresults/train_QC2.txt \
               --model_path ./model/Qwen2.5-70B \
               --template_name prompt_template_qc \
               --filter_i 1


#其余答不对的数据，用更强的模型去对它验证，如果更强的模型能答对，这个作为数据-3
python run.py --input_path ./filter_dataset/dataset-2_3.txt \
               --output_path ./myresults/train_QC3.txt \
               --model_path ./model/Qwen1.5-0.5B \
               --template_name prompt_template_qc \
               --filter_i 2


# SFT之前先构造Alpaca格式数据,这个是对dataset-2处理的
python dataset.py
# 然后将生成的CIKM_qc_demo.json文件，移入llamma_factory中的data文件夹下，
# 我已经在dataset_info.json里register这个名字了,生成后只需移入上面说的这个json文件

#SFT 命令,这个参数我调的很小,为了在我电脑上跑通
cd LLAMA-FACTORY
llamafactory-cli train examples/train_lora/my_lora_sft.yaml


# 评估最终模型
python run.py --input_path ./data/dev_QC.txt \
               --output_path ./myresults/final_result_QC.txt \
               --model_path ./model/Qwen2.5-1.5B \
               --template_name prompt_template_qc \
               --filter_i 999