import collections
import os
import random
import re
import json
import sys
import logging
import time
import argparse
import random
import sglang as sgl
from model import SGLangModel
from tqdm import tqdm, trange
from typing import Dict, Any, Optional, Union, List, Tuple
from prompt import prompt_template_dict
from utils import compute_accuracy,write_result
from dataset import read_dataset_label,generate_Alpaca_Dataset

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Script Arguments")
    
    parser.add_argument('--input_path', type=str, default='./data/train_QC.txt',
                        help='Path to the input file')
    parser.add_argument('--output_path', type=str, default='./myresults/train_QC1.txt',
                        help='Path to save the output file')
    parser.add_argument('--model_path', type=str, default='./model/Qwen1.5-0.5B',
                        help='Path to the model directory')
    parser.add_argument('--template_name', type=str, default='prompt_template_qc',
                        help='Name of the prompt template to use')

    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--tp_size', type=int, default=1,
                        help='Tensor parallelism size')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling value')
    """
        filter_0: 
                use Qwen2.5-0.5B/1.5B
                no filter, directly use the model to predict.
                if the prediction is True, indicating that it's easy and throw away it
                the rest are referred as data-1
        filter_1: 
                Qwen2.5-70B
                predict 8 times, if at least 1 time is True,the data is a correct data so that it can be remained.
                these are data-2
        filter_2:
                a more powerful model
                use the same way to filter,the correct data are referred as data-3
        filter_999:
                评估模型专用
    """
    parser.add_argument('--filter_i', type=int, default=0)
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()


    input_path = args.input_path
    output_path = args.output_path
    template_name = args.template_name
    
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = SGLangModel(
        model_name_or_path=args.model_path,
        context_length=30000,
        max_new_tokens=args.max_new_tokens,
        tp_size=args.tp_size
    )

    data , prompts , labels = read_dataset_label(
        input_path=input_path,
        template_name=template_name,
        prompt_template_dict=prompt_template_dict
    ) 
    
    
    
    
    # utilize the model to predict directly without filter
    if args.filter_i == 0:
        outputs = model.generate(
            prompts,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        new_outputs = []
        remaining_ids = []
        filtered_dataset1 = []
        for d, o, label in zip(data, outputs, labels):
            # 提取 prediction
            prediction = 1 if 'True' in o.split("</think>")[-1] else 0

            # 构造 new_output
            new_outputs.append({
                'id': d['id'],
                'prediction': prediction
            })

            # 如果 prediction 不正确,加入 remaining_ids,正确的视为EASY然后扔掉
            if prediction != label:
                remaining_ids.append(d['id'])
                filtered_dataset1.append(d)
        
        #compute_accuracy(new_outputs, labels)
        # to generate dataset-1
        write_result("./filter_dataset/dataset-1.txt", filtered_dataset1)
        
        
        write_result(output_path, new_outputs) # the prediction results
        
    elif args.filter_i == 1:
        
        # hash
        id2data = {d['id']: d for d in data}
        
        outputs = []
        for _ in range(8):
            # temp: a prediction
            temp = []
            temp_outputs = model.generate(
                        prompts,
                        temperature=args.temperature,
                        top_p=args.top_p
                )
            
            for d, o in zip(data, temp_outputs):
                temp.append(
                    {
                        'id': d['id'],
                        'prediction': 1 if 'True' in o.split("</think>")[-1] else 0
                    }
                )
            # every item is a prediction , total 8 predictions
            outputs.append(temp)
        
        # 构造 filtered_dataset2
        """
        对2过滤后的训练集,用Qwen2.5-70B,
        每个回答8次,如果能答对一次,就表明数据是正确的,
        可以保留下来数据,作为数据-2
        """
        filtered_dataset2 = []
        """
        其余答不对的数据,用更强的模型去对它验证,如果更强的模型能答对,这个作为数据-3
        filtered_dataset2_3就是答不对的数据,后续会用更强的模型验证它
        """
        filtered_dataset2_3 = []
        for preds,label in zip(zip(*outputs),labels):
            if any(p['prediction'] == label for p in preds):
                sample_id = preds[0]['id']
                filtered_dataset2.append(id2data[sample_id])
            else:
                sample_id = preds[0]['id']
                filtered_dataset2_3.append(id2data[sample_id])
        # generate dataset-2
        write_result("./filter_dataset/dataset-2.txt", filtered_dataset2)
        write_result("./filter_dataset/dataset-2_3.txt",filtered_dataset2_3)
    elif args.filter_i == 2:
        
        # hash
        id2data = {d['id']: d for d in data}
        
        outputs = []
        for _ in range(8):
            # temp: a prediction
            temp = []
            temp_outputs = model.generate(
                        prompts,
                        temperature=args.temperature,
                        top_p=args.top_p
                )
            
            for d, o in zip(data, temp_outputs):
                temp.append(
                    {
                        'id': d['id'],
                        'prediction': 1 if 'True' in o.split("</think>")[-1] else 0
                    }
                )
            # every item is a prediction , total 8 predictions
            outputs.append(temp)
        
        # 构造 filtered_dataset3
        filtered_dataset3 = []
        for preds,label in zip(zip(*outputs),labels):
            if any(p['prediction'] == label for p in preds):
                sample_id = preds[0]['id']
                filtered_dataset3.append(id2data[sample_id])
        # generate dataset-3
        write_result("./filter_dataset/dataset-3.txt", filtered_dataset3)
    elif args.filter_i == 999:
        outputs = model.generate(
            prompts,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        new_outputs = []

        for d, o, label in zip(data, outputs, labels):
            # 提取 prediction
            prediction = 1 if 'True' in o.split("</think>")[-1] else 0

            # 构造 new_output
            new_outputs.append({
                'id': d['id'],
                'prediction': prediction
            })

       
        accuracy = compute_accuracy(new_outputs, labels)
        print(accuracy)
        
        write_result(output_path, new_outputs) # the prediction results
    
    
    
    
