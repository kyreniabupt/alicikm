from typing import List, Dict, Any, Tuple
import json
import random
from prompt import instruction_template_dict, prompt_template_dict


def read_dataset_label(input_path: str, 
                       template_name: str, 
                       prompt_template_dict: Dict[str, str],
                       ) -> Tuple[List[str], List[int]]:
    data = []
    prompts = []
    labels = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))
            if 'qc' in template_name:
                prompts.append(
                    prompt_template_dict[template_name].format(
                        query=data[-1]['origin_query'],
                        category=data[-1]['category_path']
                    )
                )
                labels.append(data[-1]['label'])    
            else:
                prompts.append(
                    prompt_template_dict[template_name].format(
                        query=data[-1]['origin_query'],
                        product=data[-1]['item_title']
                    )
                )
                labels.append(data[-1]['label'])
    """ sample_indices = random.sample(range(len(data)),5)
    data = [data[i] for i in sample_indices]
    prompts = [prompts[i] for i in sample_indices]
    labels = [labels[i] for i in sample_indices] """
    return data, prompts, labels


def generate_Alpaca_Dataset(instruction_template_dict: Dict[str, str], 
                            template_name: str, 
                            data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alpaca_data = []
    for d in data:
        if 'qc' in template_name:
            alpaca_data.append({
                'instruction': instruction_template_dict[template_name],
                'input': f"Query: {d['origin_query']}\n category_path: {d['category_path']}",
                'output': "True" if d['label'] else "False"
            })
        else:
            alpaca_data.append({
                'instruction': instruction_template_dict[template_name],
                'input': f"Query: {d['origin_query']}\n Item Title: {d['item_title']}",
                'output': "True" if d['label'] else "False"
            })
    # 写入 JSON 文件
    if 'qc' in template_name:
        with open('./Alpaca_data/CIKM_qc_demo.json', 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    else:
        with open('./Alpaca_data/CIKM_qi_demo.json.json', 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    # test
    input_path = "filter_dataset/dataset-2.txt"  
    prompt_template_name = "prompt_template_qc"  
    instruct_template_name = "instruction_template_qc"  
    data, prompts, labels = read_dataset_label(input_path, prompt_template_name,prompt_template_dict)
    #print(f"Data: {data[:5]}")
    #print(f"Prompts: {prompts[:5]}")
    #print(f"Labels: {labels[:5]}")
    
    generate_Alpaca_Dataset(instruction_template_dict, instruct_template_name, data)
    print("Alpaca dataset generated successfully.")