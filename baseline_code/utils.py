from typing import List, Dict, Any, Tuple
import json
import random

def compute_accuracy(outputs: List[Dict[str, Any]], labels: List[int]) -> float:
    correct = 0
    for output, label in zip(outputs, labels):
        if output['prediction'] == label:
            correct += 1
    return correct / len(labels) if labels else 0.0





def write_result(output_path: str, outputs: List[Dict[str, Any]]):
    with open(output_path, 'w', encoding="utf-8") as f:
        for o in outputs:
            # 不加ensure_ascii会出现莫名其妙的字符
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
            