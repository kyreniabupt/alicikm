from tqdm import tqdm, trange
import sglang as sgl
from typing import Dict, Any, Optional, Union, List, Tuple


class SGLangModel:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 200,
        tp_size: int = 1,
        dp_size: int = 1,
        context_length: int = 2048,
        random_seed: int = 42,
        batch_size: int = 1

    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.context_length = context_length
        self.random_seed = random_seed
        self.max_prompt_length = context_length - max_new_tokens - 128

        self.model = sgl.Engine(
            model_path=model_name_or_path,
            context_length=context_length,
            tp_size=tp_size,
            dp_size=dp_size,
            random_seed=random_seed,
            mem_fraction_static=0.7
        )
        self.tokenizer = self.model.tokenizer_manager.tokenizer
        self.tokenizer.padding_side = "left"

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.2,
        top_p: float = 0.9,
        dataset_i: int = 0
    ):  
        
            
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            # "top_k": 40,
            "repetition_penalty": 1.0,
            "max_new_tokens": self.max_new_tokens,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        """
        message:
                [
            [{"role": "user", "content": "你是谁？"}],
            [{"role": "user", "content": "今天天气怎么样？"}]
        ]

        """
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        """
            input_texts:
                [
                "<|user|>你是谁？<|assistant|>",
                "<|user|>今天天气怎么样？<|assistant|>"
            ]

        """
        input_texts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = []
        batch_size = self.batch_size
        for idx in trange(0, len(input_texts), batch_size):
            """
                    self.model.generate 每次返回:
                    [
                        {
                            'text': '你好，我是一个语言模型。',
                            'usage': {
                                'prompt_tokens': 27,
                                'completion_tokens': 14,
                                'total_tokens': 41
                            }
                        },
                        {
                            'text': '今天天气晴朗，适合出门。',
                            'usage': {
                                'prompt_tokens': 30,
                                'completion_tokens': 15,
                                'total_tokens': 45
                            }
                        },
                        
                    ]

                """
            
            print(f"Processing batch {idx // batch_size + 1} of {len(input_texts) // batch_size}")
            
            outputs.extend(
                self.model.generate(
                        input_texts[idx: min(len(input_texts), idx + batch_size)],
                        sampling_params=sampling_params,
                    )
            )
            
                
                
            #print(outputs[0])
        return [e['text'] for e in outputs]