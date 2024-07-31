from typing import Optional, Dict, Sequence
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class RAHFArguments:
    user_tag: str = field(default="Human: ",metadata={"help": "User tag for chat models (eg: `USER:` or `[INST]`)"})
    assistant_tag: str = field(default="Assistant: ",metadata={"help": "Assistant tag for chat models (eg: `ASSISTANT:` or `[\INST]`)"})
    ori_type: str = field(default="ori",metadata={"help": "origin type"})
    pos_type: str = field(default="good",metadata={"help": "Concept/Function to be optimized towards (eg: 'a truthful')"})
    neg_type: str = field(default="bad",metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"})
    target_layers: str = field(default="10,12,14,16,18,20",metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    control_template: str = field(default="Give a {type} answer",metadata={"help": "Control template for Representation setting (eg: Give a {type} answer)"})
    rahf_alpha: float = field(default=5, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # RAHF Hyperparameters
    rahf_beta: float = field(default=0, metadata={"help": "vice versa of pos_type (eg: 'an untruthful')"}) # RAHF Hyperparameters
    max_res_len: int = field(default=64, metadata={"help": "truncated length for getting generated ouputs from RAHF pos/neg exampels"}) # RAHF Hyperparameters
    data_type: str = field(default="all", metadata={"help": ""}) 
    dataset_name:str = field(default="", metadata={"help": ""}) 
    data_path:str = field(default=None)
    

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class ModelArguments:
    # If the model is trained using ft
    model_name_or_path: Optional[str] = field(default=None)
    model_good_name_or_path: str = field(default=None)
    model_bad_name_or_path: str = field(default=None)
    model_base_name_or_path: str = field(default=None)

    # If the model is trained using lora
    model_lora_path: str = field(default=None)
    model_good_lora_path: str = field(default=None)
    model_bad_lora_path: str = field(default=None)
    model_base_lora_path: str = field(default=None)

    load_in_8bit: bool = field(default=False)
    adapter_name_or_path: str = field (
        default=None, metadata={"help": "Adapater name"}
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Use LoRA (default: False)"}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    data_seed: int = field(default=42)
    method:str = field(
        default=None,
         metadata={"help": "DUAL or SCIT"}
        )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    grouped_to_max_length: bool = field (
        default=False, metadata={"help": "Group to chunks of max length for pretraining"}
    )