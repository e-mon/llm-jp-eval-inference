from pathlib import Path

from pydantic import BaseModel, Field

from llm_jp_eval.schemas import WandBConfig

class DatasetProfile(BaseModel):
   num_prompt: int
   max_input_len: int
   sum_input_len: int
   max_output_len: int
   max_seq_len: int
   sum_seq_len: int


class InferenceResultInfo(BaseModel):
    dump_prompts_config: dict
    dataset_profile: dict[str, DatasetProfile]
    benchmark: dict
    time_profile: dict

# these values will be used only for logging and wandb (no effect for running environment)
class MetaInfo(BaseModel):
    base_model_name: str | None = None
    quantization: str | None = None
    lib_type: str | None = None
    gpu_type: str | None = None
    run_name_suffix: str | None = None
    model_conversion_time: float = 0.0

class BaseInferenceConfig(BaseModel):
    wandb: WandBConfig = WandBConfig()
    run: MetaInfo = MetaInfo()
    inference_result_info: InferenceResultInfo | None = None
    tokenize_kwargs: dict = Field({'add_special_tokens': True}, description="additional arguments for tokenizer.tokenize")

    prompt_json_path: str | list[str] = Field(["../../../dataset/1.4.1/evaluation/test/prompts/*.eval-prompt.json"], description="path to the prompt json file")
    output_base_dir: Path = Field("outputs", description="output directory")
    # If null, the top level run_name or the default run_name determined by model configuration will be used. 
    # Do not specify both this value and top level run_name at the same time. The final output path is {output_base_dir}/{run_name}/
    run_name : str | None = Field(None, description="run_name for logging")
