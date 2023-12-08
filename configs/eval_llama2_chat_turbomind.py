from mmengine.config import read_base
from opencompass.models.turbomind import TurboMindModel


with read_base():
    # choose a list of datasets
    #from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    #from .datasets.agieval.agieval_gen import agieval_datasets
    #from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    #from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    #from .datasets.humaneval.humaneval_gen import humaneval_datasets
    # and output the results in a choosen format
    from .summarizers.medium import summarizer

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])


begin = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""  # noqa: E501

meta_template = dict(
            begin='',
            end='',
            round=[
                    dict(role='HUMAN', begin='[INST] ', end='[/INST] '),
                    dict(role='BOT', generate=True),
            ],
            #reserved_roles=[dict(role='SYSTEM', begin='<<SYS>>\n', end='\n<</SYS>>\n\n'),],
            eos_token_id=2,
         )

# config for internlm-chat-7b
models = [
    dict(
        type=TurboMindModel,
        abbr='llama2-chat-7b-turbomind',
        path="/workspace/models/llama2-7b-chat-tb",
        max_out_len=512,
        max_seq_len=2048,
        batch_size=16,
        concurrency=32,
        meta_template=meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
