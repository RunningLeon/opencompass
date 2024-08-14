from opencompass.models import LongCacheCausalLM

models = [
    dict(
        type=LongCacheCausalLM,
        abbr='internlm2_5-7b-chat-1m-longcache',
        path='internlm/internlm2_5-7b-chat-1m',
        model_type='internlm2',
        attn_implementation='flash_attention_2',
        max_out_len=2048,
        tokenizer_path='internlm/internlm2_5-7b-chat-1m',
        run_cfg=dict(num_gpus=4),
        cache_config = dict(
                    global_size= 32, mid_size= 1, span_size= 32, 
                    local_size=4096, chunk_size= 512, 
                    rope_scaling= None, recall_option= 'default', 
                    unique_option= 'group_unique', recall_clip= 128,
                    ),
    )
]

