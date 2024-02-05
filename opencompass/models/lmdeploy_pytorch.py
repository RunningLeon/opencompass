from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


class LmdeployPytorchModel(BaseModel):
    """Model wrapper for lmdeploy pytorch engine through python API.

    Args:
        path (str): path of the supported pytorch model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        engine_config (Dict, optional): The engine config to set
            arguments like session_len, max_batch_size for TurboMind.
        gen_config (Dict, optional): Generation config to set
                arguments like top_k, top_p, temperature.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.
    """

    def __init__(self,
                 path: str,
                 concurrency: int = 8,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 engine_config: Optional[Dict] = None,
                 gen_config: Optional[Dict] = None,
                 end_str: Optional[str] = None):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template)
        from lmdeploy.pytorch import engine as tm
        from transformers import AutoTokenizer

        if engine_config is not None:
            from lmdeploy.messages import PytorchEngineConfig
            engine_config = PytorchEngineConfig(**engine_config)
        if gen_config is not None:
            from lmdeploy.messages import EngineGenerationConfig
            gen_config = EngineGenerationConfig(**gen_config)

        self.logger = get_logger()
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        tm_model = tm.Engine(path, engine_config)
        self.generators = [
            tm_model.create_instance() for i in range(concurrency)
        ]
        self.generator_ids = [i + 1 for i in range(concurrency)]
        self.gen_config = gen_config
        self.end_str = end_str

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of prompts
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        assert isinstance(
            inputs, List), f'List(str) is expected, but got {type(inputs)}'

        # split inputs into batches
        batch_size = len(self.generators)
        batch_inputs = [
            inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)
        ]

        results = []
        for batch_input in batch_inputs:
            with ThreadPoolExecutor() as executor:
                _results = list(
                    executor.map(
                        self._generate,
                        self.generators[:len(batch_input)],
                        self.generator_ids[:len(batch_input)],
                        batch_input,
                        [max_out_len] * len(batch_input),
                        [self.gen_config] * len(batch_input),
                        [self.end_str] * len(batch_input),
                    ))
                results += _results
        return results

    def get_token_len(self, prompt: str) -> int:
        input_ids = self.tokenizer.encode(prompt)
        return len(input_ids)

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        return self.token_bucket.get_token()

    def _generate(self,
                  generator,
                  session_id,
                  prompt: str or PromptList,
                  max_out_len: int,
                  gen_config=None,
                  end_str: Optional[str] = None) -> str:
        """Generate results given a list of inputs.

        Args:
            prompt (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.
            max_out_len (int): The maximum length of the output.
            gen_config (EngineGenerationConfig, optional): Generation
                config to set arguments like top_k, top_p, temperature.
            end_str (str, optional): Whether to trim generated strings
                with end_str if the model has special ending strings
                that are not handled well.
                Defaults to None.
        Returns:
            str: The generated string.
        """
        assert type(
            prompt) is str, 'We only support string for TurboMind Python API'
        input_ids = self.tokenizer.encode(prompt)
        response_size = 0

        for outputs in generator.stream_infer(session_id=session_id,
                                              input_ids=input_ids,
                                              gen_config=gen_config,
                                              request_output_len=max_out_len,
                                              step=0):
            status, res, tokens = outputs
            response_all = self.tokenizer.decode(res)
            response_cur = response_all[response_size:]
            response_all = valid_str(response_all)
            response_size += len(response_cur)
        if hasattr(generator, 'end'):
            generator.end(session_id)
        return response_all
