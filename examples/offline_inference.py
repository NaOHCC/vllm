import random
import time
from typing import cast

import torch
import torch.cuda as cuda

from examples.comm import *

from vllm import LLM, SamplingParams
from vllm.executor.gpu_executor import GPUExecutor
from vllm.worker.worker import Worker

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]

# prompts = [{"prompt_token_ids": [random.randint(1, 1000) for _ in range(1000)]}]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-1.3b")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
executor = cast(GPUExecutor, llm.llm_engine.model_executor)
worker = cast(Worker, executor.driver_worker)
model = worker.model_runner.model
model = model.cpu()

configs = getLayerConfig(
    len(travel_layers(model)),
    mode=TRANSFER_MODE.Interleave,
    directHostAccessLayerList=[],
)

TransferEngine.init()
te = TransferEngine(model, configs)

stream = torch.cuda.Stream(device=0)


def f():
    prompt_token_ids = [
        [random.randint(1, 10000) for _ in range(1000)] for _ in range(4)
    ]
    now = time.perf_counter()

    te.schedule()
    with torch.cuda.stream(cast(cuda.Stream, stream)):
        outputs = llm.generate(
            prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
        )

    synchronizeAll()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print(f"TTFT: {output.metrics.first_token_time - now}")

    te.clearState()


# myprof(f)
for _ in range(10):
    f()
    # myprof(f)

TransferEngine.stop()
