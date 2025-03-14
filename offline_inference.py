from vllm import LLM, SamplingParams
import time

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument(
    "-esd",
    "--enable_spec_decode",
    action='store_true', 
    help="enable speculative decode",
)
parser.add_argument(
    "-tmdl",
    "--target_model",
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="target model name or path",
)
parser.add_argument(
    "-dmdl",
    "--draft_model",
    default="meta-llama/Llama-3.2-3B-Instruct",
    help="draft model name or path",
)

# Sample prompts.
prompts = [
    "Tell me about future of global warming.",
    "Can you describe Bernoulli's Equation?",
    "How does deepspeed work?",
    "The future of AI is",
]
# Parse the arguments
args = parser.parse_args()

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

if args.enable_spec_decode:
    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=1,
        speculative_model=args.draft_model,
        num_speculative_tokens=5,
    )
else:
    llm = LLM(
        model=args.target_model,
        tensor_parallel_size=1,
    )
    

iter = 5
for i in range(iter):
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"Total time at iter {i} is={time.time() - start}")
    # Print the outputs.
    if i == iter - 1:
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
