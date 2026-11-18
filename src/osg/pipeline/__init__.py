"""Construction: which model, which strategy, which benchmark.

Everything in this package answers "what is this run made of" rather than "what
does the agent do". It is separate from `eval/runner.py` so that swapping a
detector, a VLM or a belief model never means editing the file that drives the
episode loop.
"""
