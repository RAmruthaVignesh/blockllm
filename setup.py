from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="blockllm",
    version="1.0",
    description="BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks",
    url="https://arxiv.org/abs/2406.17296",  # Reference to the paper
    author="Amrutha Varshini Ramesh",  # Updated author
    author_email="avramesh@cs.ubc.ca",  # Updated author email
    license="Apache 2.0",
    packages=["blockllm_torch"],  # Updated package name
    install_requires=required,
)
