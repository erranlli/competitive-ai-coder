# Model Evaluation

This directory contains instructions to setup code evaluation docker and run model evaluation with respect to benchmarks. 

## Setup Evaluation Docker
Default docker configuration has several issuesm e.g. very small payload. Follow instructions below to setup a proper evaluation docker.  

### clone and enter repo
git clone https://github.com/engineer-man/piston

### Build your new, fixed image from the Dockerfile
docker build -t piston-fixed .

### Stop any old container
docker stop piston_api
docker rm piston_api

### Run the new image. The command is now simple and clean!
docker run \
    --privileged \
    -dit \
    -p 2000:2000 \
    --name piston_api \
    --env-file ./piston.env \
    piston-fixed

### Install latest python
cli/index.js ppman install python

# Run evaluation

python  eval_with_piston_gentest_checker_stats.py \
  --solutions-path model_solutions/deepseek-ai-deepseek-r1-0528-qwen3-8b__open-r1-codeforces__default__test__vllm.jsonl
