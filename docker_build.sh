source ./local-variables.bashrc

docker build -t ${docker_tag}:latest -f Dockerfile.ig-gpu-qwen2p5vl . 
