source ./local-variables.bashrc

docker run --gpus all --env https_proxy=proxy-dmz.intel.com:912 --privileged -it --ipc=host ${docker_tag}
