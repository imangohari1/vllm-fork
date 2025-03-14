GPU=true
HPU=false
VLLM_DIR=$(realpath .)
RunSpecDecode=true

if $GPU; then
        tmdl=/scratch/users/sgohari/others/ai/models/llama3.1-8b-instruct
        dmdl=/scratch/users/sgohari/others/ai/models/Llama-3.2-3B-Instruct
        VLLMFLAGS=" "
        uv pip install datasets -q
fi

if $HPU; then
        tmdl=meta-llama/Meta-Llama-3-8B-Instruct
        dmdl=meta-llama/Llama-3.2-3B-Instruct
        VLLMFLAGS="  VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=false"
        pip install datasets -q
fi

if $RunOnline; then
        cd /tmp
        export HTTPS_PROXY=http://proxy-dmz.intel.com:912
        export HTTP_PROXY=http://proxy-dmz.intel.com:912
        export no_proxy=0.0.0.0,localhost,intel.com,.intel.com,10.0.0.0/8,192.168.0.0/16

        if $RunSpecDecode; then
                cmd="$VLLMFLAGS  python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model $tmdl --seed 42 -tp 1 --speculative_model $dmdl --num_speculative_tokens 5 --gpu_memory_utilization 0.9 &"
        else
                # we run the draft model to get basic values
                cmd="$VLLMFLAGS  python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model $dmdl --seed 42 -tp 1 --gpu_memory_utilization 0.9 &"
        fi
        echo $cmd && eval $cmd

        echo "checking the server status"
        while true; do if [[ -n $(curl -s http://localhost:8000/v1/models) ]]; then
                echo "Found the server running"
                break
        else
                echo "server not found..sleeping for 10"
                sleep 10
        fi; done

        ln -sf $VLLM_DIR/benchmarks/ .
        cd benchmarks/
        if [ ! -f "./ShareGPT_V3_unfiltered_cleaned_split.json" ]; then wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json; fi
        if $RunSpecDecode; then
                cmd="python benchmark_serving.py --port 8000 --backend vllm --model $tmdl --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-prompts 128"
        else
                # we run the draft model to get basic values
                cmd="python benchmark_serving.py --port 8000 --backend vllm --model $dmdl --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 1 --num-prompts 128"
        fi
        for i in {1..3}; do echo $cmd && eval $cmd; done
fi

if $RunOfline; then
        cd /tmp
        cp $VLLM_DIR/offline_inference.py .
        if $RunSpecDecode; then
                OFFLINEARGS=" -tmdl $tmdl -dmdl $dmdl -esd"
        else
                # we run the draft model to get basic values
                OFFLINEARGS=" -tmdl $dmdl"
        fi
        cmd="$VLLMFLAGS  python  offline_inference.py $OFLINEARGS"
        echo $cmd && eval $cmd
fi
