

# first step  -> launch model
# conda activate chatbot && vllm serve /home/linux/Downloads/models/Qwen2.5-7B-Instruct --dtype bfloat16 --served-model-name Qwen2.5-7B-Instruct --max-model-len 8096

# second step launch webui
conda activate chatbot && streamlit run webdemo.py --server.port 6006 --server.runOnSave true