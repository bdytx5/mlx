from huggingface_hub import hf_hub_download
cp /Users/brettyoung/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1/snapshots/ff3c701f2424c7625fdefb9dd470f45ef18b02d6/tokenizer.model ./
cp /Users/brettyoung/.cache/huggingface/hub/models--TinyLlama--TinyLlama_v1.1/snapshots/ff3c701f2424c7625fdefb9dd470f45ef18b02d6/pytorch_model.bin ./
# Replace with the correct model repository name
model_repo = "TinyLlama/TinyLlama_v1.1"

# Download specific files
pytorch_model_path = hf_hub_download(repo_id=model_repo, filename="pytorch_model.bin")
tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.model")

print(f"Downloaded model weights: {pytorch_model_path}")
print(f"Downloaded tokenizer: {tokenizer_path}")
