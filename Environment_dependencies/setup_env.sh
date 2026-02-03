!pip uninstall -y unsloth unsloth_zoo
!pip install git+https://github.com/unslothai/unsloth-zoo.git
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.29" "trl<0.13.0" peft accelerate bitsandbytes
!pip install chromadb langchain-huggingface sentence-transformers