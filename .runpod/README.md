
1. Clone the repository
git clone git@github.com:MARA1976/hallo3.git
cd hallo3

2. Download pretrained models

pip install huggingface_hub
huggingface-cli download fudan-generative-ai/hallo3 --local-dir ./pretrained_models

3. Docker Image Build (RunPod Serverless)
RunPod automatically builds the image using the provided Dockerfile.
- Entry point: handler.py
- Pretrained models: copied via COPY pretrained_models/ pretrained_models/
- Dependencies: installed from .runpod/requirements.txt
- Conda environment: hallo3
⚠️ Do not modify CMD in the Dockerfile — it's designed for serverless execution.

4. Licence
This deployment setup is based on the original Hallo project. See the main repository for licensing details.

[![Runpod](https://api.runpod.io/badge/MARA1976/hallo3)](https://console.runpod.io/hub/MARA1976/hallo3)

