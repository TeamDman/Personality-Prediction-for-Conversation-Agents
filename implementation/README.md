We'll proceed with this implementation in a notebook form. We also share our docker environment for everybody to run the code.

# Notebooks
- [01_Transliteration_via_LLM.ipynb](https://github.com/shinshoji01/MacST-project-page/blob/main/implementation/notebooks/01_Transliteration_via_LLM.ipynb)
  - Transliteration via Large Language Models (LLMs)
- [02_TTS_via_Elevenlabs.ipynb](https://github.com/shinshoji01/MacST-project-page/blob/main/implementation/notebooks/02_TTS_via_Elevenlabs.ipynb)
  - Multilingual Text-to-Speech (TTS) via Elevenlabs
- [A_Transliteration_the_and_a.ipynb](https://github.com/shinshoji01/MacST-project-page/blob/main/implementation/notebooks/A_Transliteration_the_and_a.ipynb)
  - Transliteration of "the" and "a". We support "Hindi", "Korean", and "Japanese". If you want to transliterate to other languages you need to run this notebook in advance.

# How to Start
## Import and Modify External Codes
```bash
cd <directory of git repository>
git clone https://github.com/jrgillick/laughter-detection.git
git clone https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents.git
cd Personality-Prediction-for-Conversation-Agents
git submodule update --init --recursive
```
Please modify some lines in `<directory of git repository>/laughter-detection/`.
Modify lines 95 and 109 of `utils/torch_utils.py` as follows.
```bash
Line 95
- def load_checkpoint(checkpoint, model, optimizer=None):
+ def load_checkpoint(checkpoint, model, optimizer=None, device="cuda"):
Line 109
-       checkpoint = torch.load(checkpoint)
+       checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
```
## Docker Environment Setup
I created my environment with docker-compose, so if you want to run my notebooks, please execute the following codes.
```bash
cd Personality-Prediction-for-Conversation-Agents/implementation/Docker
docker-compose up -d --build
docker-compose exec personatab bash
nohup jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --port 1234 &
```
If you are unfamiliar with Vim, please run the following code before opening JupyterLab.
```bash
pip uninstall jupyter-vim
```

Then, go to http://localhost:1234/lab

## Run Notebooks from 01-05
Visit `implementation/notebooks/`
