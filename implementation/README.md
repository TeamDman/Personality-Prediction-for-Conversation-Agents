We'll proceed with this implementation in a notebook form. We also share our docker environment for everybody to run the code.

# Notebooks
- [01_Preprocess_Two-Channel_Audio.ipynb](https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents/blob/main/implementation/notebooks/01_Preprocess_Two-Channel_Audio.ipynb)
    - Preprocessing two-channel audio to obtain word-level transcriptions and laughter probabilities.
- [02_Dialog_Structure_Generation.ipynb](https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents/blob/main/implementation/notebooks/02_Dialog_Structure_Generation.ipynb)
    - Constructing dialog structure.
- [03_Emotion_Sentiment_Prediction.ipynb](https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents/blob/main/implementation/notebooks/03_Emotion_Sentiment_Prediction.ipynb)
    - Predicting utterance-level emotion and sentiment.
- [04_Backchannel_Prediction.ipynb](https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents/blob/main/implementation/notebooks/04_Backchannel_Prediction.ipynb)
    - Predicting backchannel types using LLMs.
- [05_Personality_Prediction.ipynb](https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents/blob/main/implementation/notebooks/05_Personality_Prediction.ipynb)
    - Predicting personality types using LLMs.

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
