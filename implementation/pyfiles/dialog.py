from tqdm import tqdm
import librosa
import numpy as np
import pandas as pd
import os
import torch

import sys
sys.path.append("../sho_util/pyfiles/")
from basic import get_bool_base_on_conditions

sys.path.append('./../../../laughter-detection/')
import laugh_segmenter
sys.path.append('./../../../laughter-detection/utils/')
import audio_utils, data_loaders, torch_utils
from functools import partial

import soundfile as sf
def save_audio(path, y, fs):
    sf.write(path, y, fs, "PCM_16")

class GetLaughs:
    def __init__(self, config, sample_rate, device, model_path):
        model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
        feature_fn = config['feature_fn']
        model.set_device(device)
        if os.path.exists(model_path):
            torch_utils.load_checkpoint(model_path+'/best.pth.tar', model, device=device)
            model.eval()
        else:
            raise Exception(f"Model checkpoint not found at {model_path}")
            
        self.model = model
        self.feature_fn = feature_fn
        self.sample_rate = sample_rate
        self.config = config
        self.device = device
        
    def get(self, audio_path):
    
        ##### Preprocess Audio
        inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
            audio_path=audio_path, feature_fn=self.feature_fn, sr=self.sample_rate)
        collate_fn=partial(audio_utils.pad_sequences_with_labels,
                                expand_channel_dim=self.config['expand_channel_dim'])
        inference_generator = torch.utils.data.DataLoader(
            inference_dataset, num_workers=1, batch_size=128, shuffle=False, collate_fn=collate_fn)

        ##### Make Predictions
        probs = []
        # for model_inputs, _ in tqdm(inference_generator):
        for model_inputs, _ in inference_generator:
            x = torch.from_numpy(model_inputs).float().to(self.device)
            preds = self.model(x).cpu().detach().numpy().squeeze()
            if len(preds.shape)==0:
                preds = [float(preds)]
            else:
                preds = list(preds)
            probs += preds
        probs = np.array(probs)
        file_length = audio_utils.get_audio_length(audio_path)
        fps = len(probs)/float(file_length)
        probs = laugh_segmenter.lowpass(probs)
        return probs, fps

def concatenate_close_voice(data, thres=0.25, dbtkey="duration-before-talking", if_consecutive=True, bridge_text=" "):
    if len(data.speaker.unique())==1:
        diffspk = list(set(["A", "B"])-set([data.iloc[0]["speaker"]]))[0]
        dfadd = data.iloc[0:1].copy()
        dfadd.loc[:, ["start", "end", "speaker", "duration", "duration-before-talking"]] = [1000000, 1000000, diffspk, 0.0, 1000000]
        dfadd.index = [np.array(list(data.index)).max()+1]
        data = pd.concat([data, dfadd], axis=0)
        lastadded = True
    else:
        lastadded = False

    df = {}
    for spk in ["A", "B"]:
        df[spk] = data[get_bool_base_on_conditions(data, {"speaker": [spk]})]

    indices = {}
    idx_list = {}
    for spk in ["A", "B"]:
        indices[spk] = df[spk].index
        bl = (df[spk][dbtkey]<=thres).values
        if if_consecutive:
            try:
                bl = bl*(data.speaker[np.array(list(indices[spk]))-1]==spk).values
            except KeyError:
                bl = bl*(np.array([False] + list(data.speaker[np.array(list(indices[spk]))[1:]-1]==spk)))
        bl[0] = False
        idx_list[spk] = np.arange(len(bl))[bl]
        for idx in idx_list[spk][::-1]:
            a = df[spk].loc[indices[spk][idx]]
            df[spk].loc[indices[spk][idx-1], "end"] = a["end"]
            try:
                df[spk].loc[indices[spk][idx-1], "end-timeshift"] = a["end-timeshift"]
            except KeyError:
                pass
            prev = df[spk].loc[indices[spk][idx-1], "transcription"] 
            nxt = a["transcription"]
            if prev=="[Laugh]" and nxt=="[Laugh]":
                b = "[Laugh]"
            else:
                b = prev + bridge_text + nxt
            df[spk].loc[indices[spk][idx-1], "transcription"] = " ".join(b.split())
        data.loc[df[spk].index] = df[spk]
    data = data.drop([indices[spk][idx]  for spk in ["A", "B"] for idx in idx_list[spk][::-1]])
    data = data.reset_index(drop=True)
    if lastadded:
        data = data.iloc[:-1]
    return data

def get_overlap(data, margin=0.0, startkey="start", endkey="end", overlapkey="Overlap"):
    df = {}
    for spk in ["A", "B"]:
        df[spk] = data[get_bool_base_on_conditions(data, {"speaker": [spk]})]

    data[overlapkey] = np.array([""]*len(data)).reshape(-1, 1)
    for i in range(len(data)):
        array = data.iloc[i]
        checkdf = df["A"] if array["speaker"]=="B" else df["B"]
        a = checkdf[checkdf[startkey] >= array[startkey]]
        aa = a[a[startkey]+margin<=array[endkey]]

        b = checkdf[checkdf[startkey]<=array[startkey]]
        bb = b[checkdf[endkey]>=array[startkey]+margin]
        if (len(aa)+len(bb))>0:
            l = list(aa.index)+list(bb.index)
            l.sort()
            data.iloc[i, list(data.columns).index(overlapkey)] = "-".join([str(v) for v in l])
        else:
            data.iloc[i, list(data.columns).index(overlapkey)] = ""
    return data
    
def get_fully_overlap(data, margin=0.0, startkey="start", endkey="end", overlapkey="Overlap", fokey="Fully-Overlap", if_consecutive=False):
    data[fokey] = np.zeros((len(data), 1)).astype(bool)
    for i in range(len(data)):
        array = data.iloc[i]
        ol = array[overlapkey]
        if len(ol)>0:
            ol = ol.split("-")
            if if_consecutive:
                a = data.iloc[int(ol[0])][startkey]+margin<=array[startkey]
                b = array[endkey]+margin<=data.iloc[int(ol[-1])][endkey]
                data.iloc[i, list(data.columns).index(fokey)] = bool(a*b)
            else:
                bl_list = []
                for oneol in ol:
                    a = data.iloc[int(oneol)][startkey]+margin<=array[startkey]
                    b = array[endkey]+margin<=data.iloc[int(oneol)][endkey]
                    bl_list += [bool(a*b)]
                data.iloc[i, list(data.columns).index(fokey)] = np.max(bl_list)>0
                    
    return data

def get_duration(data):
    data["duration"] = data["end"] - data["start"]
    return data

def get_duration_before_talking(data, startkey="start", endkey="end", key="duration-before-talking"):
    data[key] = np.zeros((len(data), 1))

    df = {}
    for spk in ["A", "B"]:
        df[spk] = data[get_bool_base_on_conditions(data, {"speaker": [spk]})]
        data.iloc[df[spk].index, list(data.columns).index(key)] = np.array([0.0] + list(df[spk][startkey].values[1:] - df[spk][endkey].values[:-1]))
    return data

def delete_fully_overlap(data, thres_duration=np.inf, thres_word=np.inf, fokey="Fully-Overlap"):
    bl = data[fokey]*(data["duration"]<thres_duration)*(np.array([len(a.replace("(", " ").replace(")", " ").split()) for a in data["transcription"].values])<=thres_word)
    data = data[(1-bl).astype(bool)]
    data = data.reset_index(drop=True)
    return data

def get_timeshifted_for_overlap(data, thres_overlap=2.0):
    key1 = "start-timeshift" 
    key2 = "end-timeshift"
    data[key1] = data["start"]
    data[key2] = data["end"]

    samples = []
    for i in range(len(data)):
        array = data.iloc[i]
        if array["Overlap"]=="":
            continue
        for ol in array["Overlap"].split("-"):
            ol = int(ol)
            if i<ol:
                diff = array["end"]-data.iloc[ol]["start"]
                if thres_overlap>=diff:
                    idx1 = list(data.columns).index(key1)
                    idx2 = list(data.columns).index(key2)
                    data.iloc[ol:, [idx1,idx2]] = data.iloc[ol:, [idx1,idx2]] + diff
                
    data = update_information(data, updateinfo=["-timeshift"])
    return data

def delete_silence_invervals(data, startkey="start-timeshift", endkey="end-timeshift", overlapkey="Overlap-timeshift"):
    zeroing =  data[startkey][0]
    data.loc[:, startkey] = data.loc[:, startkey] - zeroing
    data.loc[:, endkey] = data.loc[:, endkey] - zeroing

    for i in range(len(data)-1):
        array = data.iloc[i]
        if len(array[overlapkey])>0:
            continue
        diff = data.iloc[i+1][startkey] - array[endkey]
        if diff>1e-6:
            idx1 = list(data.columns).index(startkey)
            idx2 = list(data.columns).index(endkey)
            data.iloc[i+1:, [idx1,idx2]] = data.iloc[i+1:, [idx1,idx2]] - diff
    return data

def update_information(data, updateinfo=["", "-timeshift"], margin_overlap=0.1, margin_fully_overlap=0.0, if_consecutive=False):
    for add in updateinfo:
        startkey = "start" + add
        endkey = "end" + add
        btkey = "duration-before-talking" + add
        overlapkey = "Overlap" + add
        fokey = "Fully-Overlap" + add
        
        data = get_duration(data)
        data = get_duration_before_talking(data, startkey, endkey, btkey)
        data = get_overlap(data, margin_overlap, startkey, endkey, overlapkey)
        data = get_fully_overlap(data, margin_fully_overlap, startkey, endkey, overlapkey, fokey, if_consecutive)
    return data

def get_sentence_start(data, thres_sentence_length=3.0, startkey="start-timeshift", endkey="end-timeshift", overlapkey="overlap-timeshift"):
    key = "sentence start"
    data[key] = ""
    data[key][0] = "0"
    snum = 1
    for i in range(1, len(data)):
        array = data.iloc[i]
        arrayback = data.iloc[:i]
        try:
            a = arrayback[arrayback["speaker"]=="A"].iloc[-1]
            b = arrayback[arrayback["speaker"]=="B"].iloc[-1]
        except IndexError:
            continue
        if (array["start-timeshift"]-a["end-timeshift"]>thres_sentence_length) and (array["start-timeshift"]-b["end-timeshift"]>thres_sentence_length):
            data[key][i] = str(snum) + ": Too long break"
            snum += 1
            continue
        if len(array["Overlap-timeshift"])==0 and (len(a["Overlap-timeshift"])>0 and len(b["Overlap-timeshift"])>0):
            data[key][i] = str(snum) + ": Long time overlap"
            snum += 1
            continue
    return data

def silence_removal(data, top_db=25, trim_window=256, trim_stride=128):
    for i in tqdm(range(len(data))):
        array = data.iloc[i]
        for s, speaker in enumerate(["A", "B"]):
            if array["speaker"]==speaker:
                start = array["start"]
                end = array["end"]
                # print(array["transcription"])
                segment = audio[s][int(sr*start):int(sr*end)]
                segment, spans = librosa.effects.trim(segment, top_db=top_db, frame_length=trim_window, hop_length=trim_stride)
                data.iloc[i, :2] = start + spans/sr
    data = data.sort_values("start")
    data = data.reset_index(drop=True)
    return data

def most_frequent(List):
    return max(set(List), key=List.count)

def get_start_end_referencedf(referencedf, array):
    referencedf = referencedf.copy().reset_index(drop=True)
    referencedf = referencedf[get_bool_base_on_conditions(referencedf, {"speaker": [array["speaker"]]})]
    startend = {}
    for mode in ["start", "end"]:
        candidates = list(referencedf[np.abs(referencedf[mode]-array[mode])<1e-5].index)
        if len(candidates)==0:
            startend[mode] = None
        elif len(candidates)==1:
            startend[mode] = candidates[0]
        elif len(candidates)>=2:
            startend[mode] = candidates[-1*int(mode=="end")]
    return startend["start"], startend["end"]

import re
from whisper.normalizers.english import EnglishNumberNormalizer, EnglishSpellingNormalizer, remove_symbols_and_diacritics
# keep numbers '
# don't ignore hmm mm mhm
class EnglishTextNormalizer:
    def __init__(self):
        # self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            # r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # s = re.sub(self.ignore_patterns, "", s)
        # s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        # s = remove_symbols_and_diacritics(s, keep=".%$¢€£'")  # keep numeric symbols
        s = remove_symbols_and_diacritics(s, keep="%$¢€£'")  # keep numeric symbols

        # s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space
        
        return s