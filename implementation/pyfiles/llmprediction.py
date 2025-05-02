import numpy as np

import sys
sys.path.append("../sho_util/pyfiles/")
from gpt import gpt_api_no_stream, get_json_result

sys.path.append('../pyfiles/')
from dialog import concatenate_close_voice

##################################
##### Backchannel Prediction #####
##################################

def get_past_future_conversation(a, b, past, future, delete_toomuch_overlap):
    dfpast = a.copy()
    dffuture = b.copy()
    if delete_toomuch_overlap:
        bl = np.arange(len(dfpast))[dfpast["Overlap-timeshift"]!=""]
        if len(bl)>0:
            dfpast = dfpast[bl[-1]+1:]
        if len(dffuture)>0:
            bl = np.arange(len(dffuture))[dffuture["Overlap-timeshift"]!=""]
            if len(bl)>0:
                dffuture = dffuture[:bl[0]]
    try:
        dfpast = concatenate_close_voice(dfpast, np.inf, if_consecutive=True, bridge_text=" [Pause] ")
        dfpast = dfpast.iloc[-past:]
    except IndexError:
        pass
    try:
        dffuture = concatenate_close_voice(dffuture, np.inf, if_consecutive=True, bridge_text=" [Pause] ")
        dffuture = dffuture.iloc[:future]
    except IndexError:
        pass
    return dfpast, dffuture

def get_prompt_backchannel(dfpast, dffuture, current):
    speaker, spk_interject, backchannel, transcript, both = current
    alltext = f"""
Your task is to classify the type of backchannel. \
There are two types of responses: turn-taking and interjections. \
We focus on interjections. We have two types of encouraging interjections (back-channel): emotive and cognitive.
The backchannel responses are feedbacks given while someone else is talking, to show interest, attention and/or a willingness to keep listening
Therefore, the interjections are not backchannels if the listener attempts to claim the speaking turn.

The Emotive class is one of the backchannel response, which is used to express the speaker's emotional state. \
The Cognitive class is another backchannel response used to reflect the speaker's thought processes or cognitive states. \
Your task is to classify the following interjection, delimited by triple backticks, into "emotive", "cognitive", "not backchannel".
The texts delimited by triple backticks contain the dialog and the target text response with '(TARGET)'.
Triple curly blankets indicate the position of the interjection. 

You also need to determine "sentiment" and "emotion" of the target interjection.
"sentiment" has 5 classes: 'very positive', 'positive', 'neutral', 'negative', 'very negative'.
"emotion" has 5 classes: 'neutral', 'sad', 'angry', 'happy', 'surprised'.

```
Target interjection text: {backchannel}
The speaker who speaks the text inside triple curly blankets: {spk_interject}
"""
    alltext += "\n"
    for i in range(len(dfpast)):
        array = dfpast.iloc[i]
        text = f"Speaker {array['speaker']}: {array['transcription']}\n" 
        alltext += text
        
    both = f"{{{{{{(TARGET) Speaker {spk_interject}: ".join(both.split("{{{"))
    text = f"Speaker {speaker}: {both}\n" 
    alltext += text
    for i in range(len(dffuture)):
        array = dffuture.iloc[i]
        text = f"Speaker {array['speaker']}: {array['transcription']}\n" 
        alltext += text

    alltext += f"""
```

Your response must include the classification result in JSON format at the end of the response.
You must perform analysis via following steps:
1. Summarize conversation before and after the target response.
2. Notice the target text.
3. Determine whether the target text information is backchannel or not.
4. [if backchannel] Determine whether the backchannel is emotive or cognitive. Note that if it's not backchannel, 'interjection type' becomes 'not backchannel'.
5. Classify "interjection type"
6. Classify "emotion" and "sentiment"
This is an example of JSON:
{{
'interjection text': "{backchannel}",
'interjection type': ...,
'emotion': ...,
'sentiment': ...,
}}
"""[1:]
    return alltext[1:-1]

def GetResult_Backchannel(client, prompt, gptmodel, display_print=False, error_path="gpt_error.txt"):
    repeat = True
    trial = 1
    while repeat:
        response = gpt_api_no_stream(client, prompt, model=gptmodel)[1]
        getresult, a = get_json_result(response)
        if getresult:
            valid = sum([cl in a for cl in ["interjection text", "interjection type", "emotion", "sentiment"]])==4
            result = a
            if valid:
                if display_print:
                    print(f"Trial {trial}: Success!!!")
                repeat = False
            else:
                if display_print:
                    with open(error_path, 'w') as a:
                        a.write(response)
                    print(f"Trial {trial}: The result is not valid")
        else:
            if display_print:
                with open(error_path, 'w') as a:
                    a.write(response)
                print(f"Trial {trial}: Error in Converting Json Format")
        trial += 1
    return result

##################################
##### Personality Prediction #####
##################################

def get_prompt_character(array, target_columns, orders, cl2name, eval2step, sampledf=None):
    alltext = f"""
Your task is to classify "Character" of the speaker in conversation using Big Five Inventory (BFI) Personality Traits. \
BFI includes five features: openness, conscientinousness, extraversion, agreeableness, and neuroticism, which is detailed as follows. We also display the opposite term for each class:
- "openness": intellectual, imaginative, independent-minded; opposite term is "closedness".
- "conscientiousness": orderly, responsible, dependable; opposite term is "lack of direction".
- "extraversion": talkative, assertive, energetic; opposite term is "introversion".
- "agreeableness": good-natured, cooperative, trustful; opposite term is "antagonism".
- "neuroticism": emotional instability, irritability, anxiety, self-doubt, depression; opposite term is "emotion stability".

We analyzed the real conversation between two speakers and summarize their behaviours. \
Your task is to classify the characters using those information and sample responses. \
Here, I define some words used in the analysis.
- "backchannel": The backchannel responses are feedbacks given while someone else is talking, to show interest, attention and/or a willingness to keep listening. 
- "emotive": The emotive backchannel is used to express the speaker's emotional state. 
- "cognitive": The cognitive backchannel is another backchannel response used to reflect the speaker's thought processes or cognitive states. 
- "interjection": Interjections are the responses that interject someone to stop talking and claim the speaking turn.
"""
    
    if "samples" in orders:
        alltext += f"""
Additionally, we also show some sample responses spoken by the target speaker. \
We randomly extract some samples from 12-minute conversation. \
We excluded special tokens like commas, periods, or question marks. \
"""
    
    alltext += "\n\nWe put all information about the target speaker within the text delimited by triple backtics.\n\n```"
    for feature in orders:
        if feature=="emotion":
            # Emotion Type
            alltext += "\nEmotions:\n"
            for key in array["emotion"].index:
                value = np.round(array[("emotion", key)]*100, 1)
                text = f"  {key}: {value}%\n" 
                alltext += text
                
        elif feature=="sentiment":
            # Sentiment
            alltext += "\nSentiment:\n"
            for key in array["sentiment"].index:
                value = np.round(array[("sentiment", key)]*100, 1)
                text = f"  {key}: {value}%\n" 
                alltext += text

        elif feature=="basics":
            # Basic Statistics
            alltext += "\nBasic Statistics:\n"
            for cl in target_columns:
                text = f"  {cl2name[cl]}: {array[('basics', cl)]}\n" 
                alltext += text
                
        elif feature=="samples":
            # Sample Responses
            alltext += "\nSample Responses:\n"
            for j in range(len(sampledf)):
                sample = sampledf.iloc[j] 
                text = f"  Sample {j+1}: {sample['transcription']}\n" 
                alltext += text
        
    alltext += f"""
```

Your response must include the classification result in JSON format at the end of the response.
You must perform analysis via following steps:
1. Summarize features you need to consider for predicting conversation characters.
2. Summarize the five conversation characters (BFI)
3. Determine the analyzed features related to each conversation character.
"""
    for f, feature in enumerate(orders):
        text = f'{4+f}. {eval2step[feature]}\n'
        alltext += text
        
    text = f'{4+len(orders)}. Classify the five conversation features with five options "highly aligned", "aligned", "neutral", "opposed", "highly opposed". Give "opposed" or "highly opposed" if the character is more aligned with the opposite terms.'
    alltext += text
    alltext += """\n\n
This is an example of JSON:
{
'openness': ...,
'conscientiousness': ...,
'extraversion': ...,
'agreeableness': ...,
'neuroticism': ...,
}
"""[1:]
    return alltext[1:-1]

def GetResult_Personality(client, prompt, gptmodel, display_print=False, error_path="gpt_error.txt", max_trials=5):
    repeat = True
    trial = 1
    while repeat:
        if trial>max_trials:
            print("Too many failures")
            assert False
        response = gpt_api_no_stream(client, prompt, model=gptmodel)[1]
        getresult, a = get_json_result(response)
        if getresult:
            valid = sum([cl in a for cl in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']])==5
            result = a
            if valid:
                if display_print:
                    print(f"Trial {trial}: Success!!!")
                repeat = False
            else:
                if display_print:
                    with open(error_path, 'w') as a:
                        a.write(response)
                    print(f"Trial {trial}: The result is not valid")
        else:
            if display_print:
                with open(error_path, 'w') as a:
                    a.write(response)
                print(f"Trial {trial}: Error in Converting Json Format")
        trial += 1
    return result