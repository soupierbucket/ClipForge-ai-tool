from __future__ import annotations

import json
import re
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from app.config import settings
from app.prompts.effect_analysis import EFFECT_TAGGING_PROMPT
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MOMENT_TYPES = {"punchline", "surprise", "reveal", "fail", "emphasis", "awkward_pause", "dramatic"}
SFX_NAMES = {"hit", "whoosh", "rimshot", "record_scratch", "ding", "crickets"}
FILTER_NAMES = {"none", "vignette", "saturation", "desaturate", "contrast"}
RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "effect_rules.json"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "sfx"

class EffectItem(BaseModel):
    time: float = Field(ge=0)
    duration: float = Field(ge=.3, le=1.5)
    moment_type: Literal["punchline", "surprise", "reveal", "fail", "emphasis", "awkward_pause", "dramatic"]
    sfx: Literal["hit", "whoosh", "rimshot", "record_scratch", "ding", "crickets"] | None = None
    filter: Literal["none", "vignette", "saturation", "desaturate", "contrast"] = "none"
    overlay: Literal["", "😂", "😮", "✨", "😬", "🔥", "⚡"] = ""
    enabled: bool = True

class EffectsPlanRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    candidate_id: str = Field(pattern=r"^[a-f0-9]{12}$")

class GenerateClipRequestWithEffects(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    candidate_id: str = Field(pattern=r"^[a-f0-9]{12}$")
    effects: list[EffectItem] = Field(default_factory=list, max_length=24)

def load_rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))["rules"]

def _word_events(transcript: list[dict], start: float, end: float) -> list[dict]:
    events=[]
    words=[]
    for segment in transcript:
        segment_start=float(segment.get("start",0)); segment_end=float(segment.get("end",0))
        raw=segment.get("words") or []
        for item in raw:
            t=float(item.get("start",-1)); e=float(item.get("end",-1)); text=str(item.get("text", "")).strip()
            if text and start <= t < end and t < e:
                words.append({"start":t,"end":e,"text":text.lower().strip(".,!?;:()[]{}\"'" ),"question":text.rstrip().endswith("?")})
    if not words:
        for segment in transcript:
            seg_start=max(start,float(segment.get("start",0))); seg_end=min(end,float(segment.get("end",0)))
            tokens=str(segment.get("text", "")).split()
            if not tokens or seg_end <= seg_start: continue
            step=(seg_end-seg_start)/len(tokens)
            words.extend({"start":seg_start+i*step,"end":seg_start+(i+1)*step,"text":w.lower().strip(".,!?;:")} for i,w in enumerate(tokens))
    words.sort(key=lambda w:w["start"])
    patterns={
      "punchline": {"haha","hahaha","lol","laugh","laughed","laughing","joke","funny","hilarious"},
      "surprise": {"wow","what","really","seriously","unbelievable","unexpected","suddenly","wait"},
      "reveal": {"actually","turns","revealed","secret","finally","realized","truth"},
      "fail": {"oops","failed","fail","mistake","wrong","lost","crashed","broke"},
      "emphasis": {"absolutely","never","always","exactly","incredible","insane","important"},
      "dramatic": {"danger","dead","death","fight","attack","explosion","exploded","battle","destroyed"},
    }
    for i,w in enumerate(words):
        for kind, cues in patterns.items():
            if w["text"] in cues:
                events.append({"time":max(0,w["start"]-start),"duration":.65,"moment_type":kind,"word_start":w["start"]-start,"word_end":w["end"]-start})
                break
        if i+1<len(words) and words[i+1]["start"]-w["end"]>=.9 and (w.get("question") or w["text"] in {"well","so","uh","um"}):
            events.append({"time":max(0,w["end"]-start),"duration":.65,"moment_type":"awkward_pause","word_start":w["end"]-start,"word_end":words[i+1]["start"]-start})
    return events

def build_effects_plan(transcript: list[dict], start: float, end: float, audio_events: list[dict] | None = None, intensity: str = "low", llm_events: list[dict] | None = None) -> list[dict]:
    """Conservative, transcript-timed effects; unsupported visual claims are never generated."""
    if intensity == "off" or end <= start: return []
    factor={"low":1,"med":1.2,"high":1.5}.get(intensity,1)
    rules=load_rules(); duration=end-start
    candidates=_word_events(transcript,start,end)
    candidates.extend(llm_events or [])
    speech=[]
    for segment in transcript:
        for word in (segment.get("words") or []):
            ws=max(0.0,float(word.get("start",start))-start); we=min(end-start,float(word.get("end",end))-start)
            if we>ws: speech.append((ws,we))
    speech.sort()
    for signal in audio_events or []:
        t=float(signal.get("time",-1))-start
        if 0 <= t < duration:
            candidates.append({"time":t,"duration":.55,"moment_type":"emphasis","audio_signal":True,"word_start":t,"word_end":t+.15})
    # Keep only cues backed by words or a measured action/audio signal. Sort by confidence then time.
    candidates.sort(key=lambda x:(x["time"], 0 if x.get("audio_signal") else 1))
    result=[]; last_sfx=None
    min_gap=max(4.0,6.0/factor); cap=min(6,max(1,int(duration/5)))
    for item in candidates:
        if len(result)>=cap: break
        t=max(0.0,min(duration-.3,float(item["time"])))
        if any(abs(t-old["time"])<min_gap for old in result): continue
        kind=item["moment_type"]
        if kind not in MOMENT_TYPES: continue
        rule=rules[kind]; sfx=rule["sfx"]
        cue_duration=round(max(.3,min(1.0,float(item.get("duration",.65)))),3)
        if sfx and speech:
            gap_start=None
            for current, following in zip(speech, speech[1:]):
                proposed=max(t,current[1])
                if following[0]-proposed >= cue_duration and proposed <= t+1.25:
                    gap_start=proposed; break
            if gap_start is None and any(max(t,ws)<min(t+cue_duration,we) for ws,we in speech):
                sfx=None
            elif gap_start is not None:
                t=gap_start
        if sfx and sfx==last_sfx:
            alternatives=[name for name in sorted(SFX_NAMES) if name!=last_sfx]
            sfx=alternatives[0] if alternatives else None
        effect={"time":round(t,3),"duration":cue_duration,"moment_type":kind,"sfx":sfx,"filter":rule["filter"],"overlay":rule["overlay"],"enabled":True}
        try: result.append(EffectItem.model_validate(effect).model_dump())
        except Exception: continue
        last_sfx=sfx
    return result

def tag_moments_with_local_llm(transcript: list[dict], start: float, end: float, audio_events: list[dict] | None = None) -> list[dict]:
    """Ask local Ollama for word-indexed tags, then reject unsupported tags."""
    words=[]
    for segment in transcript:
        for word in segment.get("words", []):
            t=float(word.get("start", -1)); e=float(word.get("end", -1)); text=str(word.get("text", "")).strip()
            if text and start <= t < end and t < e: words.append({"start":t,"end":e,"text":text})
    if not words: return []
    prompt_words="\n".join(f"[{i}] {w['start']-start:.3f}-{w['end']-start:.3f}: {w['text']}" for i,w in enumerate(words))
    user="Clip duration: %.3f seconds\n%s" % (end-start,prompt_words)
    body=json.dumps({"model":settings.llm_model,"messages":[{"role":"system","content":EFFECT_TAGGING_PROMPT},{"role":"user","content":user}],"format":"json","stream":False,"options":{"temperature":0.1}}).encode("utf-8")
    try:
        request=Request(f"{settings.ollama_base_url.rstrip('/')}/api/chat",data=body,headers={"Content-Type":"application/json"},method="POST")
        with urlopen(request,timeout=90) as response: payload=json.loads(json.loads(response.read().decode("utf-8")).get("message",{}).get("content","{}"))
        raw=payload.get("moments", []) if isinstance(payload,dict) else []
    except (URLError,HTTPError,TimeoutError,KeyError,TypeError,json.JSONDecodeError,ValueError):
        return []
    if not isinstance(raw,list): return []
    audio_times=[float(event.get("time",-100))-start for event in (audio_events or [])]
    grounded={
      "punchline": {"haha","hahaha","lol","laugh","laughed","laughing","joke","funny","hilarious"},
      "surprise": {"wow","what","really","seriously","unbelievable","unexpected","suddenly","wait"},
      "reveal": {"actually","turns","revealed","secret","finally","realized","truth"},
      "fail": {"oops","failed","fail","mistake","wrong","lost","crashed","broke"},
      "emphasis": {"absolutely","never","always","exactly","incredible","insane","important"},
      "dramatic": {"danger","dead","death","fight","attack","explosion","exploded","battle","destroyed"},
    }
    accepted=[]
    for item in raw[:12]:
        if not isinstance(item,dict) or item.get("moment_type") not in MOMENT_TYPES: continue
        try: index=int(item["word_index"])
        except (KeyError,TypeError,ValueError): continue
        if not 0 <= index < len(words): continue
        word=words[index]; local=word["start"]-start; token=re.sub(r"[^a-z]", "", word["text"].lower()); kind=item["moment_type"]
        if kind=="awkward_pause":
            next_word=words[index+1] if index+1<len(words) else None
            if not next_word or next_word["start"]-word["end"] < .75: continue
            local=word["end"]-start
        elif token not in grounded.get(kind,set()) and not any(abs(local-t)<=.8 for t in audio_times): continue
        accepted.append({"time":max(0,local),"duration":.65,"moment_type":kind,"word_start":word["start"]-start,"word_end":word["end"]-start,"llm":True})
    return accepted[:6]


def validate_effects_plan(effects: list[dict], clip_duration: float) -> list[dict]:
    clean=[]; last_time=-100; last_sfx=None
    for raw in sorted(effects,key=lambda item:float(item.get("time",0))):
        item=EffectItem.model_validate(raw).model_dump()
        if item["time"]+item["duration"]>clip_duration+.02: raise ValueError("An effect extends beyond the clip duration.")
        if not item["enabled"]: continue
        if item["time"]-last_time < 4.0: raise ValueError("Effects must be at least 4 seconds apart.")
        if item["sfx"] and item["sfx"]==last_sfx: raise ValueError("The same sound effect cannot play twice in a row.")
        last_time=item["time"]; last_sfx=item["sfx"]
        clean.append(item)
    for index, item in enumerate(clean):
        if sum(1 for other in clean if item["time"] <= other["time"] < item["time"] + 30.0) > 6:
            raise ValueError("Effects are limited to six per 30 seconds.")
        if any(item["filter"] != "none" and other["filter"] != "none" and max(item["time"],other["time"]) < min(item["time"]+item["duration"],other["time"]+other["duration"]) for other in clean[index+1:]):
            raise ValueError("Visual filters cannot overlap.")
    return clean

def build_video_filter(effect: dict, input_label: str, output_label: str) -> str:
    start=float(effect["time"]); end=start+float(effect["duration"]); enable=f"between(t\\,{start:.3f}\\,{end:.3f})"
    filters={
      "vignette": "vignette=PI/5",
      "saturation": "eq=saturation=1.12:contrast=1.03",
      "desaturate": "eq=saturation=0.65:contrast=1.04",
      "contrast": "eq=contrast=1.10:saturation=1.04",
      "none": "null",
    }
    selected=effect.get("filter","none")
    if selected not in FILTER_NAMES: raise ValueError("Unknown visual filter.")
    return f"[{input_label}]{filters[selected]}:enable='{enable}'[{output_label}]" if selected!="none" else f"[{input_label}]null[{output_label}]"
