"""
English to Tagalog Video Dubbing - Multi-Translator Version (With Sub-Batching, GPU, BGM, Glossary & Collision Avoidance)
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import ffmpeg
import edge_tts
import whisper
import google.generativeai as genai
import re
import requests
from dotenv import load_dotenv

from dataclasses import dataclass

# Load .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Subukang i-load ang FREE Google Translator (deep-translator)
try:
    from deep_translator import GoogleTranslator
    google_translator_available = True
    logger.info("✅ Google Translator (deep-translator) loaded successfully")
except ImportError:
    google_translator_available = False
    logger.info("ℹ️ Google Translator not installed. Install with: pip install deep-translator")

# Subukang i-import ang OpenAI library
try:
    from openai import OpenAI
    openai_available = True
    logger.info("✅ OpenAI library loaded successfully")
except ImportError:
    openai_available = False
    logger.info("ℹ️ OpenAI library not installed. If you want to use OpenAI, run 'pip install openai'")

# Subukang i-import ang Groq library
try:
    from groq import Groq
    groq_available = True
    logger.info("✅ Groq SDK loaded successfully")
except ImportError:
    groq_available = False
    logger.info("ℹ️ Groq SDK not installed. Run 'pip install groq' to use Groq.")

# Initialize Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    logger.info("✅ Google Gemini loaded successfully")
else:
    logger.warning("No Gemini API key found!")

@dataclass
class SpeechSegment:
    text: str
    start_time: float
    end_time: float
    speaker_gender: str = "unknown"
    translated_text: str = ""
    audio_path: str = ""

class VideoDubber:
    def __init__(self):
        self.whisper_model = None
        
        # Ipunin ang lahat na magagamit na Gemini API Keys mula sa .env
        self.api_keys = []
        primary = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if primary:
            self.api_keys.append(primary)
        
        for idx in range(2, 11):
            key = os.getenv(f"GEMINI_API_KEY_{idx}")
            if key:
                self.api_keys.append(key)
                
        self.current_key_index = 0
        
        if self.api_keys:
            genai.configure(api_key=self.api_keys[self.current_key_index])
            self.model = genai.GenerativeModel('gemini-3.1-flash-lite')
            logger.info(f"✅ Loaded {len(self.api_keys)} Gemini API Key(s) for auto-rotation on [Gemini 3.1 Flash Lite].")
        else:
            self.model = None
            logger.warning("No Gemini API keys found!")

    def rotate_api_key(self):
        """
        Lilipat sa susunod na available na Gemini API Key kung sakaling ma-rate limit.
        """
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            next_key = self.api_keys[self.current_key_index]
            genai.configure(api_key=next_key)
            logger.info(f"🔄 [Gemini] Switched to API Key #{self.current_key_index + 1} due to rate-limit/quota.")
            return True
        return False

    def load_whisper(self, model_name="base"):
        """Inayos upang mag-auto-detect ng GPU (CUDA) para sa mabilis na transcription"""
        if self.whisper_model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper {model_name} model on [{device.upper()}]...")
            self.whisper_model = whisper.load_model(model_name, device=device)
        return self.whisper_model

    def get_audio_duration(self, filepath):
        """Kuhanin ang eksaktong haba ng isang audio file gamit ang ffprobe"""
        try:
            probe = ffmpeg.probe(filepath)
            return float(probe['format']['duration'])
        except Exception as e:
            logger.warning(f"Hindi makuha ang haba ng audio {filepath}: {e}")
            return 1.0

    def get_voice_params(self, speaker_type, base_voice="fil-PH-AngeloNeural", rate_offset="+0%", pitch_offset="+0Hz", multi_speaker=True, emotion="normal"):
        """
        Dito ginagaya ang iba't ibang boses (Lalaki, Babae, Bata, Lolo, Lola)
        AT inilalapat ang bilis, tono, at lakas ng boses batay sa natukoy na EMOTION!
        """
        if base_voice == "Google-Translate-TTS":
            return "Google-Translate-TTS", "+0%", "+0Hz", "+0%"

        voice = base_voice
        rate_num = 0
        pitch_num = 0
        volume_num = 0  # 0% base offset para sa volume ng Edge-TTS (volume reduction/increase)

        # 1. Base Character Mappings (Tukuyin ang boses batay sa edad/kasarian)
        if multi_speaker:
            if speaker_type == "female":
                voice = "fil-PH-BlessicaNeural"
                rate_num, pitch_num = 4, 0
            elif speaker_type == "child":
                voice = "fil-PH-BlessicaNeural"
                rate_num, pitch_num = 12, 13
            elif speaker_type == "elderly_female":
                voice = "fil-PH-BlessicaNeural"
                rate_num, pitch_num = -12, -3
            elif speaker_type == "elderly_male":
                voice = "fil-PH-AngeloNeural"
                rate_num, pitch_num = -12, -5
            else: # "male" o fallback
                voice = "fil-PH-AngeloNeural"
                rate_num, pitch_num = 4, 0
        else:
            try:
                rate_num = int(rate_offset.replace("%", "").replace("+", ""))
            except Exception:
                rate_num = 0
            try:
                pitch_num = int(pitch_offset.replace("Hz", "").replace("+", ""))
            except Exception:
                pitch_num = 0

        # 2. Dynamic Emotion Offsets
        emotion = str(emotion).lower().strip()
        if emotion == "whisper":
            rate_num -= 10
            pitch_num -= 2
            volume_num -= 45 # Hihinaan nang husto ang boses (pabulong)
        elif emotion == "angry":
            rate_num += 12
            pitch_num += 4
            volume_num += 15 # Pasigaw / tense
        elif emotion == "sad":
            rate_num -= 15
            pitch_num -= 3
            volume_num -= 20 # Malungkot / mahina
        elif emotion == "excited":
            rate_num += 12
            pitch_num += 5
            volume_num += 15 # Masaya / energetic

        # I-format muli para sa Edge-TTS
        rate = f"{rate_num:+d}%"
        pitch = f"{pitch_num:+d}Hz"
        volume = f"{volume_num:+d}%"

        return voice, rate, pitch, volume

    async def translate_with_google(self, texts: list[str]) -> list[str]:
        """Libre at Walang Limitasyong Google Translate"""
        if not google_translator_available:
            logger.warning("deep-translator is not installed. Run 'pip install deep-translator'.")
            return []

        logger.info(f"📦 [Google Translate] Translating ALL {len(texts)} lines...")
        try:
            loop = asyncio.get_event_loop()
            translator_instance = GoogleTranslator(source='auto', target='tl')
            
            def make_call():
                return translator_instance.translate_batch(texts)
                
            translated_list = await loop.run_in_executor(None, make_call)
            logger.info("✅ Google Translate successful!")
            return translated_list
        except Exception as e:
            logger.error(f"❌ Google Translate Error: {e}")
            return []

    async def translate_with_huggingface(self, texts: list[str], glossary="") -> list[dict]:
        """Translation, character, at emotion analysis gamit ang Hugging Face Serverless API (Qwen 2.5 72B)"""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("No HF_TOKEN found in .env. Falling back to Google Translate.")
            return []

        logger.info(f"📦 [Hugging Face] Translating ALL {len(texts)} dialogue lines in sub-batches of 80...")
        
        model_id = "Qwen/Qwen2.5-72B-Instruct"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}

        SUB_BATCH_SIZE = 80
        translated_results = []

        glossary_rules = ""
        if glossary and glossary.strip():
            glossary_rules = f"\nGLOSSARY / DICTIONARY RULES (You MUST strictly follow these mappings):\n{glossary.strip()}"

        for start_idx in range(0, len(texts), SUB_BATCH_SIZE):
            sub_batch = texts[start_idx : start_idx + SUB_BATCH_SIZE]
            sub_batch_translated = []
            
            logger.info(f"  -> [Hugging Face] Translating sub-batch {start_idx // SUB_BATCH_SIZE + 1}...")

            success = False
            for attempt in range(3):
                try:
                    prompt = f"""<|im_start|>system
You are an expert video dubbing translator and script analyst. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.
{glossary_rules}

CRITICAL INSTRUCTIONS:
1. Translate EVERY single line in the list.
2. The output MUST be a valid JSON object containing exactly one key "translations", which is an array of exactly {len(sub_batch)} elements. Do NOT merge, omit, or combine lines.
3. Keep the exact same order as the input.

Analyze the context of each line and detect:
A. The most likely speaker type for that line:
   - "male" (adult male)
   - "female" (adult female)
   - "child" (a young kid)
   - "elderly_male" (Lolo)
   - "elderly_female" (Lola)
B. The emotion behind the line:
   - "normal" (default)
   - "whisper" (quiet, pabulong, secret)
   - "angry" (mad, shouting, tense)
   - "sad" (crying, depressed, soft)
   - "excited" (happy, shouting in joy, enthusiastic)

Return ONLY a valid JSON object with the following structure:
{{
  "translations": [
     {{"translated": "Tagalog translation here", "speaker": "male/female/child/elderly_male/elderly_female", "emotion": "normal/whisper/angry/sad/excited"}},
     ...
  ]
}}
Do not output markdown code blocks or any other explanation.<|im_end|>
<|im_start|>user
Input English lines:
{json.dumps(sub_batch, ensure_ascii=False)}<|im_end|>
<|im_start|>assistant
"""
                    loop = asyncio.get_event_loop()
                    def make_call():
                        payload = {
                            "inputs": prompt,
                            "parameters": {
                                "max_new_tokens": 4096,
                                "return_full_text": False
                            }
                        }
                        return requests.post(api_url, headers=headers, json=payload, timeout=60)

                    response = await loop.run_in_executor(None, make_call)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        
                        if isinstance(res_json, list) and len(res_json) > 0:
                            content = res_json[0].get("generated_text", "").strip()
                        elif isinstance(res_json, dict):
                            content = res_json.get("generated_text", "").strip()
                        else:
                            content = str(res_json).strip()

                        if content.startswith("```json"):
                            content = content[7:]
                        if content.endswith("```"):
                            content = content[:-3]

                        data = json.loads(content.strip())

                        if isinstance(data, dict):
                            if "translations" in data and isinstance(data["translations"], list):
                                translated_list = data["translations"]
                            elif "translated" in data:
                                translated_list = [data]
                            else:
                                lists = [v for v in data.values() if isinstance(v, list)]
                                translated_list = lists[0] if lists else []
                        elif isinstance(data, list):
                            translated_list = data
                        else:
                            translated_list = []

                        for item in translated_list:
                            if "emotion" not in item:
                                item["emotion"] = "normal"
                            if "speaker" not in item:
                                item["speaker"] = "male"

                        if len(translated_list) == len(sub_batch):
                            sub_batch_translated = translated_list
                            success = True
                            break
                        else:
                            logger.warning(f"⚠️ Mismatch in Hugging Face sub-batch length. Retrying...")
                            await asyncio.sleep(2)
                            
                    elif response.status_code == 503:
                        logger.warning("⏳ Hugging Face model is loading... Waiting 15s before retry...")
                        await asyncio.sleep(15)
                    else:
                        logger.warning(f"⚠️ HF error: {response.text}. Retrying...")
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"⚠️ Hugging Face Error: {e}. Retrying...")
                    await asyncio.sleep(2)

            if success:
                translated_results.extend(sub_batch_translated)
            else:
                logger.warning(f"🔄 Hugging Face failed for sub-batch. Falling back to Google Translate...")
                try:
                    google_translated = await self.translate_with_google(sub_batch)
                    if len(google_translated) == len(sub_batch):
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in google_translated]
                    else:
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)
                except Exception as ex:
                    fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)

        return translated_results

    async def translate_with_groq(self, texts: list[str], glossary="") -> list[dict]:
        """Translation at character/emotion analysis gamit ang Groq Cloud (Llama 3.3 70B)"""
        if not groq_available:
            logger.warning("Groq SDK is not installed.")
            return []

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("No GROQ_API_KEY found in .env.")
            return []

        logger.info(f"📦 [Groq] Translating ALL {len(texts)} dialogue lines in sub-batches of 80...")
        client = Groq(api_key=groq_api_key)

        SUB_BATCH_SIZE = 80
        translated_results = []

        glossary_rules = ""
        if glossary and glossary.strip():
            glossary_rules = f"\nGLOSSARY / DICTIONARY RULES (You MUST strictly follow these mappings):\n{glossary.strip()}"

        for start_idx in range(0, len(texts), SUB_BATCH_SIZE):
            sub_batch = texts[start_idx : start_idx + SUB_BATCH_SIZE]
            sub_batch_translated = []
            
            logger.info(f"  -> [Groq] Translating sub-batch {start_idx // SUB_BATCH_SIZE + 1}...")

            success = False
            for attempt in range(3):
                try:
                    prompt = f"""You are an expert video dubbing translator and script analyst. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.
{glossary_rules}

CRITICAL INSTRUCTIONS:
1. Translate EVERY single line in the list.
2. The output MUST be a valid JSON object containing exactly one key "translations", which is an array of exactly {len(sub_batch)} elements. Do NOT merge, omit, or combine lines.
3. Keep the exact same order as the input.

Analyze the context of each line and detect:
A. The most likely speaker type for that line:
   - "male" (adult male)
   - "female" (adult female)
   - "child" (a young kid)
   - "elderly_male" (Lolo)
   - "elderly_female" (Lola)
B. The emotion behind the line:
   - "normal" (default)
   - "whisper" (quiet, pabulong, secret)
   - "angry" (mad, shouting, tense)
   - "sad" (crying, depressed, soft)
   - "excited" (happy, shouting in joy, enthusiastic)

Return ONLY a valid JSON object with the following structure:
{{
  "translations": [
     {{"translated": "Tagalog translation here", "speaker": "male/female/child/elderly_male/elderly_female", "emotion": "normal/whisper/angry/sad/excited"}},
     ...
  ]
}}
Do not output markdown code blocks or any other explanation.

Input English lines:
{json.dumps(sub_batch, ensure_ascii=False)}"""

                    loop = asyncio.get_event_loop()
                    def make_call():
                        return client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"}
                        )

                    response = await loop.run_in_executor(None, make_call)
                    content = response.choices[0].message.content.strip()
                    
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]

                    data = json.loads(content.strip())

                    if isinstance(data, dict):
                        if "translations" in data and isinstance(data["translations"], list):
                            translated_list = data["translations"]
                        elif "translated" in data:
                            translated_list = [data]
                        else:
                            lists = [v for v in data.values() if isinstance(v, list)]
                            translated_list = lists[0] if lists else []
                    elif isinstance(data, list):
                        translated_list = data
                    else:
                        translated_list = []

                    for item in translated_list:
                        if "emotion" not in item:
                            item["emotion"] = "normal"
                        if "speaker" not in item:
                            item["speaker"] = "male"

                    if len(translated_list) == len(sub_batch):
                        sub_batch_translated = translated_list
                        success = True
                        break
                    else:
                        logger.warning(f"⚠️ Mismatch in Groq sub-batch length. Retrying...")
                        await asyncio.sleep(2)
                except Exception as e:
                    logger.warning(f"⚠️ Groq error: {e}. Retrying...")
                    await asyncio.sleep(2)

            if success:
                translated_results.extend(sub_batch_translated)
            else:
                logger.warning(f"🔄 Groq failed for sub-batch {start_idx // SUB_BATCH_SIZE + 1}. Falling back to Google Translate...")
                try:
                    google_translated = await self.translate_with_google(sub_batch)
                    if len(google_translated) == len(sub_batch):
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in google_translated]
                    else:
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)
                except Exception as ex:
                    fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)

        return translated_results

    async def batch_translate_to_tagalog(self, texts: list[str], glossary="") -> list[dict]:
        """Isang bagsakang translation gamit ang Gemini (May Sub-Batching, Auto-Key Rotation & Google Fallback)"""
        if not texts:
            return []

        if not self.model or not self.api_keys:
            logger.warning("No Gemini keys available.")
            return []

        SUB_BATCH_SIZE = 80
        translated_results = []
        
        glossary_rules = ""
        if glossary and glossary.strip():
            glossary_rules = f"\nGLOSSARY / DICTIONARY RULES (You MUST strictly follow these mappings):\n{glossary.strip()}"

        logger.info(f"📦 [Gemini] Translating ALL {len(texts)} lines in sub-batches of {SUB_BATCH_SIZE}...")

        for start_idx in range(0, len(texts), SUB_BATCH_SIZE):
            sub_batch = texts[start_idx : start_idx + SUB_BATCH_SIZE]
            sub_batch_translated = []
            
            logger.info(f"  -> Translating sub-batch {start_idx // SUB_BATCH_SIZE + 1} (Lines {start_idx + 1} to {min(start_idx + SUB_BATCH_SIZE, len(texts))})...")
            
            success = False
            for attempt in range(4):
                try:
                    prompt = f"""You are an expert video dubbing translator and script analyst. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.
{glossary_rules}

CRITICAL INSTRUCTIONS:
1. Translate EVERY single line in the list.
2. The output MUST be a valid JSON object containing exactly one key "translations", which is an array of exactly {len(sub_batch)} elements. Do NOT merge, omit, or combine lines.
3. Keep the exact same order of lines as the input.

Analyze the context of each line and detect:
A. The most likely speaker type for that line:
   - "male" (adult male)
   - "female" (adult female)
   - "child" (a young kid, boy or girl)
   - "elderly_male" (Lolo / old man)
   - "elderly_female" (Lola / old woman)
B. The emotion behind the line:
   - "normal" (default)
   - "whisper" (quiet, pabulong, secret)
   - "angry" (mad, shouting, tense)
   - "sad" (crying, depressed, soft)
   - "excited" (happy, shouting in joy, enthusiastic)

Return ONLY a valid JSON object with the following structure:
{{
  "translations": [
     {{"translated": "Tagalog translation here", "speaker": "male/female/child/elderly_male/elderly_female", "emotion": "normal/whisper/angry/sad/excited"}},
     ...
  ]
}}
Do NOT output markdown code blocks or introductory text.

Input English lines:
{json.dumps(sub_batch, ensure_ascii=False)}"""

                    genai.configure(api_key=self.api_keys[self.current_key_index])
                    
                    try:
                        response = self.model.generate_content(
                            prompt,
                            generation_config={"response_mime_type": "application/json"}
                        )
                    except Exception:
                        response = self.model.generate_content(prompt)

                    clean_text = response.text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]

                    data = json.loads(clean_text.strip())

                    # Gemini stable JSON parsing
                    if isinstance(data, dict):
                        if "translations" in data and isinstance(data["translations"], list):
                            sub_list = data["translations"]
                        elif "translated" in data:
                            sub_list = [data]
                        else:
                            lists = [v for v in data.values() if isinstance(v, list)]
                            sub_list = lists[0] if lists else []
                    elif isinstance(data, list):
                        sub_list = data
                    else:
                        sub_list = []

                    # Tiyaking may emotion at speaker keys
                    for item in sub_list:
                        if "emotion" not in item:
                            item["emotion"] = "normal"
                        if "speaker" not in item:
                            item["speaker"] = "male"

                    if len(sub_list) == len(sub_batch):
                        sub_batch_translated = sub_list
                        success = True
                        await asyncio.sleep(1.5)
                        break
                    else:
                        logger.warning(f"⚠️ Mismatch in sub-batch length ({len(sub_list)} vs {len(sub_batch)}). Retrying... (Attempt {attempt+1}/4)")
                        await asyncio.sleep(3)

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "Quota" in err_msg or "quota" in err_msg or "ResourceExhausted" in err_msg:
                        logger.warning(f"⚠️ Gemini Key #{self.current_key_index + 1} hit rate limit.")
                        if self.rotate_api_key():
                            logger.info("⏳ Waiting 8 seconds for IP-level cool-down before trying new key...")
                            await asyncio.sleep(8)
                        else:
                            wait_time = 25
                            logger.warning(f"⚠️ No other keys available. Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"⚠️ Gemini sub-batch parse error: {e}. Retrying... (Attempt {attempt+1}/4)")
                        await asyncio.sleep(3)

            if success:
                translated_results.extend(sub_batch_translated)
            else:
                logger.warning(f"🔄 Gemini failed for sub-batch {start_idx // SUB_BATCH_SIZE + 1}. Automatically falling back to Google Translate...")
                try:
                    google_translated = await self.translate_with_google(sub_batch)
                    if len(google_translated) == len(sub_batch):
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in google_translated]
                    else:
                        fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)
                except Exception as ex:
                    fallback_sub = [{"translated": t, "speaker": "male", "emotion": "normal"} for t in sub_batch]
                    translated_results.extend(fallback_sub)

        return translated_results

    async def translate_with_openai(self, texts: list[str], glossary="") -> list[dict]:
        """Translation at character/emotion analysis gamit ang OpenAI"""
        if not openai_available:
            logger.warning("OpenAI library is not installed.")
            return []

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("No OpenAI API key found in .env.")
            return []

        logger.info(f"📦 [OpenAI] Translating ALL {len(texts)} lines...")
        client = OpenAI(api_key=openai_api_key)

        glossary_rules = ""
        if glossary and glossary.strip():
            glossary_rules = f"\nGLOSSARY / DICTIONARY RULES (You MUST strictly follow these mappings):\n{glossary.strip()}"

        prompt = f"""You are an expert video dubbing translator and script analyst. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.
{glossary_rules}

Additionally, analyze the context of each line and detect:
A. The most likely speaker type for that line:
   - "male" (adult male)
   - "female" (adult female)
   - "child" (a young kid)
   - "elderly_male" (Lolo / old man)
   - "elderly_female" (Lola / old woman)
B. The emotion behind the line:
   - "normal" (default)
   - "whisper" (quiet, pabulong, secret)
   - "angry" (mad, shouting, tense)
   - "sad" (crying, depressed, soft)
   - "excited" (happy, shouting in joy, enthusiastic)

Return ONLY a valid JSON object with the following structure:
{{
  "translations": [
     {{"translated": "Tagalog translation here", "speaker": "male/female/child/elderly_male/elderly_female", "emotion": "normal/whisper/angry/sad/excited"}},
     ...
  ]
}}
Do NOT output anything else.

Input English lines:
{json.dumps(texts, ensure_ascii=False)}"""

        try:
            loop = asyncio.get_event_loop()
            def make_call():
                return client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )

            response = await loop.run_in_executor(None, make_call)
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            if isinstance(data, dict):
                if "translations" in data and isinstance(data["translations"], list):
                    translated_list = data["translations"]
                elif "translated" in data:
                    translated_list = [data]
                else:
                    lists = [v for v in data.values() if isinstance(v, list)]
                    translated_list = lists[0] if lists else []
            elif isinstance(data, list):
                translated_list = data
            else:
                translated_list = []

            for item in translated_list:
                if "speaker" not in item:
                    item["speaker"] = "male"
                if "emotion" not in item:
                    item["emotion"] = "normal"

            if len(translated_list) == len(texts):
                logger.info("✅ OpenAI translation successful!")
                return translated_list
            else:
                logger.warning("Mismatch in OpenAI translation length.")
                return []

        except Exception as e:
            logger.error(f"❌ OpenAI Error: {e}")
            return []

    async def translate_text(self, texts: list[str], translator="google", base_voice="fil-PH-AngeloNeural", glossary="") -> list[dict]:
        """Pinagsamang translator na may awtomatikong fallback sa isa't isa"""
        default_speaker = "male" if "Angelo" in base_voice else "female"
        translated = []
        
        if translator == "google":
            raw_texts = await self.translate_with_google(texts)
            translated = [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in raw_texts]
        elif translator == "openai":
            translated = await self.translate_with_openai(texts, glossary=glossary)
        elif translator == "groq":
            translated = await self.translate_with_groq(texts, glossary=glossary)
        elif translator == "huggingface":
            translated = await self.translate_with_huggingface(texts, glossary=glossary)
        else:
            translated = await self.batch_translate_to_tagalog(texts, glossary=glossary)

        # MGA AUTOMATIC FALLBACKS KUNG SAKALING MAY PUMALYA:
        if not translated:
            if translator == "google":
                if self.api_keys:
                    logger.info("🔄 Google Translate failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts, glossary=glossary)
                elif openai_available and os.getenv("OPENAI_API_KEY"):
                    logger.info("🔄 Google Translate failed. Falling back to OpenAI...")
                    translated = await self.translate_with_openai(texts, glossary=glossary)
            elif translator == "gemini":
                if google_translator_available:
                    logger.info("🔄 Gemini failed. Falling back to Google Translate...")
                    raw_texts = await self.translate_with_google(texts)
                    translated = [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in raw_texts]
                elif openai_available and os.getenv("OPENAI_API_KEY"):
                    logger.info("🔄 Gemini failed. Falling back to OpenAI...")
                    translated = await self.translate_with_openai(texts, glossary=glossary)
            elif translator == "openai":
                if google_translator_available:
                    logger.info("🔄 OpenAI failed. Falling back to Google Translate...")
                    raw_texts = await self.translate_with_google(texts)
                    translated = [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in raw_texts]
                elif self.api_keys:
                    logger.info("🔄 OpenAI failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts, glossary=glossary)
            elif translator == "groq":
                if google_translator_available:
                    logger.info("🔄 Groq failed. Falling back to Google Translate...")
                    raw_texts = await self.translate_with_google(texts)
                    translated = [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in raw_texts]
                elif self.api_keys:
                    logger.info("🔄 Groq failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts, glossary=glossary)
            elif translator == "huggingface":
                if google_translator_available:
                    logger.info("🔄 Hugging Face failed. Falling back to Google Translate...")
                    raw_texts = await self.translate_with_google(texts)
                    translated = [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in raw_texts]
                elif self.api_keys:
                    logger.info("🔄 Hugging Face failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts, glossary=glossary)

        if not translated:
            logger.error("❌ All translation engines failed. Keeping original English text.")
            return [{"translated": t, "speaker": default_speaker, "emotion": "normal"} for t in texts]

        return translated

    async def synthesize_tagalog(self, text: str, output_path: str, voice: str, rate: str, pitch: str, volume="+0%"):
        """Inayos na may Rate-Limit & VOLUME Protection para sa emosyon, at Google TTS (gTTS) support!"""
        # 1. Punctuation at Blank Filter
        if not re.search(r'[a-zA-Z0-9\u00c0-\u00ff\u0100-\u017f]', text):
            logger.warning(f"⚠️ Laktawan ang TTS para sa '{text}' dahil walang mababasang letra (Punctuation o Blank lamang).")
            return False

        # Kung Google Translate (gTTS) voice ang pinili
        if voice == "Google-Translate-TTS":
            for attempt in range(3):
                try:
                    from gtts import gTTS
                    tts = gTTS(text=text, lang='tl')
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: tts.save(output_path))
                    await asyncio.sleep(0.3)
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Google TTS Error (Subok {attempt+1}/3): {e}. Naghihintay ng 2.5s bago sumubok ulit...")
                    await asyncio.sleep(2.5)
            return False

        # 2. Standard Edge-TTS (Default)
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch, volume=volume)
                await communicate.save(output_path)
                
                # Cooldown bawat segment
                await asyncio.sleep(0.3)
                return True
            except Exception as e:
                logger.warning(f"⚠️ Edge-TTS Error (Subok {attempt+1}/3): {e}. Naghihintay ng 2.5s bago sumubok ulit...")
                await asyncio.sleep(2.5)
        
        logger.error(f"❌ Tuluyang pumalya ang Edge-TTS para sa linyang: '{text}'")
        return False

    async def process_video(self, input_path: str, output_path: str, whisper_model="base", voice="fil-PH-AngeloNeural", rate="+5%", pitch="+0Hz", translator="google", video_speed=1.0, multi_speaker=True, bgm_volume=0.15, glossary=""):
        try:
            if not os.path.exists(input_path):
                logger.error(f"File not found: {input_path}")
                return False

            logger.info(f"🚀 Starting English → Tagalog dubbed pipeline: {input_path}")
            logger.info(f"🎙️ Base Voice: {voice} | Speed: {rate} | Pitch: {pitch}")
            logger.info(f"🌐 Translator: {translator} | Video Speed: {video_speed}x | BGM/SFX Volume: {bgm_volume*100}%")

            # 1. Extract Audio
            audio_path = "temp_extracted.wav"
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar='16000', ac=1, vn=None)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info("✅ Audio extracted")

            # 2. Transcribe gamit ang Whisper
            model = self.load_whisper(whisper_model)
            result = model.transcribe(audio_path, language="en", fp16=False)
            
            segments = result.get("segments", [])
            if not segments:
                logger.warning("⚠️ Walang nakitang boses sa video.")
                return False

            logger.info(f"✅ Transcription done. Nakahanap ng {len(segments)} segments.")

            # 3. Translate with Glossary Support!
            english_texts = [seg["text"].strip() for seg in segments]
            tagalog_translations = await self.translate_text(english_texts, translator=translator, base_voice=voice, glossary=glossary)

            temp_files = []
            batch_mixed_files = []
            current_batch_streams = []
            
            BATCH_SIZE = 40

            # 4. Synthesize & Sync per Segment (Batch Mode)
            logger.info("🗣️ Synthesizing synchronized speech per segment...")
            for i, seg in enumerate(segments):
                text = english_texts[i]
                if not text:
                    continue

                start_time = seg["start"] / video_speed
                end_time = seg["end"] / video_speed
                original_duration = end_time - start_time
                
                # Kunin ang data ng segment (translation, speaker, at emotion)
                seg_data = tagalog_translations[i] if i < len(tagalog_translations) else {"translated": text, "speaker": "male", "emotion": "normal"}
                translated_text = seg_data.get("translated", text)
                speaker_type = seg_data.get("speaker", "male")
                emotion_type = seg_data.get("emotion", "normal")

                # Kunin ang boses at emosyon
                seg_voice, seg_rate, seg_pitch, seg_volume = self.get_voice_params(
                    speaker_type, 
                    base_voice=voice, 
                    rate_offset=rate, 
                    pitch_offset=pitch, 
                    multi_speaker=multi_speaker,
                    emotion=emotion_type
                )

                if i % 20 == 0 or i == len(segments) - 1:
                    logger.info(f"Processing segment {i+1}/{len(segments)} ({start_time:.2f}s -> {end_time:.2f}s) | Speaker: [{speaker_type.upper()}] | Emotion: [{emotion_type.upper()}]")

                seg_audio_path = f"temp_seg_{i}.wav"
                
                success = await self.synthesize_tagalog(translated_text, seg_audio_path, seg_voice, seg_rate, seg_pitch, volume=seg_volume)
                if not success:
                    continue
                
                temp_files.append(seg_audio_path)

                # Sukatin ang haba ng audio
                tts_duration = self.get_audio_duration(seg_audio_path)
                audio_input = ffmpeg.input(seg_audio_path).audio

                if tts_duration > original_duration and original_duration > 0.2:
                    speed_factor = tts_duration / original_duration
                    if speed_factor > 1.45:
                        speed_factor = 1.45
                    
                    if speed_factor > 1.05:
                        audio_input = audio_input.filter('atempo', speed_factor)

                delay_ms = int(start_time * 1000)
                delayed_stream = audio_input.filter('adelay', f"{delay_ms}|{delay_ms}")
                current_batch_streams.append(delayed_stream)

                if len(current_batch_streams) >= BATCH_SIZE or i == len(segments) - 1:
                    if current_batch_streams:
                        batch_idx = len(batch_mixed_files)
                        batch_file = f"temp_batch_{batch_idx}.wav"
                        batch_mixed_files.append(batch_file)
                        temp_files.append(batch_file)
                        
                        logger.info(f"🎛️ Mixing intermediate batch {batch_idx + 1} ({len(current_batch_streams)} streams)...")
                        mixed_batch = ffmpeg.filter(current_batch_streams, 'amix', inputs=len(current_batch_streams), normalize=0)
                        out_stream = ffmpeg.output(mixed_batch, batch_file)
                        ffmpeg.run(out_stream, overwrite_output=True, quiet=True)
                        
                        current_batch_streams = []

            # 5. Pagsamahin ang lahat ng intermediate batches
            logger.info("🎛️ Mixing all intermediate batch audio tracks...")
            if not batch_mixed_files:
                logger.error("❌ Walang audio stream ang matagumpay na na-synchronize.")
                return False

            batch_inputs = [ffmpeg.input(bf).audio for bf in batch_mixed_files]
            mixed_audio = ffmpeg.filter(batch_inputs, 'amix', inputs=len(batch_inputs), normalize=0)

            dubbed_audio_path = "temp_dubbed_tl.wav"
            out_stream = ffmpeg.output(mixed_audio, dubbed_audio_path)
            ffmpeg.run(out_stream, overwrite_output=True, quiet=True)
            logger.info("✅ Audio track synchronized successfully")

            # 6. Merge synchronized audio with video
            logger.info("🎥 Merging synchronized audio with video...")
            video = ffmpeg.input(input_path)
            audio_stream = ffmpeg.input(dubbed_audio_path).audio
            
            if bgm_volume > 0.0:
                logger.info(f"🎛️ Mixing original background BGM/SFX track at {bgm_volume*100}% volume...")
                orig_audio = video.audio
                if video_speed != 1.0:
                    orig_audio = orig_audio.filter('atempo', video_speed)
                orig_audio_lowered = orig_audio.filter('volume', bgm_volume)
                final_audio = ffmpeg.filter([orig_audio_lowered, audio_stream], 'amix', inputs=2, normalize=0)
            else:
                final_audio = audio_stream
            
            if video_speed != 1.0:
                logger.info(f"⚡ Applying video slow-down stretching to {video_speed}x speed...")
                video_stream = video.video.filter('setpts', f'PTS/{video_speed}')
                final = ffmpeg.output(video_stream, final_audio, output_path, vcodec='libx264', preset='veryfast', acodec='aac', shortest=None)
            else:
                final = ffmpeg.output(video.video, final_audio, output_path, vcodec='copy', acodec='aac', shortest=None)
            
            try:
                ffmpeg.run(final, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                logger.error("❌ FFmpeg failed during merging. Error details:")
                logger.error(e.stderr.decode('utf-8') if e.stderr else str(e))
                return False

            # 7. Cleanup Temporary Files
            logger.info("🧹 Cleaning up temporary files...")
            for temp_f in temp_files:
                try:
                    os.remove(temp_f)
                except Exception:
                    pass
            try:
                os.remove(audio_path)
                os.remove(dubbed_audio_path)
            except Exception:
                pass

            logger.info(f"🎉 SUCCESS! Tagalog dubbed video saved as: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
            return False

async def main():
    parser = argparse.ArgumentParser(description="English to Tagalog Dubbing - Dynamic timing sync & Multi-Speaker")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--whisper_model', default='base')
    parser.add_argument('--translator', default='google', choices=['google', 'gemini', 'openai', 'groq', 'huggingface'], help="Pangunahing translator")
    parser.add_argument('--voice', default='fil-PH-AngeloNeural', choices=['fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural'], help="Pumili ng boses")
    parser.add_argument('--rate', default='+5%')
    parser.add_argument('--pitch', default='+0Hz')
    parser.add_argument('--video_speed', type=float, default=1.0)
    parser.add_argument('--multi_speaker', type=bool, default=True)
    parser.add_argument('--bgm_volume', type=float, default=0.15)
    parser.add_argument('--glossary', default="")
    
    args = parser.parse_args()

    dubber = VideoDubber()
    await dubber.process_video(
        args.input, 
        args.output, 
        whisper_model=args.whisper_model, 
        voice=args.voice, 
        rate=args.rate, 
        pitch=args.pitch,
        translator=args.translator,
        video_speed=args.video_speed,
        multi_speaker=args.multi_speaker,
        bgm_volume=args.bgm_volume,
        glossary=args.glossary
    )

if __name__ == "__main__":
    asyncio.run(main())