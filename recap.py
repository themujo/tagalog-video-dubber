"""
AI Video Recap & Explanation Module with Timing Sync, gTTS, Auto-Failover, Quotes Cleaner & Tagalog Title Support (recap.py)
"""

import os
import sys
import json
import logging
import asyncio
import ffmpeg
import whisper
import google.generativeai as genai
import requests
import re
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Subukang i-import ang Groq at OpenAI para sa recap engine
try:
    from groq import Groq
    groq_available = True
except ImportError:
    groq_available = False

try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False

def sanitize_filename(name):
    """
    Tanggalin ang mga bawal na karakter sa Windows filename at gawing malinis.
    """
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    clean_name = clean_name.replace('"', '').replace("'", "").replace("“", "").replace("”", "")
    return clean_name.replace(" ", "_")

def clean_raw_json_string(raw_str):
    """
    Matalinong tagalinis ng JSON string upang maiwasan ang syntax errors mula sa unescaped double quotes sa loob ng teksto.
    """
    try:
        def replace_quotes(match):
            key = match.group(1)
            val = match.group(2)
            cleaned_val = val.replace('"', "'")
            return f'"{key}": "{cleaned_val}"'
        
        cleaned = re.sub(r'"(text|translated)":\s*"(.*?)"', replace_quotes, raw_str, flags=re.DOTALL)
        return cleaned
    except Exception as e:
        logger.warning(f"⚠️ Hindi nalinis ang raw JSON: {e}")
        return raw_str


class VideoRecapper:
    def __init__(self):
        self.whisper_model = None
        
    def load_whisper(self, model_name="base"):
        """Inayos upang mag-auto-detect ng GPU (CUDA) para sa napakabilis na pakikinig"""
        if self.whisper_model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Whisper {model_name} model on [{device.upper()}] for Recap...")
            self.whisper_model = whisper.load_model(model_name, device=device)
        return self.whisper_model

    def get_audio_duration(self, filepath):
        try:
            probe = ffmpeg.probe(filepath)
            return float(probe['format']['duration'])
        except Exception as e:
            logger.warning(f"Hindi makuha ang haba ng audio {filepath}: {e}")
            return 1.0

    def get_recap_prompt(self, dialogue_segments, total_duration, glossary):
        """Buuin ang pangunahing prompt para sa AI storytelling"""
        glossary_rules = f"\nGLOSSARY:\n{glossary.strip()}" if glossary and glossary.strip() else ""
        return f"""You are an expert anime and movie recap creator. Your task is to analyze the following video transcript with timestamps, and generate a highly engaging, dramatic, and simplified Tagalog storytelling narration script that is perfectly timed to the video.

Divide the story into key narrative segments (narrator blocks). For each block, specify the exact 'start' and 'end' timestamps matching when those events happen in the video, so that the narrator's voice matches the scenes.

CRITICAL INSTRUCTIONS:
1. Write the narration in rich, conversational, and dramatic Tagalog (recap style).
2. The output MUST be a valid JSON object containing a key "recap_narration", which is an array of objects.
3. Each object must contain exactly: "start" (float), "end" (float), and "text" (the Tagalog narration text).
4. The timestamps "start" and "end" must be realistic and fall within the video's total length (0 to {total_duration} seconds).
5. Output ONLY the raw JSON object. Do not output markdown code blocks.
6. You must strictly output the response as a valid json object.
7. CRITICAL: Keep each narration line short, concise, and punchy. Avoid overly long sentences. Ensure that the spoken duration of the text easily fits within the designated 'start' and 'end' time slot to prevent overlaps.
8. If you use any quotes inside the "text" string, you MUST use single quotes (') instead of double quotes (") to prevent JSON parsing errors.
{glossary_rules}

Input transcript with timestamps:
{json.dumps(dialogue_segments, ensure_ascii=False)}"""

    # ==================== MGA INDIBIDWAL NA RECAP CALLS ====================

    async def try_groq(self, dialogue_segments, total_duration, glossary):
        if not groq_available:
            return None
        key = os.getenv("GROQ_API_KEY")
        if not key:
            return None
        
        prompt = self.get_recap_prompt(dialogue_segments, total_duration, glossary)
        try:
            client = Groq(api_key=key)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            ))
            logger.info("✅ Recap generated successfully using Groq Llama 3.3!")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"⚠️ Groq Recap failed: {e}")
            return None

    async def try_gemini(self, dialogue_segments, total_duration, glossary):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return None
        
        prompt = self.get_recap_prompt(dialogue_segments, total_duration, glossary)
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-3.1-flash-lite')
            response = model.generate_content(prompt)
            logger.info("✅ Recap generated successfully using Gemini!")
            return response.text.strip()
        except Exception as e:
            logger.warning(f"⚠️ Gemini Recap failed: {e}")
            return None

    async def try_openai(self, dialogue_segments, total_duration, glossary):
        if not openai_available:
            return None
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        
        prompt = self.get_recap_prompt(dialogue_segments, total_duration, glossary)
        try:
            client = OpenAI(api_key=key)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            ))
            logger.info("✅ Recap generated successfully using OpenAI!")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"⚠️ OpenAI Recap failed: {e}")
            return None

    async def try_huggingface(self, dialogue_segments, total_duration, glossary):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return None
        
        prompt = self.get_recap_prompt(dialogue_segments, total_duration, glossary)
        model_id = "Qwen/Qwen2.5-72B-Instruct"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "inputs": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            "parameters": {"max_new_tokens": 4096, "return_full_text": False}
        }
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.post(api_url, headers=headers, json=payload, timeout=60))
            if response.status_code == 200:
                res_json = response.json()
                return res_json[0].get("generated_text", "").strip() if isinstance(res_json, list) else res_json.get("generated_text", "").strip()
            elif response.status_code == 503:
                wait_sec = response.json().get("estimated_time", 15)
                logger.warning(f"⏳ HF model is loading... Waiting {wait_sec}s for Recap...")
                await asyncio.sleep(wait_sec)
                response = await loop.run_in_executor(None, lambda: requests.post(api_url, headers=headers, json=payload, timeout=60))
                if response.status_code == 200:
                    res_json = response.json()
                    return res_json[0].get("generated_text", "").strip() if isinstance(res_json, list) else res_json.get("generated_text", "").strip()
            return None
        except Exception as e:
            logger.warning(f"⚠️ Hugging Face Recap failed: {e}")
            return None

    # ==================== MAIN TRANSLATION MANAGER ====================

    async def generate_tagalog_recap_json(self, dialogue_segments, total_duration, translator="groq", glossary=""):
        """
        Pinagsamang translator para sa Recap na may awtomatikong Failover Chain!
        """
        res = None
        
        # Subukan ang iyong napiling pangunahing translator
        if translator == "groq":
            res = await self.try_groq(dialogue_segments, total_duration, glossary)
        elif translator == "gemini":
            res = await self.try_gemini(dialogue_segments, total_duration, glossary)
        elif translator == "openai":
            res = await self.try_openai(dialogue_segments, total_duration, glossary)
        elif translator == "huggingface":
            res = await self.try_huggingface(dialogue_segments, total_duration, glossary)

        # 🔄 AUTOMATIC FAILOVER CHAIN KUNG SAKALING MAG-ERROR ANG NAPILI:
        if not res:
            logger.warning(f"⚠️ Selected recap engine '{translator}' failed. Activating auto-failover chain...")
            
            # 1. Subukan ang Hugging Face (Dahil ito ay walang limitasyon at napakalakas)
            if translator != "huggingface" and os.getenv("HF_TOKEN"):
                logger.info("🔄 Trying Hugging Face (Qwen 72B) as first fallback...")
                res = await self.try_huggingface(dialogue_segments, total_duration, glossary)
                
            # 2. Kung ayaw pa rin, subukan ang Gemini
            if not res and translator != "gemini" and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                logger.info("🔄 Trying Gemini as second fallback...")
                res = await self.try_gemini(dialogue_segments, total_duration, glossary)
                
            # 3. Kung ayaw pa rin, subukan ang OpenAI
            if not res and translator != "openai" and os.getenv("OPENAI_API_KEY") and openai_available:
                logger.info("🔄 Trying OpenAI (GPT-4o-mini) as third fallback...")
                res = await self.try_openai(dialogue_segments, total_duration, glossary)

        return res

    async def process_recap(self, video_path, whisper_model="base", translator="groq", glossary="", translated_title=""):
        try:
            if not os.path.exists(video_path):
                return None, f"❌ File not found: {video_path}"

            logger.info(f"🎬 Starting AI Video Recap pipeline for: {video_path}")

            # 1. Extract audio
            audio_path = "temp_recap_extract.wav"
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar='16000', ac=1, vn=None)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info("✅ Audio extracted for recap")

            total_duration = self.get_audio_duration(audio_path)

            # 2. Transcribe gamit ang Whisper (gamit ang GPU)
            model = self.load_whisper(whisper_model)
            result = model.transcribe(audio_path, language="en", fp16=False)
            segments = result.get("segments", [])

            try:
                os.remove(audio_path)
            except Exception:
                pass

            if not segments:
                return None, "⚠️ Walang nakitang boses o usapan sa video para gawan ng recap."

            # I-format ang dialogue segments na may timestamps para sa AI
            dialogue_segments = []
            for seg in segments:
                dialogue_segments.append({
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip()
                })

            logger.info(f"✅ Transcription done. Starting Tagalog Storytelling Recap ({translator})...")

            # 3. Generate Tagalog Recap JSON
            recap_json_str = await self.generate_tagalog_recap_json(dialogue_segments, total_duration, translator=translator, glossary=glossary)
            
            if not recap_json_str:
                return None, "❌ Error: Hindi nakakonekta sa kahit anong AI translator."

            # I-clean ang json string kung may markdown formatting
            if recap_json_str.startswith("```json"):
                recap_json_str = recap_json_str[7:]
            if recap_json_str.endswith("```"):
                recap_json_str = recap_json_str[:-3]

            # LINISIN ANG KAHIT ANONG UNESCAPED DOUBLE QUOTES BAGO I-PARSE ANG JSON!
            cleaned_json_str = clean_raw_json_string(recap_json_str.strip())

            try:
                data = json.loads(cleaned_json_str)
                if isinstance(data, list):
                    recap_list = data
                elif isinstance(data, dict):
                    recap_list = data.get("recap_narration", [])
                    if not recap_list:
                        # Fallback: Hanapin ang unang listahan sa loob ng dict
                        lists = [v for v in data.values() if isinstance(v, list)]
                        recap_list = lists[0] if lists else []
                else:
                    recap_list = []
            except Exception as e:
                logger.error(f"JSON Parse Error sa Recap: {e}. Cleaned content: {cleaned_json_str}")
                return None, f"❌ Error sa pagsusuri ng recap JSON: {e}\n\nRaw Content:\n{recap_json_str}"

            if not recap_list:
                return None, "⚠️ Walang nabuong narrative segments mula sa AI."

            logger.info(f"🗣️ Synthesizing {len(recap_list)} recap narration segments using gTTS...")
            
            # 4. Synthesize each narration segment using Google TTS (gTTS)
            temp_files = []
            delayed_streams = []
            recap_script_text = ""

            for idx, item in enumerate(recap_list):
                start_time = float(item.get("start", 0.0))
                end_time = float(item.get("end", 0.0))
                original_duration = end_time - start_time
                text_content = str(item.get("text", "")).strip()

                if not text_content:
                    continue

                recap_script_text += f"[{start_time:.2f}s - {end_time:.2f}s]\n🎙️ Narrator: {text_content}\n\n"

                seg_audio_path = f"temp_recap_seg_{idx}.wav"
                temp_files.append(seg_audio_path)

                # Synthesize gamit ang gTTS (Google Translate Voice)
                try:
                    tts = gTTS(text=text_content, lang='tl')
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: tts.save(seg_audio_path))
                except Exception as tts_err:
                    logger.error(f"gTTS segment {idx} failed: {tts_err}")
                    continue

                # Collision Avoidance
                if idx < len(recap_list) - 1:
                    next_start_time = float(recap_list[idx+1].get("start", 0.0))
                    max_allowed_duration = next_start_time - start_time
                else:
                    max_allowed_duration = original_duration

                if max_allowed_duration < original_duration:
                    max_allowed_duration = original_duration

                # Kuhanin ang totoong haba ng gTTS audio
                tts_duration = self.get_audio_duration(seg_audio_path)
                audio_input = ffmpeg.input(seg_audio_path).audio

                # I-speed up gamit ang 'atempo' kung ang audio ay lalampas sa magagamit na oras
                if tts_duration > max_allowed_duration and max_allowed_duration > 0.2:
                    speed_factor = tts_duration / max_allowed_duration
                    if speed_factor > 1.55:
                        speed_factor = 1.55
                    
                    if speed_factor > 1.05:
                        logger.info(f"  ⚡ Speeding up recap segment {idx+1} by {speed_factor:.2f}x to prevent overlap (Available: {max_allowed_duration:.2f}s)")
                        audio_input = audio_input.filter('atempo', speed_factor)

                # I-delay ang segment ayon sa kaniyang simula
                delay_ms = int(start_time * 1000)
                delayed_stream = audio_input.filter('adelay', f"{delay_ms}|{delay_ms}")
                delayed_streams.append(delayed_stream)

            if not delayed_streams:
                return None, "❌ Error: Walang audio track ang matagumpay na na-synthesize."

            # 5. Mix all delayed segments together
            logger.info("🎛️ Mixing all recap audio segments...")
            mixed_audio = ffmpeg.filter(delayed_streams, 'amix', inputs=len(delayed_streams), normalize=0)

            temp_recap_audio = "temp_recap_voice_mixed.wav"
            out_stream = ffmpeg.output(mixed_audio, temp_recap_audio)
            ffmpeg.run(out_stream, overwrite_output=True, quiet=True)

            # 6. Merge with the video (Tanging Tagalog Recap Voice lamang, walang original background sound)
            logger.info("🎥 Merging mixed recap audio with original video...")
            orig_name = os.path.basename(video_path)
            
            # Pangalan ng output file sa output folder
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 📌 INAYOS: Gagamitin na ngayon ang translated_title para sa pangalan ng video!
            if translated_title and translated_title.strip():
                clean_title = sanitize_filename(translated_title)
                output_filename = os.path.join(output_dir, f"{clean_title}_tagalog_recap.mp4")
            else:
                name_without_ext, _ = os.path.splitext(orig_name)
                if "temp_youtube_video" in name_without_ext:
                    output_filename = os.path.join(output_dir, "youtube_video_tagalog_recap.mp4")
                else:
                    clean_title = sanitize_filename(name_without_ext)
                    output_filename = os.path.join(output_dir, f"{clean_title}_tagalog_recap.mp4")

            if os.path.exists(output_filename):
                try: os.remove(output_filename)
                except Exception: pass

            video = ffmpeg.input(video_path)
            recap_audio = ffmpeg.input(temp_recap_audio).audio
            
            # Gagamit lamang ng Tagalog audio track nang direkta nang walang background sfx/music
            final_output = ffmpeg.output(video.video, recap_audio, output_filename, vcodec='libx264', preset='veryfast', acodec='aac', shortest=None)
            ffmpeg.run(final_output, overwrite_output=True, quiet=True)

            # 7. Cleanup temp files
            logger.info("🧹 Cleaning up temporary files...")
            for temp_f in temp_files:
                try: os.remove(temp_f)
                except Exception: pass
            try:
                os.remove(temp_recap_audio)
            except Exception:
                pass

            return output_filename, recap_script_text

        except Exception as e:
            logger.error(f"Recap processing error: {e}", exc_info=True)
            return None, f"❌ ERROR sa pagbuo ng recap: {str(e)}"
