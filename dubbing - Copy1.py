"""
English to Tagalog Video Dubbing - Multi-Translator Version (Google Translate, Gemini, OpenAI)
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
    logger.info("ℹ️ OpenAI library not installed. If you want to use OpenAI fallback, run 'pip install openai'")

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
        self.model = genai.GenerativeModel('gemini-2.5-flash') if gemini_api_key else None

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

    async def translate_with_google(self, texts: list[str]) -> list[str]:
        """Libre at Walang Limitasyong Google Translate (Walang API Key na kailangan)"""
        if not google_translator_available:
            logger.warning("deep-translator is not installed. Run 'pip install deep-translator'.")
            return []

        logger.info(f"📦 [Google Translate] Translating ALL {len(texts)} lines...")
        try:
            # Dahil ang deep-translator ay synchronous/blocking, patatakbuhin natin ito sa thread executor
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

    async def batch_translate_to_tagalog(self, texts: list[str]) -> list[str]:
        """Isang bagsakang translation gamit ang Gemini (Kailangan ng API Key)"""
        if not texts:
            return []

        if not self.model:
            logger.warning("No Gemini model available.")
            return []

        logger.info(f"📦 [Gemini] Translating ALL {len(texts)} dialogue lines in ONE request...")

        prompt = f"""You are an expert video dubbing translator. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.

CRITICAL INSTRUCTION:
Return ONLY a valid JSON array of strings containing the translations in the exact same order as the input.
Do NOT output anything else. No introductory text, no markdown formatting outside the JSON.

Input English lines:
{json.dumps(texts, ensure_ascii=False)}"""

        for attempt in range(3):
            try:
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

                translated_list = json.loads(clean_text.strip())

                if len(translated_list) == len(texts):
                    logger.info("✅ Gemini translation successful!")
                    return translated_list
                else:
                    logger.warning(f"⚠️ Mismatch in translation length. Retrying...")
                    await asyncio.sleep(3)

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "Quota" in err_msg or "quota" in err_msg:
                    logger.warning(f"⚠️ Gemini Rate limit hit. Waiting 20s before retry... (Attempt {attempt+1}/3)")
                    await asyncio.sleep(20)
                else:
                    logger.warning(f"⚠️ JSON parse error: {e}. Retrying...")
                    await asyncio.sleep(3)

        return []

    async def translate_with_openai(self, texts: list[str]) -> list[str]:
        """Translation gamit ang OpenAI (Kailangan ng API Key)"""
        if not openai_available:
            logger.warning("OpenAI library is not installed.")
            return []

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("No OpenAI API key found in .env.")
            return []

        logger.info(f"📦 [OpenAI] Translating ALL {len(texts)} dialogue lines in ONE request...")
        client = OpenAI(api_key=openai_api_key)

        prompt = f"""You are an expert video dubbing translator. Translate the following English dialogue lines into natural, conversational Tagalog.
Keep the exact same emotion, tone, and context.

Return ONLY a valid JSON array of strings containing the translations in the exact same order as the input.
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
                lists = [v for v in data.values() if isinstance(v, list)]
                translated_list = lists[0] if lists else []
            elif isinstance(data, list):
                translated_list = data
            else:
                translated_list = []

            if len(translated_list) == len(texts):
                logger.info("✅ OpenAI translation successful!")
                return translated_list
            else:
                logger.warning("Mismatch in OpenAI translation length.")
                return []

        except Exception as e:
            logger.error(f"❌ OpenAI Error: {e}")
            return []

    async def translate_text(self, texts: list[str], translator="google") -> list[str]:
        """Pinagsamang translator na may awtomatikong fallback sa isa't isa"""
        translated = []
        
        if translator == "google":
            translated = await self.translate_with_google(texts)
        elif translator == "openai":
            translated = await self.translate_with_openai(texts)
        else:
            translated = await self.batch_translate_to_tagalog(texts)

        # MGA AUTOMATIC FALLBACKS KUNG SAKALING MAY PUMALYA:
        if not translated:
            if translator == "google":
                if gemini_api_key:
                    logger.info("🔄 Google Translate failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts)
                elif openai_available and os.getenv("OPENAI_API_KEY"):
                    logger.info("🔄 Google Translate failed. Falling back to OpenAI...")
                    translated = await self.translate_with_openai(texts)
            elif translator == "gemini":
                if google_translator_available:
                    logger.info("🔄 Gemini failed. Falling back to Google Translate...")
                    translated = await self.translate_with_google(texts)
                elif openai_available and os.getenv("OPENAI_API_KEY"):
                    logger.info("🔄 Gemini failed. Falling back to OpenAI...")
                    translated = await self.translate_with_openai(texts)
            elif translator == "openai":
                if google_translator_available:
                    logger.info("🔄 OpenAI failed. Falling back to Google Translate...")
                    translated = await self.translate_with_google(texts)
                elif gemini_api_key:
                    logger.info("🔄 OpenAI failed. Falling back to Gemini...")
                    translated = await self.batch_translate_to_tagalog(texts)

        # Kung pare-parehong pumalya, panatilihin ang orihinal na English
        if not translated:
            logger.error("❌ All translation engines failed. Keeping original English text.")
            return texts

        return translated

    async def synthesize_tagalog(self, text: str, output_path: str, voice: str, rate: str, pitch: str):
        try:
            communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return False

    async def process_video(self, input_path: str, output_path: str, whisper_model="base", voice="fil-PH-AngeloNeural", rate="+5%", pitch="+0Hz", translator="google"):
        try:
            if not os.path.exists(input_path):
                logger.error(f"File not found: {input_path}")
                return False

            logger.info(f"🚀 Starting English → Tagalog dubbed pipeline: {input_path}")
            logger.info(f"🎙️ Using TTS Voice: {voice} | Speed: {rate} | Pitch: {pitch}")
            logger.info(f"🌐 Primary Translator: {translator}")

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

            # 3. Translate gamit ang napiling translator (Default: Google)
            english_texts = [seg["text"].strip() for seg in segments]
            tagalog_translations = await self.translate_text(english_texts, translator=translator)

            delayed_streams = []
            temp_files = []

            # 4. Synthesize & Sync per Segment
            logger.info("🗣️ Synthesizing synchronized speech per segment...")
            for i, seg in enumerate(segments):
                text = english_texts[i]
                if not text:
                    continue

                start_time = seg["start"]
                end_time = seg["end"]
                original_duration = end_time - start_time
                translated_text = tagalog_translations[i] if i < len(tagalog_translations) else text

                logger.info(f"Segment {i+1}/{len(segments)} ({start_time:.2f}s -> {end_time:.2f}s)")
                logger.info(f"  [EN]: {text}")
                logger.info(f"  [TL]: {translated_text}")

                seg_audio_path = f"temp_seg_{i}.wav"
                temp_files.append(seg_audio_path)

                success = await self.synthesize_tagalog(translated_text, seg_audio_path, voice, rate, pitch)
                if not success:
                    continue

                # Sukatin ang haba ng audio
                tts_duration = self.get_audio_duration(seg_audio_path)
                audio_input = ffmpeg.input(seg_audio_path).audio

                # I-speed up ang boses gamit ang 'atempo' kung lalampas sa orihinal na tagal
                if tts_duration > original_duration and original_duration > 0.2:
                    speed_factor = tts_duration / original_duration
                    if speed_factor > 1.45:
                        speed_factor = 1.45
                    
                    if speed_factor > 1.05:
                        logger.info(f"  ⚡ Speeding up segment {i+1} by {speed_factor:.2f}x to fit segment timing ({original_duration:.2f}s)")
                        audio_input = audio_input.filter('atempo', speed_factor)

                delay_ms = int(start_time * 1000)
                delayed_stream = audio_input.filter('adelay', f"{delay_ms}|{delay_ms}")
                delayed_streams.append(delayed_stream)

            if not delayed_streams:
                logger.error("❌ Walang audio stream ang matagumpay na na-synchronize.")
                return False

            # 5. Mix delayed segments
            logger.info("🎛️ Mixing all synchronized audio segments...")
            mixed_audio = ffmpeg.filter(delayed_streams, 'amix', inputs=len(delayed_streams), normalize=0)

            dubbed_audio_path = "temp_dubbed_tl.wav"
            out_stream = ffmpeg.output(mixed_audio, dubbed_audio_path)
            ffmpeg.run(out_stream, overwrite_output=True, quiet=True)
            logger.info("✅ Audio track synchronized successfully")

            # 6. Merge synchronized audio with video
            logger.info("🎥 Merging synchronized audio with video...")
            video = ffmpeg.input(input_path)
            audio = ffmpeg.input(dubbed_audio_path)
            
            final = ffmpeg.output(video.video, audio.audio, output_path, vcodec='copy', acodec='aac', shortest=None)
            
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
    parser = argparse.ArgumentParser(description="English to Tagalog Dubbing - Dynamic timing sync & Multi-Translator")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--whisper_model', default='base')
    
    # Ginawang default translator ang "google"
    parser.add_argument('--translator', default='google', choices=['google', 'gemini', 'openai'], 
                        help="Pangunahing translator na gagamitin (default: google)")
    
    parser.add_argument('--voice', default='fil-PH-AngeloNeural', 
                        choices=['fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural'], 
                        help="Pumili ng boses: fil-PH-AngeloNeural (Lalaki) o fil-PH-BlessicaNeural (Babae)")
    parser.add_argument('--rate', default='+5%', 
                        help="Bilis ng pagsasalita (hal. +0%%, +5%%, -10%%).")
    parser.add_argument('--pitch', default='+0Hz', 
                        help="Tono ng boses (hal. +0Hz, -3Hz para mas lumalim, o +2Hz para tuminis).")
    
    args = parser.parse_args()

    dubber = VideoDubber()
    await dubber.process_video(
        args.input, 
        args.output, 
        whisper_model=args.whisper_model, 
        voice=args.voice, 
        rate=args.rate, 
        pitch=args.pitch,
        translator=args.translator
    )

if __name__ == "__main__":
    asyncio.run(main())