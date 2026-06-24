"""
Web GUI for English to Tagalog Video Dubbing - Ultimate Workstation with Multi-Tabs & Recap Title Translation
"""

import os
import sys
import re
import logging
import asyncio
import pandas as pd
import gradio as gr
import ffmpeg
from dubbing import VideoDubber

# Subukang i-import ang recap module
try:
    from recap import VideoRecapper
    recap_available = True
    logging.info("✅ recap.py module loaded successfully")
except Exception as e:
    recap_available = False
    logging.warning(f"⚠️ Error loading recap.py: {e}")
    import traceback
    traceback.print_exc()

# Ayusin ang logging para makita sa terminal habang tumatakbo ang web app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Siguraduhing may "output" folder na nakalaan para sa mga natapos na video
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(name):
    """
    Tanggalin ang mga bawal na karakter sa Windows filename at gawing malinis.
    """
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    clean_name = clean_name.replace('"', '').replace("'", "").replace("“", "").replace("”", "")
    return clean_name.replace(" ", "_")

def format_srt_time(seconds):
    """
    I-convert ang seconds (float) sa standard SRT timestamp format: HH:MM:SS,mmm
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds % 1) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

def generate_srt_file(df, srt_filepath):
    """
    Awtomatikong gagawa ng .srt subtitle file batay sa DataFrame
    """
    try:
        with open(srt_filepath, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                idx = int(row["ID"])
                start = float(row["Start"])
                end = float(row["End"])
                text = str(row["Translated Tagalog"]).strip()
                
                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
                f.write(f"{text}\n\n")
        logger.info(f"✅ Subtitle file saved successfully: {srt_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error generating SRT: {e}")
        return False


# ==================== 📌 INAYOS: RECAP FUNCTIONS (Nawawala kanina) ====================

async def process_recap_gui(video_file, youtube_url, whisper_model, translator, glossary):
    """
    Function na nag-da-download ng video mula sa YouTube o gumagamit ng uploaded file,
    at tinatawag ang recap.py para gawan ito ng Tagalog Recap Story Video & Audio!
    """
    if not recap_available:
        yield None, None, "❌ ERROR: Hindi mahanap ang recap.py file sa iyong folder."
        return

    input_path = None
    temp_yt_file = "temp_recap_youtube_video.mp4"
    translated_title = ""
    
    dubber = VideoDubber() # Gamitin ang dubber para sa translation ng title
    
    # 1. Alamin kung YouTube URL o File Upload ang gagamitin at isalin ang pamagat sa Tagalog
    if youtube_url and youtube_url.strip():
        yield None, None, "📥 Kinukuha ang impormasyon at pamagat ng video mula sa YouTube..."
        try:
            if os.path.exists(temp_yt_file):
                try:
                    os.remove(temp_yt_file)
                except Exception:
                    pass
            
            loop = asyncio.get_event_loop()
            
            def get_yt_info():
                import yt_dlp
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(youtube_url.strip(), download=False)
                    return info.get('title', 'youtube_video')
            
            yt_title = await loop.run_in_executor(None, get_yt_info)
            
            yield None, None, f"🌐 Isinasalin sa Tagalog ang pamagat ng YouTube video: '{yt_title}'..."
            
            # Isalin ang pamagat sa Tagalog gamit ang glossary
            translated_title_list = await dubber.translate_text([yt_title], translator=translator, glossary=glossary)
            translated_title = translated_title_list[0].get("translated", yt_title)
            
            yield None, None, f"📥 Nagda-download mula sa YouTube: '{translated_title}'..."
            
            def download_yt():
                import yt_dlp
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'outtmpl': temp_yt_file,
                    'overwrites': True,
                    'quiet': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['ios', 'android']
                        }
                    }
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url.strip()])
            
            await loop.run_in_executor(None, download_yt)
            input_path = temp_yt_file
            logger.info("✅ YouTube video downloaded successfully")
            
        except Exception as e:
            yield None, None, f"❌ ERROR sa YouTube download: {str(e)}"
            return
    else:
        input_path = video_file
        if video_file:
            orig_name = os.path.basename(video_file)
            name_without_ext, _ = os.path.splitext(orig_name)
            
            yield None, None, f"🌐 Isinasalin sa Tagalog ang pamagat ng local file: '{name_without_ext}'..."
            
            # Isalin ang pangalan ng local file sa Tagalog gamit ang glossary
            translated_title_list = await dubber.translate_text([name_without_ext], translator=translator, base_voice="fil-PH-BlessicaNeural", glossary=glossary)
            translated_title = translated_title_list[0].get("translated", name_without_ext)

    # 2. Suriin kung may valid na input file
    if not input_path or not os.path.exists(input_path):
        yield None, None, "❌ ERROR: Walang video file. Mag-upload ng file o mag-paste ng YouTube link."
        return

    yield None, None, "⚙️ Nagsisimula na ang transcription sa Whisper at pagsusuri ng AI... (Maaaring tumagal ng ilang minuto, pakihintay)"
    await asyncio.sleep(0.5)

    recapper = VideoRecapper()
    
    try:
        # Tawagin ang process_recap at ipasa ang translated_title!
        recap_video_path, recap_script_text = await recapper.process_recap(
            video_path=input_path,
            whisper_model=whisper_model,
            translator=translator,
            glossary=glossary,
            translated_title=translated_title
        )
        
        # I-cleanup ang temporary downloaded file
        if os.path.exists(temp_yt_file):
            try:
                os.remove(temp_yt_file)
            except Exception:
                pass
        
        if recap_video_path and os.path.exists(recap_video_path):
            display_name = os.path.basename(recap_video_path)
            yield recap_video_path, recap_script_text, f"🎉 SUCCESS! Matagumpay na natapos ang AI Recap! Na-save bilang: 'output/{display_name}'"
        else:
            yield None, None, f"❌ FAILED: {recap_script_text}"
            
    except Exception as e:
        logger.error(f"Recap GUI Error: {e}", exc_info=True)
        yield None, None, f"❌ ERROR sa pagbuo ng recap: {str(e)}"


# ==================== DUBBING FUNCTIONS ====================

async def generate_dubbed_video_step(df, input_path, output_filename, voice, rate_val, pitch_hz_val, video_speed, multi_speaker, bgm_volume):
    """
    Step 2: Gamit ang na-edit na DataFrame, i-synthesize ang bawat linya, i-mix ang BGM, at i-merge sa video.
    """
    if df is None or df.empty:
        yield None, "❌ ERROR: Walang data sa talahanayan. Mangyaring mag-transcribe muna sa Step 1."
        return
    
    if not input_path or not os.path.exists(input_path):
        yield None, "❌ ERROR: Walang nakitang video file. Mangyaring mag-transcribe muna sa Step 1."
        return

    if not output_filename:
        output_filename = os.path.join(OUTPUT_DIR, "output_web_dubbed.mp4")

    yield None, "⚙️ Nagsisimula na ang pagbuo ng audio track para sa bawat segment..."
    await asyncio.sleep(1)

    # I-format ang speed (rate) at pitch para sa Edge-TTS
    rate = f"{rate_val:+d}%"
    pitch = f"{pitch_hz_val:+d}Hz"
    
    if os.path.exists(output_filename):
        try:
            os.remove(output_filename)
        except Exception:
            pass

    dubber = VideoDubber()
    temp_files = []
    batch_mixed_files = []
    current_batch_streams = []
    BATCH_SIZE = 40

    try:
        total_rows = len(df)
        
        # I-process ang bawat row na na-edit ng user sa DataFrame
        for index, row in df.iterrows():
            i = int(row["ID"]) - 1
            start_time = float(row["Start"]) / video_speed
            end_time = float(row["End"]) / video_speed
            original_duration = end_time - start_time
            
            translated_text = str(row["Translated Tagalog"]).strip()
            speaker_type = str(row["Speaker Type"]).strip().lower()
            emotion_type = str(row["Emotion"]).strip().lower() if "Emotion" in row else "normal"

            if not translated_text:
                continue

            if i % 15 == 0 or i == total_rows - 1:
                yield None, f"🗣️ [{i+1}/{total_rows}] Isinasalin sa boses: '{translated_text[:40]}...' ({speaker_type}) | Emotion: [{emotion_type.upper()}]"

            seg_audio_path = f"temp_seg_{i}.wav"
            
            # Kunin ang dynamic voice params batay sa karakter at emosyon
            seg_voice, seg_rate, seg_pitch, seg_volume = dubber.get_voice_params(
                speaker_type, 
                base_voice=voice, 
                rate_offset=rate, 
                pitch_offset=pitch, 
                multi_speaker=multi_speaker,
                emotion=emotion_type
            )

            success = await dubber.synthesize_tagalog(translated_text, seg_audio_path, seg_voice, seg_rate, seg_pitch, volume=seg_volume)
            if not success:
                continue
                
            temp_files.append(seg_audio_path)

            # Sukatin ang haba ng audio
            tts_duration = dubber.get_audio_duration(seg_audio_path)
            audio_input = ffmpeg.input(seg_audio_path).audio

            # I-speed up ang boses gamit ang 'atempo' kung lalampas pa rin sa orihinal na tagal
            if tts_duration > original_duration and original_duration > 0.2:
                speed_factor = tts_duration / original_duration
                if speed_factor > 1.45:
                    speed_factor = 1.45
                
                if speed_factor > 1.05:
                    audio_input = audio_input.filter('atempo', speed_factor)

            delay_ms = int(start_time * 1000)
            delayed_stream = audio_input.filter('adelay', f"{delay_ms}|{delay_ms}")
            current_batch_streams.append(delayed_stream)

            # Kapag umabot sa limitasyon ng BATCH_SIZE o index ay pinaka-huling segment
            if len(current_batch_streams) >= BATCH_SIZE or index == total_rows - 1:
                if current_batch_streams:
                    batch_idx = len(batch_mixed_files)
                    batch_file = f"temp_batch_{batch_idx}.wav"
                    batch_mixed_files.append(batch_file)
                    temp_files.append(batch_file)
                    
                    yield None, f"🎛️ Mixing intermediate batch {batch_idx + 1} ({len(current_batch_streams)} streams)..."
                    mixed_batch = ffmpeg.filter(current_batch_streams, 'amix', inputs=len(current_batch_streams), normalize=0)
                    out_stream = ffmpeg.output(mixed_batch, batch_file)
                    ffmpeg.run(out_stream, overwrite_output=True, quiet=True)
                    
                    current_batch_streams = []

        # 5. Pagsamahin ang lahat ng intermediate batches
        yield None, "🎛️ Pagsasama-samahin ang lahat ng batch tracks..."
        if not batch_mixed_files:
            yield None, "❌ ERROR: Walang audio stream ang matagumpay na na-synchronize."
            return

        batch_inputs = [ffmpeg.input(bf).audio for bf in batch_mixed_files]
        mixed_audio = ffmpeg.filter(batch_inputs, 'amix', inputs=len(batch_inputs), normalize=0)

        dubbed_audio_path = "temp_dubbed_tl.wav"
        out_stream = ffmpeg.output(mixed_audio, dubbed_audio_path)
        ffmpeg.run(out_stream, overwrite_output=True, quiet=True)

        # 6. Merge synchronized audio with video (BGM/SFX Mixer)
        yield None, "🎥 Pinagsasama na ang bagong audio sa orihinal na video..."
        video = ffmpeg.input(input_path)
        audio_stream = ffmpeg.input(dubbed_audio_path).audio
        
        # Awtomatikong i-mix ang original background music at sound effects!
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
            video_stream = video.video.filter('setpts', f'PTS/{video_speed}')
            final = ffmpeg.output(video_stream, final_audio, output_filename, vcodec='libx264', preset='veryfast', acodec='aac', shortest=None)
        else:
            final = ffmpeg.output(video.video, final_audio, output_filename, vcodec='copy', acodec='aac', shortest=None)
        
        try:
            ffmpeg.run(final, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            err_log = e.stderr.decode('utf-8') if e.stderr else str(e)
            yield None, f"❌ FFmpeg error: {err_log[:500]}"
            return

        # Awtomatikong bumuo ng .SRT subtitle file
        srt_filepath = output_filename.replace(".mp4", ".srt")
        try:
            with open(srt_filepath, 'w', encoding='utf-8') as f:
                for idx_row, row_data in df.iterrows():
                    idx_srt = int(row_data["ID"])
                    start_srt = float(row_data["Start"])
                    end_srt = float(row_data["End"])
                    text_srt = str(row_data["Translated Tagalog"]).strip()
                    
                    def format_time(seconds):
                        hrs = int(seconds // 3600)
                        mins = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        msecs = int((seconds % 1) * 1000)
                        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"
                        
                    f.write(f"{idx_srt}\n")
                    f.write(f"{format_time(start_srt)} --> {format_time(end_srt)}\n")
                    f.write(f"{text_srt}\n\n")
            logger.info(f"✅ Subtitle file saved: {srt_filepath}")
        except Exception as e:
            logger.error(f"Error generating SRT: {e}")

        # 7. Cleanup Temporary Files
        yield None, "🧹 Nililinis ang mga temporary files..."
        for temp_f in temp_files:
            try:
                os.remove(temp_f)
            except Exception:
                pass
        try:
            os.remove(dubbed_audio_path)
            if "temp_youtube_video" in orig_name:
                os.remove(input_path)
        except Exception:
            pass

        display_name = os.path.basename(output_filename)
        display_srt = os.path.basename(srt_filepath)
        yield output_filename, f"🎉 SUCCESS! Matagumpay na natapos ang pag-dub. Na-save sa folder na 'output' bilang: '{display_name}'!"
        
    except Exception as e:
        logger.error(f"Step 2 Error: {e}", exc_info=True)
        yield None, f"❌ ERROR sa pag-dub: {str(e)}"

# 📌 INAYOS: Kumpletong 12-arguments signature na nagbabalik ng eksaktong 6 values sa bawat solong 'yield'
async def transcribe_and_translate_step(video_file, youtube_url, whisper_model, translator, voice, speed_pct, pitch_hz, video_speed, multi_speaker, editor_mode, bgm_volume, glossary):
    """
    Step 1: I-transcribe, isalin ang bawat linya (may Glossary at Emotion support), at isalin ang pamagat ng video sa Tagalog.
    Kung naka-OFF ang editor_mode, dideretso ito agad sa pag-render ng LAHAT ng nasa pila (Bulk Queue Mode)!
    """
    # Ipunin ang lahat ng kailangang iproseso (Queue System)
    tasks = []
    temp_yt_file = "temp_youtube_video.mp4"
    dubber = VideoDubber()
    
    # Kuhanin ang maramihang YouTube links (isang link bawat linya)
    if youtube_url and youtube_url.strip():
        for line in youtube_url.strip().split('\n'):
            url = line.strip()
            if url:
                tasks.append({"type": "youtube", "source": url})
                
    # Kuhanin ang maramihang local video files
    if video_file:
        for file_obj in video_file:
            file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, 'name', None)
            if file_path and os.path.exists(file_path):
                tasks.append({"type": "file", "source": file_path})

    if not tasks:
        yield None, None, None, None, None, "❌ ERROR: Walang video o YouTube links para iproseso. Mag-upload ng mga files o mag-paste ng links."
        return

    total_tasks = len(tasks)

    # ==================== A. KUNG NAKA-ON ANG EDITOR MODE ====================
    # Ipoproseso lamang ang UNANG video sa pila para lumabas sa grid at ma-edit ng user
    if editor_mode:
        task = tasks[0]
        source_type = task["type"]
        source = task["source"]
        input_path = None
        output_filename = os.path.join(OUTPUT_DIR, "output_web_dubbed.mp4")
        video_title = "Video"

        if source_type == "youtube":
            yield None, None, None, None, None, "📥 Kinukuha ang impormasyon at pamagat mula sa YouTube..."
            try:
                if os.path.exists(temp_yt_file):
                    try:
                        os.remove(temp_yt_file)
                    except Exception:
                        pass
                
                loop = asyncio.get_event_loop()
                
                def get_yt_info():
                    import yt_dlp
                    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                        info = ydl.extract_info(source, download=False)
                        return info.get('title', 'youtube_video')
                
                yt_title = await loop.run_in_executor(None, get_yt_info)
                
                yield None, None, None, None, None, f"🌐 Isinasalin sa Tagalog ang pamagat ng YouTube video: '{yt_title}'..."
                
                # Isalin ang pamagat gamit ang Glossary!
                translated_title_list = await dubber.translate_text([yt_title], translator=translator, base_voice=voice, glossary=glossary)
                translated_title = translated_title_list[0].get("translated", yt_title)
                
                clean_title = sanitize_filename(translated_title)
                output_filename = os.path.join(OUTPUT_DIR, f"{clean_title}_tagalog_dubbed.mp4")
                video_title = yt_title
                
                yield None, None, None, None, None, f"📥 Nagda-download mula sa YouTube: '{clean_title}'..."
                
                def download_yt():
                    import yt_dlp
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
                        'outtmpl': temp_yt_file,
                        'overwrites': True,
                        'quiet': True,
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['ios', 'android']
                            }
                        }
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([source])
                
                await loop.run_in_executor(None, download_yt)
                input_path = temp_yt_file
                
            except Exception as e:
                yield None, None, None, None, None, f"❌ ERROR sa YouTube download: {str(e)}"
                return
        else:
            input_path = source
            orig_name = os.path.basename(source)
            name_without_ext, _ = os.path.splitext(orig_name)
            
            yield None, None, None, None, None, f"🌐 Isinasalin sa Tagalog ang pamagat: '{name_without_ext}'..."
            
            translated_title_list = await dubber.translate_text([name_without_ext], translator=translator, base_voice=voice, glossary=glossary)
            translated_title = translated_title_list[0].get("translated", name_without_ext)
            
            clean_title = sanitize_filename(translated_title)
            output_filename = os.path.join(OUTPUT_DIR, f"{clean_title}_tagalog_dubbed.mp4")
            video_title = orig_name

        yield None, None, None, None, None, "⚙️ Nagsisimula na ang transcription sa Whisper..."
        
        try:
            # Extract audio
            audio_path = "temp_extracted.wav"
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar='16000', ac=1, vn=None)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info("✅ Audio extracted")
            
            # Transcribe
            model = dubber.load_whisper(whisper_model)
            result = model.transcribe(audio_path, language="en", fp16=False)
            segments = result.get("segments", [])
            
            if not segments:
                yield None, None, None, None, None, "⚠️ Walang nakitang boses o usapan sa video."
                return
                
            # Translate ang bawat segment (May Glossary at Emotion support!)
            english_texts = [seg["text"].strip() for seg in segments]
            tagalog_translations = await dubber.translate_text(english_texts, translator=translator, base_voice=voice, glossary=glossary)
            
            # DataFrame for Editor
            rows = []
            for i, seg in enumerate(segments):
                start = f"{seg['start']:.2f}"
                end = f"{seg['end']:.2f}"
                orig = english_texts[i]
                
                seg_data = tagalog_translations[i] if i < len(tagalog_translations) else {"translated": orig, "speaker": "male", "emotion": "normal"}
                trans = seg_data.get("translated", orig)
                spk = seg_data.get("speaker", "male")
                emo = seg_data.get("emotion", "normal")
                
                rows.append({
                    "ID": i + 1,
                    "Start": start,
                    "End": end,
                    "Original English": orig,
                    "Translated Tagalog": trans,
                    "Speaker Type": spk,
                    "Emotion": emo
                })
            df = pd.DataFrame(rows)
            
            try:
                os.remove(audio_path)
            except Exception:
                pass
            
            display_name = os.path.basename(output_filename)
            # Tiyaking eksaktong 6 values ang ibinabalik sa yield
            yield df, input_path, output_filename, input_path, None, f"✅ TRANSCRIPTION SUCCESSFUL! Nakahanap ng {len(segments)} segments.\n\n👉 Naka-ON ang Interactive Editor Mode. Ipinoproseso ang unang video sa pila. Maaari mong i-edit ang talahanayan sa ibaba.\n📁 Ang pinal na video ay ise-save bilang: 'output/{display_name}'.\n\nPindutin ang 'Step 2: Generate Dubbed Video' kapag handa ka na."
            
        except Exception as e:
            yield None, None, None, None, None, f"❌ ERROR sa Transcription: {str(e)}"

    # ==================== B. KUNG NAKA-OFF ANG EDITOR MODE (BULK QUEUE) ====================
    # Awtomatikong ipoproseso ang LAHAT ng video sa pila isa-isa mula simula hanggang pinal na render!
    else:
        yield None, None, None, None, None, f"📋 [Bulk Mode] Nakahanap ng {total_tasks} na dubbing tasks sa pila. Nagsisimula na..."
        await asyncio.sleep(1.5)
        
        successful_files = []
        failed_tasks = []
        
        for idx, task in enumerate(tasks):
            task_num = idx + 1
            source_type = task["type"]
            source = task["source"]
            
            input_path = None
            output_filename = os.path.join(OUTPUT_DIR, f"output_batch_{task_num}.mp4")
            video_title = "Video"
            
            yield None, None, None, None, None, f"🔄 [{task_num}/{total_tasks}] Kinukuha ang impormasyon para sa: {source}..."
            
            # YouTube task processing
            if source_type == "youtube":
                try:
                    if os.path.exists(temp_yt_file):
                        try:
                            os.remove(temp_yt_file)
                        except Exception:
                            pass
                    
                    loop = asyncio.get_event_loop()
                    def get_yt_info():
                        import yt_dlp
                        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                            info = ydl.extract_info(source, download=False)
                            return info.get('title', 'youtube_video')
                    
                    yt_title = await loop.run_in_executor(None, get_yt_info)
                    
                    # Isalin ang pamagat ng YouTube video sa Tagalog (May Glossary support!)
                    translated_title_list = await dubber.translate_text([yt_title], translator=translator, base_voice=voice, glossary=glossary)
                    translated_title = translated_title_list[0].get("translated", yt_title)
                    clean_title = sanitize_filename(translated_title)
                    output_filename = os.path.join(OUTPUT_DIR, f"{clean_title}_tagalog_dubbed.mp4")
                    video_title = yt_title
                    
                    yield None, None, None, None, None, f"📥 [{task_num}/{total_tasks}] Nagda-download mula sa YouTube: '{clean_title}'..."
                    
                    def download_yt():
                        import yt_dlp
                        ydl_opts = {
                            'format': 'best[ext=mp4]/best',
                            'outtmpl': temp_yt_file,
                            'overwrites': True,
                            'quiet': True,
                            'extractor_args': {
                                'youtube': {
                                    'player_client': ['ios', 'android']
                                }
                            }
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([source])
                    
                    await loop.run_in_executor(None, download_yt)
                    input_path = temp_yt_file
                except Exception as e:
                    logger.error(f"Failed to process YouTube task: {e}")
                    failed_tasks.append(f"YouTube: {source} (Error: {str(e)})")
                    continue
            # Local File task processing
            else:
                orig_name = os.path.basename(source)
                name_without_ext, _ = os.path.splitext(orig_name)
                
                # Isalin ang pamagat (May Glossary support!)
                translated_title_list = await dubber.translate_text([name_without_ext], translator=translator, base_voice=voice, glossary=glossary)
                translated_title = translated_title_list[0].get("translated", name_without_ext)
                clean_title = sanitize_filename(translated_title)
                output_filename = os.path.join(OUTPUT_DIR, f"{clean_title}_tagalog_dubbed.mp4")
                input_path = source
                video_title = orig_name
                
            yield None, None, None, None, None, f"🎙️ [{task_num}/{total_tasks}] Isinasalin at dinee-dub: '{video_title}' -> '{output_filename}'..."
            await asyncio.sleep(1)
            
            try:
                # Extract Audio
                audio_path = "temp_extracted.wav"
                stream = ffmpeg.input(input_path)
                stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ar='16000', ac=1, vn=None)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Transcribe
                model = dubber.load_whisper(whisper_model)
                result = model.transcribe(audio_path, language="en", fp16=False)
                segments = result.get("segments", [])
                
                if not segments:
                    failed_tasks.append(f"Video: {video_title} (Walang boses na nahanap)")
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                    continue
                    
                english_texts = [seg["text"].strip() for seg in segments]
                tagalog_translations = await dubber.translate_text(english_texts, translator=translator, base_voice=voice, glossary=glossary)
                
                # Buuin ang temporary DataFrame (Kasama ang emotion)
                rows = []
                for i, seg in enumerate(segments):
                    start = f"{seg['start']:.2f}"
                    end = f"{seg['end']:.2f}"
                    orig = english_texts[i]
                    seg_data = tagalog_translations[i] if i < len(tagalog_translations) else {"translated": orig, "speaker": "male", "emotion": "normal"}
                    rows.append({
                        "ID": i + 1,
                        "Start": start,
                        "End": end,
                        "Original English": orig,
                        "Translated Tagalog": seg_data.get("translated", orig),
                        "Speaker Type": seg_data.get("speaker", "male"),
                        "Emotion": seg_data.get("emotion", "normal")
                    })
                df = pd.DataFrame(rows)
                
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
                
                # Patakbuhin ang Step 2 (Synthesis, BGM Mix & Subtitle) para sa task na ito
                async for output_video_file, status_msg in generate_dubbed_video_step(df, input_path, output_filename, voice, speed_pct, pitch_hz, video_speed, multi_speaker, bgm_volume):
                    yield df, input_path, output_filename, input_path, output_video_file, f"[{task_num}/{total_tasks}] {status_msg}"
                
                if os.path.exists(output_filename):
                    successful_files.append(output_filename)
                    
            except Exception as e:
                logger.error(f"Error in Bulk process for {video_title}: {e}", exc_info=True)
                failed_tasks.append(f"Video: {video_title} (Error: {str(e)})")

        # Pinal na Bulk Summary
        summary_msg = f"🎉 BULK SUCCESS! Matagumpay na natapos ang queue.\n\n✅ Matagumpay na na-dub: {len(successful_files)}/{total_tasks}\n"
        for sf in successful_files:
            summary_msg += f" - {os.path.basename(sf)}\n"
        if failed_tasks:
            summary_msg += f"\n❌ May isyu sa: {len(failed_tasks)}\n"
            for ft in failed_tasks:
                summary_msg += f" - {ft}\n"
                
        last_file = successful_files[-1] if successful_files else None
        yield None, None, None, None, last_file, summary_msg

def on_segment_select(df, evt: gr.SelectData):
    if df is None or df.empty:
        return ""
    row_idx = evt.index[0]
    row = df.iloc[row_idx]
    start = row["Start"]
    end = row["End"]
    orig = row["Original English"]
    emo = row["Emotion"] if "Emotion" in row else "normal"
    return f"⏱️ Timing ng Piniling Segment: {start}s hanggang {end}s | Emotion: [{emo.upper()}] | Original: '{orig}'"


# ==================== GRADIO WORKSTATION LAYOUT ====================

with gr.Blocks(title="Tagalog Video Dubbing Workstation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Tagalog AI Video Dubbing Workstation (Cinema Edition with Bulk Support)
        **Step 1:** I-upload ang isa o maramihang local videos o YouTube links. **Step 2:** Kung naka-OFF ang Editor, awtomatikong dideretso sa pag-dub ng LAHAT ng pila, isasalin ang kanilang mga pamagat sa Tagalog, at imi-mix ang orihinal na BGM/SFX [1]!
        """
    )
    
    # State para i-save ang path ng video at output filename sa buong session
    input_video_state = gr.State()
    output_filename_state = gr.State()
    
    with gr.Tab("🎙️ AI Video Dubber (Workstation)"):
        with gr.Row():
            # Kaliwang Hanay: Settings at Uploads
            with gr.Column(scale=1):
                gr.Markdown("### 1. Bulk Input Sources (Pwedeng gamitin pareho)")
                # Maramihang file uploader
                input_video = gr.File(
                    label="Mag-upload ng Maramihang Local Video Files (Drag & Drop multiple files)",
                    file_count="multiple",
                    file_types=[".mp4", ".avi", ".mkv", ".mov"]
                )
                # Maramihang YouTube links input (isang link bawat linya)
                youtube_url = gr.Textbox(
                    lines=5,
                    label="O I-paste ang mga YouTube URL (Isang link bawat linya para sa Bulk Dubbing)", 
                    placeholder="https://www.youtube.com/watch?v=aaa\nhttps://www.youtube.com/watch?v=bbb"
                )
                
                gr.Markdown("### 2. Settings")
                with gr.Row():
                    whisper_model = gr.Dropdown(
                        choices=["base", "small", "medium", "turbo"], 
                        value="medium", 
                        label="Whisper Model"
                    )
                    translator = gr.Dropdown(
                        choices=["google", "gemini", "openai", "groq", "huggingface"], # Dinagdag ang 'huggingface' option
                        value="google", 
                        label="Translation Engine"
                    )
                
                with gr.Row():
                    video_speed = gr.Slider(
                        minimum=0.7, 
                        maximum=1.0, 
                        value=1.0, 
                        step=0.05, 
                        label="Video Speed (0.90 = 10%% Mas Mabagal)"
                    )
                    
                multi_speaker = gr.Checkbox(
                    value=True, 
                    label="Multi-Speaker / Character Mode (Awtomatikong gayahin ang boses ng Lalaki, Babae, Bata, Lolo, o Lola)"
                )
                
                # Checkbox para sa Editor Mode ON/OFF
                editor_mode = gr.Checkbox(
                    value=True, 
                    label="Interactive Editor Mode (I-on kung gusto mong i-edit ang bawat salita bago i-render)"
                )
                
                # BGM Mixer Slider
                bgm_volume = gr.Slider(
                    minimum=0.0, 
                    maximum=0.5, 
                    value=0.15, 
                    step=0.05, 
                    label="Original Video BGM/SFX Volume (0.15 = 15%% volume ng orihinal na kanta/sound effects sa background)"
                )
                
                # Translation Glossary
                glossary = gr.Textbox(
                    lines=3, 
                    label="Custom Translation Glossary (English = Tagalog per line)", 
                    placeholder="Xiao Han = Xiao Han\nimmortal realm = mundo ng mga walang kamatayan\nbruises = pasa"
                )
                
                gr.Markdown("### Voice Customization (Para sa Single Speaker)")
                voice = gr.Dropdown(
                    choices=["fil-PH-AngeloNeural", "fil-PH-BlessicaNeural", "Google-Translate-TTS"], 
                    value="fil-PH-AngeloNeural", 
                    label="Boses ng Tagapagsalita (Voice) - Tandaan: Ang Google-Translate-TTS ay walang character/emotion support"
                )
                
                with gr.Row():
                    speed_pct = gr.Slider(
                        minimum=-30, 
                        maximum=30, 
                        value=5, 
                        step=1, 
                        label="Speed / Bilis (%)"
                    )
                    pitch_hz = gr.Slider(
                        minimum=-15, 
                        maximum=15, 
                        value=0, 
                        step=1, 
                        label="Pitch / Tono (Hz)"
                    )
                    
                transcribe_btn = gr.Button("🚀 Start Dubbing Process (Step 1 / Full)", variant="primary")
                
            # Kanang Hanay: Video Preview at Status
            with gr.Column(scale=1):
                gr.Markdown("### 2. Video Preview Player")
                preview_video = gr.Video(label="Oorihinal na Video Preview (Gamitin para i-check ang timing)")
                status_output = gr.Textbox(
                    lines=5, 
                    label="Progress / Status Output", 
                    interactive=False, 
                    placeholder="Naghihintay na i-transcribe ang video..."
                )
                
        gr.Markdown("---")
        gr.Markdown("### 📝 Interactive Translation, Speaker & Emotion Editor")
        gr.Markdown("*Maaari mong baguhin ang Translated Tagalog, Speaker Type, at **Emotion** cell sa talahanayan sa ibaba sa pamamagitan ng pag-double click dito. (Allowed emotions: normal, whisper, angry, sad, excited)*")
        
        timestamp_indicator = gr.Textbox(
            label="Piniling Segment Timing Indicator", 
            value="⏱️ Pumili ng segment sa talahanayan sa ibaba para makita ang timing...", 
            interactive=False
        )
        
        # Spreadsheet-like Dataframe (Kasama ang Emotion column)
        editor_df = gr.Dataframe(
            headers=["ID", "Start", "End", "Original English", "Translated Tagalog", "Speaker Type", "Emotion"],
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            interactive=True,
            type="pandas",
            wrap=True
        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                generate_dub_btn = gr.Button("🎙️ 2. Generate Dubbed Video", variant="secondary")
            with gr.Column():
                output_video = gr.Video(label="Dito Lalabas ang Tagalog Dubbed Video")

    # Step 1 Event Binding (12 inputs at 6 outputs)
    transcribe_btn.click(
        fn=transcribe_and_translate_step,
        inputs=[input_video, youtube_url, whisper_model, translator, voice, speed_pct, pitch_hz, video_speed, multi_speaker, editor_mode, bgm_volume, glossary],
        outputs=[editor_df, input_video_state, output_filename_state, preview_video, output_video, status_output]
    )
    
    # Click event sa talahanayan para i-update ang timing indicator kapag pumipili ang user
    editor_df.select(
        fn=on_segment_select,
        inputs=[editor_df],
        outputs=[timestamp_indicator]
    )
    
    # Step 2 Event Binding (9 inputs at 2 outputs)
    generate_dub_btn.click(
        fn=generate_dubbed_video_step,
        inputs=[editor_df, input_video_state, output_filename_state, voice, speed_pct, pitch_hz, video_speed, multi_speaker, bgm_volume],
        outputs=[output_video, status_output]
    )

    # 📌 TAB 2: ANG BAGONG DEDICATED AI VIDEO RECAP TAB!
    with gr.Tab("📝 AI Video Recap & Explanation (Beta)"):
        gr.Markdown(
            """
            ### 📝 AI Video Recap / Story Teller
            I-upload ang video o i-paste ang YouTube link, at awtomatikong susuriin at ipapaliwanag ng AI ang takbo ng kwento (Plot Recap) sa dramatikong wikang Tagalog [1]!
            """
        )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Upload Video or Link")
                recap_video = gr.Video(label="I-upload ang English Video (MP4)")
                recap_yt_url = gr.Textbox(
                    lines=3,
                    label="O I-paste ang YouTube URL (Pang-isahang link lamang)", 
                    placeholder="https://www.youtube.com/watch?v=..."
                )
                
                gr.Markdown("### 2. Settings")
                with gr.Row():
                    recap_whisper = gr.Dropdown(
                        choices=["base", "small", "medium", "turbo"], 
                        value="medium", 
                        label="Whisper Model"
                    )
                    recap_translator = gr.Dropdown(
                        choices=["google", "gemini", "openai", "groq", "huggingface"], 
                        value="groq", 
                        label="Recap Engine (Inirerekomenda: groq / gemini)"
                    )
                
                recap_glossary = gr.Textbox(
                    lines=3, 
                    label="Custom Vocabulary (Glossary)", 
                    placeholder="Xiao Han = Xiao Han"
                )
                
                recap_btn = gr.Button("📝 Generate Tagalog Story Recap", variant="primary")
                
            with gr.Column():
                gr.Markdown("### 3. Output Explanation Story")
                recap_output_video = gr.Video(label="Dito Lalabas ang Lip-Synced Tagalog Video")
                recap_output_text = gr.Textbox(
                    lines=15, 
                    label="AI Tagalog Story Recap / Explanation Narration", 
                    interactive=False, 
                    placeholder="Dito lalabas ang dramatikong kwento/recap..."
                )
                
        # Bind Event para sa Recap Tab
        recap_btn.click(
            fn=process_recap_gui,
            inputs=[recap_video, recap_yt_url, recap_whisper, recap_translator, recap_glossary],
            outputs=[recap_output_video, recap_output_text, status_output] # Gagamit ng shared status box
        )

if __name__ == "__main__":
    demo.queue()
    # Awtomatikong i-enable ang share=True kung tumatakbo sa loob ng Google Colab (Linux Server)
    is_colab = os.path.exists("/content")
    demo.launch(
