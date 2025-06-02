# cekviral_project/app/services/content_analyzer.py
from bs4 import BeautifulSoup
import re
import os
import subprocess
import logging
import asyncio
import torch
import torchaudio
# import soundfile as sf # Dikomentari karena tidak terlihat digunakan, hapus jika memang tidak perlu
from transformers import pipeline

from app.core.config import settings 

logger = logging.getLogger(__name__)

# Inisialisasi ASR pipeline sekali saja saat aplikasi startup
asr_pipeline = None
try:
    device = 0 if torch.cuda.is_available() else -1 # Gunakan GPU jika tersedia
    if device == 0:
        logger.info("CUDA (GPU) terdeteksi. ASR pipeline akan menggunakan GPU.")
    else:
        logger.warning("CUDA (GPU) tidak terdeteksi. ASR pipeline akan menggunakan CPU (mungkin lebih lambat).")

    asr_pipeline = pipeline("automatic-speech-recognition", 
                            model="indonesian-nlp/wav2vec2-large-xlsr-indonesian",
                            device=device)
    logger.info(">>> ASR PIPELINE (Wav2Vec2 Indonesian) BERHASIL DIMUAT dan siap digunakan. <<<")
except Exception as e:
    logger.error(f">>> GAGAL MEMUAT ASR PIPELINE: {e}. Fitur Speech-to-text tidak akan berfungsi. <<<", exc_info=True)
    asr_pipeline = None # Pastikan tetap None jika gagal


def extract_text_from_html(html_content: str) -> str | None:
    if not html_content or not isinstance(html_content, str):
        logger.warning("Input html_content untuk extract_text_from_html kosong atau bukan string.")
        return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        for tag_to_remove in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "iframe", "img", "svg", "figcaption", "figure", "noscript", "link", "meta"]):
            tag_to_remove.decompose()

        main_content_selectors = [
            'div[itemprop="articleBody"]', 'article[itemprop="articleBody"]',
            'div.entry-content', 'div.td-post-content', 'div.post-content', 
            'div.article-content', 'div.story-content', 'div.content',
            'article', 'main', 
            'div[role="main"]', 'section[role="main"]',
            'div.read__content', 'div.detail-content', 'div.post-body', 
            'div.story-body', 'div.post-detail', 'div.body_artikel', 
            'div.section_detail_content', 
            'div[class*="article-body"]', 'div[class*="post-content"]', 
            'div[class*="entry-content"]', 'div[class*="main-content"]',
            'div[class*="text-content"]', 'div[class*="content__body"]'
        ]
        
        main_article_element = None
        for selector in main_content_selectors:
            main_article_element = soup.select_one(selector)
            if main_article_element:
                logger.debug(f"Main content found with selector: {selector}")
                break
        
        article_text_parts = []
        target_element = main_article_element if main_article_element else soup.body
        
        if target_element:
            for element in target_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div']):
                if element.name in ['div', 'span'] and (element.find(['p','h1','h2','h3','h4','h5','h6','li'])):
                    continue 
                text = element.get_text(separator=' ', strip=True)
                if text and len(text.split()) > 3 and re.search(r'[a-zA-Z]{2,}', text): 
                    article_text_parts.append(text)
        
        if not article_text_parts and not main_article_element:
             all_text = soup.get_text(separator=' ', strip=True)
             if all_text:
                article_text_parts.append(all_text)

        title_tag = soup.find('title')
        page_title = title_tag.get_text(strip=True) if title_tag else ''
        
        meta_description_tag = soup.find('meta', attrs={'name': 'description'})
        meta_desc_content = meta_description_tag.get('content', '').strip() if meta_description_tag else ''

        full_extracted_text_parts = []
        if page_title:
            full_extracted_text_parts.append(page_title + ".")
        if meta_desc_content and meta_desc_content != page_title:
            full_extracted_text_parts.append(meta_desc_content + ".")

        unique_paragraphs = []
        seen_paragraphs = set()
        for p_text in article_text_parts:
            if p_text not in seen_paragraphs:
                unique_paragraphs.append(p_text)
                seen_paragraphs.add(p_text)
        full_extracted_text_parts.extend(unique_paragraphs)
        
        final_text = ' '.join(part for part in full_extracted_text_parts if part)
        final_text = re.sub(r'\s+', ' ', final_text).strip()

        if final_text:
            logger.debug(f"Extracted text length: {len(final_text)}")
            return final_text
        else:
            logger.warning("No significant text could be extracted from HTML.")
            return None

    except Exception as e:
        logger.error(f"Gagal mengekstrak teks dari HTML: {e}", exc_info=True)
        return None

# Fungsi detect_content_type_from_html(html_content: str) -> str:
# Telah dikomentari/dihapus karena keputusan utama sekarang ada di endpoints.py 
# menggunakan is_direct_video_platform_url dari helpers.py


async def convert_video_to_text(video_url: str) -> str | None:
    if not asr_pipeline:
        logger.warning("ASR pipeline tidak dimuat. Tidak dapat melakukan Speech-to-Text.")
        return "Maaf, fitur transkripsi suara tidak tersedia karena model ASR gagal dimuat."

    temp_dir = settings.YDL_TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_filename = f"temp_audio_{os.urandom(4).hex()}.wav"
    audio_path = os.path.join(temp_dir, audio_filename)
    
    try:
        logger.info(f"Memeriksa keberadaan yt-dlp dan ffmpeg...")
        await asyncio.to_thread(subprocess.run, ['yt-dlp', '--version'], check=True, capture_output=True, text=True, timeout=10)
        await asyncio.to_thread(subprocess.run, ['ffmpeg', '-version'], check=True, capture_output=True, text=True, timeout=10)
        logger.info("yt-dlp dan ffmpeg ditemukan.")
    except FileNotFoundError:
        logger.error("yt-dlp atau FFmpeg tidak ditemukan. Mohon pastikan keduanya terinstal dan ada di PATH sistem.")
        return "Maaf, fitur transkripsi suara tidak tersedia karena aplikasi tidak dapat menemukan alat bantu yang diperlukan (yt-dlp/ffmpeg)."
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Error atau timeout saat memeriksa versi yt-dlp/FFmpeg: {e}")
        return "Maaf, terjadi masalah saat memeriksa alat bantu transkripsi suara."

    transcribed_text = None
    try:
        logger.info(f"Mulai mengunduh audio dari {video_url} ke {audio_path}")
        
        process = await asyncio.to_thread(
            subprocess.run,
            ['yt-dlp', '-x', '--audio-format', 'wav', '-o', audio_path, video_url, '--socket-timeout', '30', '--retries', '3', '--no-check-certificate'],
            capture_output=True, text=True, check=False, timeout=900 
        )

        if process.returncode != 0:
            error_output = process.stderr.strip()
            stdout_output = process.stdout.strip() # Tangkap juga stdout untuk beberapa pesan error yt-dlp
            logger.error(f"yt-dlp gagal mengunduh audio dari {video_url}. Kode keluar: {process.returncode}. Stderr: {error_output}. Stdout: {stdout_output}")
            
            combined_output = error_output + " " + stdout_output # Gabungkan untuk pencarian frasa error
            if "Private video" in combined_output: return "Maaf, video ini adalah video pribadi dan tidak dapat diakses."
            elif "No video formats" in combined_output or "no suitable format" in combined_output: return "Maaf, tidak ada format audio yang dapat diunduh dari video ini."
            elif "yt.is_blocked" in combined_output or "unavailable" in combined_output.lower(): return "Maaf, video ini tidak tersedia atau pengunduhan diblokir oleh platform."
            elif "Invalid URL" in combined_output or "Unsupported URL" in combined_output: return "Maaf, URL video tidak valid atau tidak didukung." # Tambahkan Unsupported URL
            elif "copyright" in combined_output.lower(): return "Maaf, pengunduhan video ini diblokir karena masalah hak cipta."
            elif "ERROR:" in combined_output.upper(): return "Maaf, terjadi kesalahan saat mengunduh audio dari video." # Lebih generik jika ada "ERROR:"
            return "Maaf, gagal mengunduh audio dari video tersebut. Silakan coba video lain atau pastikan link valid."

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"File audio tidak ditemukan atau kosong ({audio_path}) setelah proses unduh dari {video_url}.")
            return "Maaf, audio dari video tidak dapat diunduh atau file audio kosong."

        logger.info(f"Audio berhasil diunduh: {audio_path}, ukuran: {os.path.getsize(audio_path)} bytes.")

        speech_array, sampling_rate = await asyncio.to_thread(torchaudio.load, audio_path)
        
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            speech_array = resampler(speech_array)
            logger.info(f"Audio di-resample dari {sampling_rate}Hz ke 16000Hz.")
        
        if speech_array.shape[0] > 1: 
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            logger.info("Audio diubah ke mono.")

        duration_seconds = speech_array.shape[1] / 16000 
        if duration_seconds < 1.0: 
            logger.warning(f"Audio terlalu pendek ({duration_seconds:.2f}s) untuk URL: {video_url}")
            return "Maaf, video terlalu pendek atau tidak berisi suara yang jelas untuk ditranskripsi."
        
        # --- PERUBAHAN DI SINI: Hapus return_timestamps dari pemanggilan pipeline ---
        transcription_result = await asyncio.to_thread(
            asr_pipeline, 
            speech_array.squeeze().numpy(), 
            chunk_length_s=30, 
            batch_size=8 
            # return_timestamps=False, # DIHAPUS untuk menghindari ValueError
        )
        transcribed_text = transcription_result['text'] if isinstance(transcription_result, dict) else str(transcription_result) # Pastikan string
        
        if not transcribed_text or not transcribed_text.strip():
            logger.warning(f"Transkripsi menghasilkan teks kosong. URL: {video_url}")
            return "Maaf, tidak ada obrolan atau suara yang jelas terdeteksi di video ini."
        
        logger.info(f"Transkripsi berhasil untuk {video_url}.")
        return transcribed_text

    except subprocess.TimeoutExpired:
        logger.error(f"Proses video dari {video_url} timeout.", exc_info=True)
        return "Maaf, proses transkripsi video melebihi batas waktu. Coba lagi dengan video yang lebih pendek atau koneksi yang lebih stabil."
    # Blok except torchaudio.exceptions.TorchaudioException telah dihapus, akan ditangkap oleh Exception umum
    except Exception as e: 
        logger.error(f"Error selama konversi video ke teks untuk {video_url}: {e}", exc_info=True)
        return f"Maaf, terjadi kesalahan tak terduga saat memproses video: {str(e)}."
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Berhasil membersihkan file audio temporer: {audio_path}")
            except Exception as e:
                logger.error(f"Gagal membersihkan file audio temporer {audio_path}: {e}")