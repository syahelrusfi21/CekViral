# cekviral_project/app/api/endpoints.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import requests
import asyncio
import logging

# Impor fungsi-fungsi dari modul lain
from app.utils.helpers import is_url, is_direct_video_platform_url
from app.services.content_analyzer import extract_text_from_html, convert_video_to_text
from app.services.ml_model import predict_content_hoax_status

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Definisi Pydantic Models ---
class ContentInput(BaseModel):
    content: str = Field(..., description="Konten yang akan diverifikasi, bisa berupa teks murni atau URL.")

class PredictionProbabilities(BaseModel):
    HOAX: float = Field(..., description="Probabilitas konten sebagai HOAX.")
    FAKTA: float = Field(..., description="Probabilitas konten sebagai FAKTA.")

class MLPredictionOutput(BaseModel):
    status: str = Field(..., description="Status hasil prediksi ML (e.g., 'success', 'error').")
    message: str = Field(..., description="Pesan terkait hasil prediksi ML.")
    probabilities: PredictionProbabilities = Field(..., description="Probabilitas untuk setiap kelas (HOAX/FAKTA).")
    predicted_label_model: str = Field(..., description="Label prediksi asli dari model (sebelum penerapan threshold).")
    highest_confidence: float = Field(..., description="Nilai kepercayaan tertinggi dari prediksi model.")
    final_label_thresholded: str = Field(..., description="Label final setelah penerapan threshold (HOAX, FAKTA, atau BELUM DIVERIFIKASI).")
    inference_time_ms: float = Field(..., description="Waktu inferensi model dalam milidetik.")

class VerificationResult(BaseModel):
    original_input: str = Field(..., description="Input asli yang diberikan oleh pengguna.")
    input_type: str = Field(..., description="Tipe input (e.g., 'text', 'url').")
    processed_text: str | None = Field(None, description="Teks yang telah diekstrak atau ditranskripsi untuk analisis, jika ada.")
    prediction: MLPredictionOutput = Field(..., description="Detail hasil prediksi dari model ML.")
    processing_message: str = Field(..., description="Pesan umum terkait status pemrosesan konten.")

# --- Router Endpoint ---

@router.post("/verify", response_model=VerificationResult)
async def verify_content(input_data: ContentInput):
    user_input = input_data.content.strip() # Tambahkan strip() untuk membersihkan spasi
    processed_text: str | None = None
    input_type = "text"
    # Pesan default, akan diupdate sesuai alur
    processing_message = "Konten sedang diproses untuk verifikasi." 

    default_ml_output = MLPredictionOutput(
        status="error",
        message="Tidak ada teks yang dapat diproses atau diverifikasi oleh model ML.",
        probabilities=PredictionProbabilities(HOAX=0.0, FAKTA=0.0), # Gunakan model Pydantic
        predicted_label_model="N/A",
        highest_confidence=0.0,
        final_label_thresholded="BELUM DIVERIFIKASI",
        inference_time_ms=0.0
    )
    prediction_details = default_ml_output

    if is_url(user_input):
        input_type = "url"
        logger.info(f"Processing URL: {user_input}")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
            "Referer": "https://www.google.com/"
        }
        
        html_content: str | None = None
        try:
            response = await asyncio.to_thread(requests.get, user_input, headers=headers, timeout=20) # Naikkan timeout sedikit
            response.raise_for_status()
            html_content = response.text
        except requests.exceptions.Timeout:
            logger.error(f"Timeout saat mengakses URL {user_input}", exc_info=True)
            processing_message = f"Gagal mengakses URL: Timeout. Server tujuan mungkin lambat merespons atau tidak dapat dijangkau."
            # Tidak raise HTTPException agar bisa return struktur VerificationResult yang konsisten
        except requests.exceptions.RequestException as e:
            logger.error(f"Gagal mengakses URL {user_input}: {e}", exc_info=True)
            processing_message = f"Gagal mengakses URL: {e}. Pastikan URL valid dan dapat diakses."
            # Tidak raise HTTPException

        if html_content:
            if is_direct_video_platform_url(user_input):
                logger.info(f"URL {user_input} terdeteksi sebagai link video platform. Memulai transkripsi.")
                try:
                    temp_processed_text = await convert_video_to_text(user_input)
                    
                    # Daftar frasa error yang dikenal dari convert_video_to_text
                    known_asr_error_phrases = [
                        "Maaf, fitur transkripsi suara tidak tersedia", 
                        "Maaf, video ini adalah video pribadi",
                        "Maaf, tidak ada format audio yang dapat diunduh",
                        "Maaf, video ini tidak tersedia atau pengunduhan diblokir",
                        "Maaf, URL video tidak valid atau tidak didukung",
                        "Maaf, pengunduhan video ini diblokir karena masalah hak cipta",
                        "Maaf, gagal mengunduh audio dari video",
                        "Maaf, audio video tidak dapat diunduh atau file audio kosong",
                        "Maaf, video terlalu pendek atau tidak berisi suara yang jelas",
                        "Maaf, tidak ada obrolan atau suara yang jelas terdeteksi",
                        "Maaf, proses transkripsi video melebihi batas waktu",
                        "Maaf, terjadi kesalahan tak terduga saat transkripsi video",
                        "Maaf, terjadi masalah saat memeriksa alat bantu transkripsi suara",
                        "Maaf, terjadi masalah saat membaca file audio yang diunduh"
                    ]
                    is_asr_error = False
                    if temp_processed_text:
                        for phrase in known_asr_error_phrases:
                            if phrase in temp_processed_text: # Cukup periksa keberadaan frasa
                                is_asr_error = True
                                break
                    
                    if is_asr_error:
                        processing_message = temp_processed_text 
                        processed_text = None 
                        logger.warning(f"ASR gagal untuk URL {user_input}: {processing_message}")
                    elif not temp_processed_text or not temp_processed_text.strip():
                        processing_message = "Gagal mentranskripsi audio dari video. Konten mungkin tidak memiliki audio yang jelas."
                        processed_text = None 
                        logger.warning(processing_message + f" untuk URL: {user_input}")
                    else:
                        processed_text = temp_processed_text
                        processing_message = "Transkripsi video berhasil."
                        logger.info(f"Transkripsi video berhasil ({len(processed_text)} karakter) dari {user_input}.")
                
                except Exception as e: 
                    processing_message = f"Terjadi kesalahan sistem saat memproses video: {str(e)}."
                    processed_text = None
                    logger.error(f"Error sistem saat memproses video {user_input}: {e}", exc_info=True)

            else: # Halaman web umum
                logger.info(f"URL {user_input} terdeteksi sebagai halaman web. Mengekstrak teks artikel.")
                processed_text_from_html = await asyncio.to_thread(extract_text_from_html, html_content)
                if not processed_text_from_html or not processed_text_from_html.strip():
                    processing_message = "Gagal mengekstrak teks dari URL halaman web. Konten mungkin tidak memiliki cukup teks."
                    processed_text = None
                    logger.warning(processing_message + f" untuk URL: {user_input}")
                else:
                    processed_text = processed_text_from_html
                    processing_message = "Teks dari halaman web berhasil diekstrak."
                    logger.info(f"Teks dari web berhasil diekstrak ({len(processed_text)} karakter) dari {user_input}.")
        
        elif not processing_message.startswith("Gagal mengakses URL"): 
            processing_message = f"Gagal mendapatkan konten HTML dari URL: {user_input}."
            logger.error(processing_message)
            # processed_text sudah None by default

    else: # Input adalah teks murni
        logger.info(f"Input terdeteksi sebagai teks murni.")
        if not user_input.strip(): # Tambahan: cek jika teks murni juga kosong
            processing_message = "Input teks kosong, tidak ada yang diverifikasi."
            processed_text = None
            logger.warning(processing_message)
        else:
            processed_text = user_input
            processing_message = "Teks murni diterima untuk verifikasi."


    # Setelah pemrosesan, jika ada teks yang valid, kirim ke model ML
    if processed_text and processed_text.strip():
        logger.info(f"Mengirim teks ke model ML (awal): {processed_text[:100]}...")
        # Panggil fungsi ML secara asynchronous
        ml_output = await asyncio.to_thread(predict_content_hoax_status, processed_text)
        
        if ml_output.get("status") == "error":
            prediction_details = default_ml_output 
            # Update pesan hanya jika belum ada pesan error dari tahap sebelumnya yang lebih spesifik
            if "berhasil" in processing_message or processing_message.startswith("Konten sedang diproses") or processing_message.startswith("Teks murni diterima"):
                processing_message = f"Verifikasi ML gagal: {ml_output.get('message', 'Error model tidak diketahui')}"
            logger.error(f"Error model ML: {ml_output.get('message', 'Error model tidak diketahui')}")
        else:
            prediction_details = MLPredictionOutput(**ml_output)
            # Update pesan jika sebelumnya hanya pesan umum
            if processing_message.startswith("Konten sedang diproses") or processing_message.startswith("Teks murni diterima") or processing_message.endswith("berhasil diekstrak.") or processing_message.endswith("video berhasil."):
                 processing_message = "Verifikasi konten oleh model ML selesai."
    else:
        logger.warning("Tidak ada teks signifikan untuk dikirim ke model ML. Menggunakan default status 'BELUM DIVERIFIKASI'.")
        # Update pesan jika belum ada pesan error spesifik dari tahap ekstraksi/transkripsi
        if "berhasil" in processing_message or processing_message.startswith("Konten sedang diproses") or processing_message.startswith("Teks murni diterima"):
            if input_type == "url" and html_content is None and not processing_message.startswith("Gagal mengakses URL"):
                 processing_message = f"Gagal mendapatkan konten dari URL {user_input} untuk verifikasi."
            elif not (processing_message.startswith("Gagal") or processing_message.startswith("Maaf")):
                 processing_message = "Tidak ada teks yang dapat diekstrak atau diproses dari input untuk verifikasi ML."
        # prediction_details tetap default_ml_output

    return VerificationResult(
        original_input=user_input,
        input_type=input_type,
        processed_text=processed_text if processed_text else "", # Kirim string kosong jika None untuk Pydantic
        prediction=prediction_details,
        processing_message=processing_message
    )