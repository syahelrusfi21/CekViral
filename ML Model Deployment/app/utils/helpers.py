# cekviral_project/app/utils/helpers.py
import re
from urllib.parse import urlparse
import logging # Tambahkan import logging

logger = logging.getLogger(__name__) # Tambahkan logger

def is_url(input_string: str) -> bool:
    """Checks if the given string is a valid URL."""
    if not isinstance(input_string, str): # Tambahkan pengecekan tipe
        return False
    # This regex is more robust for general URLs
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(input_string))

def get_domain(url: str) -> str | None:
    """Extracts the domain from a URL."""
    if not isinstance(url, str): # Tambahkan pengecekan tipe
        return None
    try:
        parsed_url = urlparse(url)
        return parsed_url.netloc
    except Exception as e: # Tambahkan logging error
        logger.error(f"Error parsing URL to get domain: {url} - {e}")
        return None

# --- PENAMBAHAN DIMULAI DI SINI ---

# Daftar host platform video utama
# Menggunakan set untuk pencarian yang lebih cepat
DIRECT_VIDEO_PLATFORM_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "youtu.be",
    "tiktok.com", "www.tiktok.com", "m.tiktok.com",
    "instagram.com", "www.instagram.com", "m.instagram.com", # m.instagram.com untuk mobile
    "vimeo.com", "www.vimeo.com",
    "dailymotion.com", "www.dailymotion.com",
    "facebook.com", "www.facebook.com", "m.facebook.com", # Tambahkan Facebook video
    "fb.watch", # Link pendek Facebook video
    "twitter.com", "x.com" # Tambahkan Twitter/X video
    # Tambahkan platform lain jika perlu, misal: twitch.tv
}

# Pola regex untuk URL video spesifik dari platform tersebut
# Ini membantu membedakan antara halaman video langsung dan halaman profil/lainnya di platform yang sama
DIRECT_VIDEO_URL_PATTERNS = [
    re.compile(r"https://(www\.|m\.)?youtube\.com/(watch\?v=|embed/|shorts/|live/|playlist\?list=)"), # Tambahkan playlist
    re.compile(r"https://youtu\.be/"),
    re.compile(r"https://(www\.|m\.)?tiktok\.com/(@[^/]+/video/|v/|t/)"), # Pola lebih umum untuk TikTok
    re.compile(r"https://(www\.|m\.)?instagram\.com/(p|reel|reels|tv)/[^/]+/?"), # /p/ untuk post, /reel/ atau /reels/, /tv/ untuk IGTV
    re.compile(r"https://(www\.)?vimeo\.com/\d+"), # Video Vimeo biasanya dengan ID numerik
    re.compile(r"https://(www\.)?dailymotion\.com/video/"),
    re.compile(r"https://(www\.|m\.)?facebook\.com/([^/]+/videos/|watch/\?v=|video\.php\?v=)"), # Pola Facebook video
    re.compile(r"https://fb\.watch/"), # Link pendek Facebook video
    re.compile(r"https://(www\.)?(twitter|x)\.com/[^/]+/status/\d+") # Tweet dengan video
]

def is_direct_video_platform_url(url: str) -> bool:
    """
    Mengecek apakah URL adalah link langsung ke video di platform populer.
    """
    if not url or not isinstance(url, str): # Pastikan url adalah string dan tidak kosong
        return False
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        
        if not hostname:
            return False

        # Normalisasi hostname (misal, hapus 'www.' atau 'm.' untuk perbandingan)
        normalized_hostname = hostname.replace("www.", "").replace("m.", "")

        if normalized_hostname in DIRECT_VIDEO_PLATFORM_HOSTS:
            # Jika hostname ada di daftar platform video, cocokkan dengan pola URL video spesifik
            for pattern in DIRECT_VIDEO_URL_PATTERNS:
                if pattern.match(url):
                    logger.debug(f"URL matched direct video pattern: {url} with {pattern.pattern}")
                    return True
            # Jika hostname platform video tapi tidak cocok pola spesifik,
            # bisa jadi bukan direct video link (misalnya halaman channel, profil)
            logger.debug(f"URL from video platform but not a direct video link: {url}")
            return False
        
        logger.debug(f"URL hostname not in DIRECT_VIDEO_PLATFORM_HOSTS: {url}")
        return False
        
    except Exception as e:
        logger.error(f"Error in is_direct_video_platform_url for URL: {url} - {e}", exc_info=True)
        return False
# --- AKHIR PENAMBAHAN ---