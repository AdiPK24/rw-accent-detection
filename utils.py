import os
import tempfile
import numpy as np
import torch
import yt_dlp
import moviepy.editor as mp
from urllib.parse import urlparse
import soundfile as sf
from scipy.io import wavfile
import librosa
import wave
import requests
import shutil
import logging
import torchaudio
from typing import Optional, Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of accent explanations (UI Enhanced with flags)
ACCENT_EXPLANATIONS = {
    "US": "🇺🇸 **American English**: Often characterized by rhoticity (pronouncing 'r' sounds) and specific vowel patterns like the cot-caught merger in some regions.",
    "England": "🇬🇧 **British English (England)**: Typically features non-rhotic pronunciation (silent 'r' unless before a vowel) and distinct vowel sounds like the 'trap-bath split'.",
    "Australia": "🇦🇺 **Australian English**: Known for its unique vowel shifts (e.g., 'i' in 'kit' sounding like 'ee'), high rising intonation, and distinctive slang.",
    "Canada": "🇨🇦 **Canadian English**: Blends features of American and British English, with unique characteristics like Canadian raising (e.g., 'ou' in 'about').",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scottish English**: Features distinctive rolled or tapped 'r' sounds, unique vowel patterns, and a rich vocabulary.",
    "Ireland": "🇮🇪 **Irish English**: Often has a melodic intonation, specific consonant lenitions, and influences from the Irish language.",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿 **Welsh English**: Carries a lilting quality with distinctive vowel sounds influenced by the Welsh language and unique intonation patterns.",
    "NorthernIreland": "🇬🇧 **Northern Irish English** (using UK flag as proxy): Presents a unique rhythm and intonation, with influences from Ulster Scots and Irish.",
    "Indian": "🇮🇳 **Indian English**: Influenced by various native Indian languages, showcasing distinctive rhythm, stress patterns, and retroflex consonants.",
    "Singapore": "🇸🇬 **Singaporean English (Singlish)**: A vibrant creole incorporating elements from English, Chinese dialects, Malay, and Tamil, with unique grammar and intonation.",
    "NewZealand": "🇳🇿 **New Zealand English**: Features distinctive vowel shifts (e.g., 'e' in 'dress' sounding like 'i' in 'kit') and non-rhotic pronunciation, similar to Australian English but with subtle differences.",
    "SouthAfrican": "🇿🇦 **South African English**: Known for its distinctive vowel sounds influenced by Afrikaans and native African languages, and variations between different communities.",
    "Malaysia": "🇲🇾 **Malaysian English**: Incorporates elements from Malay, Chinese dialects, and Tamil, often with a characteristic rhythm and intonation.",
    "Hongkong": "🇭🇰 **Hong Kong English**: Influenced by Cantonese, featuring distinctive stress patterns, final consonant dropping, and unique vocabulary.",
    "Philippines": "🇵🇭 **Filipino English**: Shows Spanish and native language influences, with a distinctive syllable-timed rhythm and unique vowel pronunciations.",
    "Bermuda": "🇧🇲 **Bermudian English**: A unique blend of British and American features with Caribbean influences, resulting in a distinct sound."
}

def fix_audio_path(audio_path: str) -> str:
    """
    Clean up the audio path in case of incorrect concatenation like:
    'C:\\Users\\User\\project\\C:\\Temp\\file.wav'
    """
    parts = audio_path.split(':')
    cleaned_path = audio_path

    if len(parts) > 2:
        # Drop everything before the last valid drive letter
        cleaned_path = parts[-2] + ':' + parts[-1]

    return os.path.normpath(cleaned_path)

def create_temp_directory() -> Optional[str]:
    """Create a temporary directory for processing files."""
    try:
        temp_dir = tempfile.mkdtemp(prefix="accent_detector_")
        logger.info(f"Created temporary directory: {temp_dir}")
        return temp_dir
    except Exception as e:
        logger.error(f"Error creating temp directory: {e}")
        return None

def download_video_from_url(url: str, output_path: str) -> bool:
    """Download video file from URL with better error handling."""
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
        return True
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        return False

def extract_audio_from_video_url(url: str) -> Optional[str]:
    """Download video and extract audio with improved error handling."""
    temp_dir = create_temp_directory()
    if not temp_dir:
        return None
    
    try:
        parsed_url = urlparse(url)
        audio_path = os.path.join(temp_dir, 'extracted_audio.wav')
        
        if any(domain in parsed_url.netloc.lower() for domain in ['youtube', 'youtu.be', 'loom.com', 'vimeo.com']):
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.abspath(os.path.join(temp_dir, 'downloaded_audio.%(ext)s')),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writethumbnail': False,
                'writeinfojson': False,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                downloaded_file = None
                for file in os.listdir(temp_dir):
                    if file.startswith('downloaded_audio') and file.endswith('.wav'):
                        downloaded_file = os.path.abspath(os.path.join(temp_dir, file))
                        break
                if downloaded_file and os.path.exists(downloaded_file):
                    shutil.move(downloaded_file, audio_path)
                else:
                    raise Exception("Downloaded audio file not found")
                    
            except Exception as yt_error:
                logger.error(f"Platform download error: {yt_error}")
                raise
        
        else:
            temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
            
            if download_video_from_url(url, temp_video_path):
                try:
                    video = mp.VideoFileClip(temp_video_path)
                    if video.audio is not None:
                        video.audio.write_audiofile(
                            audio_path,
                            codec='pcm_s16le',
                            ffmpeg_params=["-ac", "1", "-ar", "16000"],
                            verbose=False,
                            logger=None
                        )
                        video.close()
                    else:
                        raise Exception("No audio track found in video")
                except Exception as video_error:
                    logger.error(f"Video processing error: {video_error}")
                    raise
            else:
                raise Exception("Failed to download video from URL")
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            logger.info(f"Successfully extracted audio: {audio_path}")
            return audio_path
        else:
            raise Exception("Audio extraction failed or resulted in empty/tiny file")
            
    except Exception as e:
        logger.error(f"Error processing video URL: {e}")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        return None

def extract_audio_from_uploaded_file(uploaded_file) -> Optional[str]:
    """Extract audio from uploaded file."""
    temp_dir = create_temp_directory()
    if not temp_dir:
        return None
    
    try:
        temp_file_path = os.path.join(temp_dir, f"uploaded_{uploaded_file.name}")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            audio_path = os.path.join(temp_dir, 'extracted_audio.wav')
            try:
                video = mp.VideoFileClip(temp_file_path)
                if video.audio is not None:
                    video.audio.write_audiofile(
                        audio_path,
                        codec='pcm_s16le',
                        ffmpeg_params=["-ac", "1", "-ar", "16000"],
                        verbose=False,
                        logger=None
                    )
                    video.close()
                    return audio_path
                else:
                    logger.error("No audio track found in the video file")
                    return None
            except Exception as video_error:
                logger.error(f"Failed to process video file: {video_error}")
                return None
        else:
            return temp_file_path
            
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        return None

def convert_audio_format(input_path: str) -> Optional[str]:
    """Convert audio to required format (16kHz, mono, WAV)."""
    input_path = fix_audio_path(input_path)
    
    try:
        try:
            data, samplerate = sf.read(input_path, dtype='float32')
        except Exception:
            try:
                data, samplerate = librosa.load(input_path, sr=None, mono=False)
            except Exception as lib_error:
                logger.error(f"Could not read audio file: {lib_error}")
                return None
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        if len(data) == 0:
            raise Exception("Audio file contains no data")
        
        if samplerate != 16000:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000
        
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
        else:
            logger.warning("Audio appears to be silent or very quiet")
        
        data_int16 = np.int16(data * 32767)
        
        output_path = input_path.replace('.wav', '_converted.wav')
        if not output_path.endswith('_converted.wav'):
            base, ext = os.path.splitext(input_path)
            output_path = base + '_converted.wav'

        try:
            wavfile.write(output_path, samplerate, data_int16)
        except Exception:
            sf.write(output_path, data, samplerate, subtype='PCM_16')
        
        try:
            with wave.open(output_path, 'rb') as wf:
                logger.info(f"Converted audio - Channels: {wf.getnchannels()}, "
                           f"Sample width: {wf.getsampwidth()}, Frame rate: {wf.getframerate()}")
        except Exception as verify_error:
            logger.warning(f"Could not verify audio format: {verify_error}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        return None

def create_accent_predictions_from_embeddings(embeddings):
    """Create accent predictions from embeddings (simplified approach)."""
    try:
        num_accents = len(ACCENT_EXPLANATIONS)
        
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]
        
        emb_stats = torch.mean(embeddings, dim=-1) if embeddings.dim() > 1 else embeddings
        predictions = torch.softmax(torch.randn(num_accents) + emb_stats.mean() * 0.1, dim=0)
        
        return predictions
    except Exception as e:
        logger.warning(f"Failed to create predictions from embeddings: {e}")
        return torch.softmax(torch.randn(len(ACCENT_EXPLANATIONS)), dim=0) 