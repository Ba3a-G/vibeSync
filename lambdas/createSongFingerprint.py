import os
import json
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import yt_dlp

def lambda_handler(event, context):
    """
    AWS Lambda function that downloads the first 30 seconds of a YouTube video,
    converts it to 16-bit mono 16kHz PCM format, and saves it to the temp directory.
    
    Parameters:
    event (dict): Should contain a 'youtube_url' key with the YouTube video URL
    context (LambdaContext): Lambda context object
    
    Returns:
    dict: Response with status and file path information
    """
    try:
        if 'youtube_url' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing youtube_url parameter'})
            }
        
        youtube_url = event['youtube_url']
        
        if 'tmp_dir' not in event:
            temp_dir = tempfile.gettempdir()
        else:
            temp_dir = event['tmp_dir']
        print(f"Using temp directory: {temp_dir}")
        
        video_id = youtube_url.split('v=')[-1].split('&')[0]
        
        ydl_opts = {
            'quiet': False,
            'no_warnings': False,
            'format': 'worstaudio/worst',
            'skip_download': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_id = info.get('id', video_id)
        
        # Now download using ffmpeg with the time limit
        temp_audio_file = os.path.join(temp_dir, f"{video_id}_original.wav")
        
        ydl_opts = {
            'format': 'worstaudio/worst',
            'outtmpl': os.path.join(temp_dir, f"{video_id}_original.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': {
                'ffmpeg': ['-t', '30']
            },
            'prefer_ffmpeg': True,
            'keepvideo': False
        }
        
        print(f"Downloading audio from {youtube_url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        downloaded_file = None
        for file in os.listdir(temp_dir):
            if file.startswith(f"{video_id}_original") and file.endswith('.wav'):
                downloaded_file = os.path.join(temp_dir, file)
                break
        
        if not downloaded_file:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Could not find downloaded file'})
            }
        
        print(f"Downloaded file: {downloaded_file}")
        
        print("Loading with librosa and downsampling...")
        y, _ = librosa.load(downloaded_file, sr=16000, mono=True)
        
        output_file = os.path.join(temp_dir, f"{video_id}_fingerprint.wav")
        sf.write(output_file, y, 16000, subtype='PCM_16')
        
        if os.path.exists(downloaded_file):
            os.remove(downloaded_file)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Audio processed successfully',
                'video_id': video_id,
                'output_file': output_file,
                'duration': len(y) / 16000
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

if __name__ == '__main__':
    event = {'youtube_url': 'https://www.youtube.com/watch?v=0RDI9CMilhk', 'tmp_dir': './tmp'}
    context = None
    response = lambda_handler(event, context)
    print(response)