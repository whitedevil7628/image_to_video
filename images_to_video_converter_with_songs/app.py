from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

UPLOAD_FOLDER = 'uploads'
VIDEOS_FOLDER = 'videos'
MUSIC_FOLDER = 'music'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'm4a'}

# Create directories
for folder in [UPLOAD_FOLDER, VIDEOS_FOLDER, MUSIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Get form data
        images = request.files.getlist('images')
        audio = request.files.get('audio')
        duration = float(request.form.get('duration', 2.0))
        transition = request.form.get('transition', 'fade')
        effect = request.form.get('effect', 'none')
        music_url = request.form.get('music_url', '')
        
        # Download music from URL if provided
        if music_url and not audio:
            try:
                import urllib.request
                import time
                music_filename = f"downloaded_music_{int(time.time())}.wav"
                audio_path = os.path.join(MUSIC_FOLDER, music_filename)
                urllib.request.urlretrieve(music_url, audio_path)
                
                # Verify the file was downloaded
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    audio_path = None
            except Exception as e:
                print(f"Music download failed: {e}")
                audio_path = None
        
        if not images or len(images) < 2:
            return jsonify({'error': 'Please upload at least 2 images'}), 400
        
        # Save images
        image_paths = []
        for i, image in enumerate(images):
            if image and allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = f"img_{i}_{secure_filename(image.filename)}"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                image.save(image_path)
                
                # Resize image to standard size
                img = Image.open(image_path)
                img = img.resize((1080, 1920), Image.Resampling.LANCZOS)
                img.save(image_path)
                image_paths.append(image_path)
        
        # Save audio if provided
        audio_path = None
        if audio and allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
            audio_filename = secure_filename(audio.filename)
            audio_path = os.path.join(MUSIC_FOLDER, audio_filename)
            audio.save(audio_path)
        
        # Create video
        video_filename = f"reel_{len(os.listdir(VIDEOS_FOLDER)) + 1}.mp4"
        video_path = os.path.join(VIDEOS_FOLDER, video_filename)
        
        success = create_video(image_paths, video_path, duration, transition, effect, audio_path)
        
        # Cleanup uploaded images
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        if success:
            return jsonify({
                'success': True,
                'video_filename': video_filename,
                'redirect_url': url_for('result', filename=video_filename)
            })
        else:
            return jsonify({'error': 'Failed to create video'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Error creating video: {str(e)}'}), 500

def apply_effect(img, effect, frame_num, total_frames):
    """Apply visual effects to image"""
    if effect == 'zoom':
        # Zoom in effect
        progress = frame_num / total_frames
        scale = 1.0 + (progress * 0.1)  # Zoom from 1.0 to 1.1
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize and crop to center
        resized = cv2.resize(img, (new_w, new_h))
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    
    elif effect == 'blur_focus':
        # Blur to focus effect
        progress = frame_num / total_frames
        blur_amount = int(15 * (1 - progress))  # Start blurred, become sharp
        if blur_amount > 0:
            return cv2.GaussianBlur(img, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
    
    elif effect == 'brightness':
        # Brightness fade in
        progress = frame_num / total_frames
        brightness = int(255 * progress)
        return cv2.convertScaleAbs(img, alpha=progress, beta=brightness * 0.1)
    
    return img

def create_video(image_paths, output_path, duration_per_image, transition, effect, audio_path=None):
    try:
        import time
        
        # Video settings
        fps = 30
        width, height = 1080, 1920
        frames_per_image = int(fps * duration_per_image)
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            # Fallback to mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            return False
        
        for i, image_path in enumerate(image_paths):
            # Load and resize image
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.resize(img, (width, height))
            
            # Apply transition and effects
            for frame in range(frames_per_image):
                current_img = img.copy()
                
                # Apply visual effects
                if effect != 'none':
                    current_img = apply_effect(current_img, effect, frame, frames_per_image)
                
                # Apply transitions
                if transition == 'fade':
                    if i == 0 and frame < 15:  # Fade in first image
                        alpha = frame / 15
                        current_img = cv2.convertScaleAbs(current_img, alpha=alpha, beta=0)
                    elif i == len(image_paths) - 1 and frame > frames_per_image - 15:  # Fade out last image
                        alpha = (frames_per_image - frame) / 15
                        current_img = cv2.convertScaleAbs(current_img, alpha=alpha, beta=0)
                
                elif transition == 'slide':
                    if frame < 10:  # Slide in effect
                        shift = int((10 - frame) * width / 10)
                        M = np.float32([[1, 0, shift], [0, 1, 0]])
                        current_img = cv2.warpAffine(current_img, M, (width, height))
                
                video_writer.write(current_img)
        
        video_writer.release()
        
        # Add audio using ffmpeg with proper encoding
        if audio_path and os.path.exists(audio_path):
            # Check if ffmpeg is available
            if shutil.which('ffmpeg'):
                # Use ffmpeg to combine video and audio with proper codecs
                cmd = f'ffmpeg -i "{temp_output}" -i "{audio_path}" -c:v libx264 -c:a aac -strict experimental -b:a 192k -shortest "{output_path}" -y'
                result = os.system(cmd)
                
                # Clean up temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                if result == 0 and os.path.exists(output_path):
                    return True
            else:
                # If no ffmpeg, just rename the video file
                os.rename(temp_output, output_path)
                return True
        else:
            # No audio, just rename the temp file
            os.rename(temp_output, output_path)
        
        return os.path.exists(output_path)
    
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

@app.route('/download/<filename>')
def download_video(filename):
    try:
        video_path = os.path.join(VIDEOS_FOLDER, filename)
        return send_file(video_path, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading video: {str(e)}')
        return redirect(url_for('index'))

@app.route('/result/<filename>')
def result(filename):
    # Check if video has audio by looking for corresponding audio file
    video_path = os.path.join(VIDEOS_FOLDER, filename)
    has_audio = False
    
    # Simple check - if file size is larger, likely has audio
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path)
        has_audio = file_size > 1000000  # If larger than 1MB, likely has audio
    
    return render_template('result.html', 
                         video_filename=filename, 
                         has_audio=has_audio,
                         timestamp=int(time.time()))

@app.route('/preview/<filename>')
def preview_video(filename):
    try:
        video_path = os.path.join(VIDEOS_FOLDER, filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
        
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/search_music')
def search_music():
    query = request.args.get('q', 'trending')
    
    # Using Freesound API for real music (you need to get API key from freesound.org)
    # For demo, using sample tracks - replace with real API
    sample_tracks = [
        {
            'id': 1,
            'name': 'Upbeat Summer Vibes',
            'artist': 'AudioLibrary',
            'duration': 30,
            'url': 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav',
            'preview_url': 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav',
            'genre': 'Pop'
        },
        {
            'id': 2,
            'name': 'Chill Lo-Fi Beat',
            'artist': 'BeatsLibrary',
            'duration': 30,
            'url': 'https://www.soundjay.com/misc/sounds/bell-ringing-04.wav',
            'preview_url': 'https://www.soundjay.com/misc/sounds/bell-ringing-04.wav',
            'genre': 'Lo-Fi'
        },
        {
            'id': 3,
            'name': 'Electronic Dance',
            'artist': 'EDMBeats',
            'duration': 30,
            'url': 'https://www.soundjay.com/misc/sounds/bell-ringing-03.wav',
            'preview_url': 'https://www.soundjay.com/misc/sounds/bell-ringing-03.wav',
            'genre': 'Electronic'
        },
        {
            'id': 4,
            'name': 'Acoustic Guitar Melody',
            'artist': 'GuitarTracks',
            'duration': 30,
            'url': 'https://www.soundjay.com/misc/sounds/bell-ringing-02.wav',
            'preview_url': 'https://www.soundjay.com/misc/sounds/bell-ringing-02.wav',
            'genre': 'Acoustic'
        },
        {
            'id': 5,
            'name': 'Hip Hop Instrumental',
            'artist': 'HipHopBeats',
            'duration': 30,
            'url': 'https://www.soundjay.com/misc/sounds/bell-ringing-01.wav',
            'preview_url': 'https://www.soundjay.com/misc/sounds/bell-ringing-01.wav',
            'genre': 'Hip Hop'
        },
        {
            'id': 6,
            'name': 'Trending Pop Beat',
            'artist': 'PopMusic',
            'duration': 30,
            'url': 'https://actions.google.com/sounds/v1/alarms/beep_short.ogg',
            'preview_url': 'https://actions.google.com/sounds/v1/alarms/beep_short.ogg',
            'genre': 'Pop'
        }
    ]
    
    # Filter based on search query
    if query and query != 'trending':
        filtered_tracks = [track for track in sample_tracks 
                          if query.lower() in track['name'].lower() 
                          or query.lower() in track['genre'].lower()]
    else:
        filtered_tracks = sample_tracks
    
    return jsonify(filtered_tracks)

@app.route('/download_music')
def download_music():
    music_url = request.args.get('url')
    music_id = request.args.get('id')
    
    if not music_url:
        return jsonify({'error': 'No music URL provided'}), 400
    
    try:
        import urllib.request
        import urllib.parse
        
        # Create filename
        filename = f"music_{music_id}_{int(time.time())}.wav"
        file_path = os.path.join(MUSIC_FOLDER, filename)
        
        # Download the music file
        urllib.request.urlretrieve(music_url, file_path)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': file_path
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to download music: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)