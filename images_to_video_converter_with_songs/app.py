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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max for cloud deployment

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
        
        # Handle music from URL if provided
        audio_path = None
        if music_url and not audio:
            try:
                import urllib.request
                import urllib.parse
                
                # Parse URL to get file extension
                parsed_url = urllib.parse.urlparse(music_url)
                file_ext = os.path.splitext(parsed_url.path)[1] or '.mp3'
                
                music_filename = f"downloaded_music_{int(time.time())}{file_ext}"
                audio_path = os.path.join(MUSIC_FOLDER, music_filename)
                
                # Add headers to avoid blocking
                req = urllib.request.Request(
                    music_url,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    with open(audio_path, 'wb') as f:
                        f.write(response.read())
                
                # Verify the file was downloaded and has content
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    audio_path = None
                    print(f"Downloaded file too small or empty")
                    
            except Exception as e:
                print(f"Music download failed: {e}")
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
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
                
                # Resize image to optimized size
                img = Image.open(image_path)
                img = img.resize((720, 1280), Image.Resampling.LANCZOS)
                img.save(image_path)
                image_paths.append(image_path)
        
        # Save uploaded audio if provided (and no URL audio)
        if not audio_path and audio and allowed_file(audio.filename, ALLOWED_AUDIO_EXTENSIONS):
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
    """Apply lightweight visual effects optimized for cloud deployment"""
    try:
        h, w = img.shape[:2]
        progress = frame_num / max(total_frames, 1)
        
        if effect == 'zoom_glow':
            # Simple zoom with glow
            scale = 1.0 + (progress * 0.1)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Lightweight glow
            blurred = cv2.GaussianBlur(result, (9, 9), 0)
            return cv2.addWeighted(result, 0.8, blurred, 0.2, 0)
        
        elif effect == 'pan_zoom_particles':
            # Simple pan zoom
            scale = 1.0 + (progress * 0.15)
            pan_x = int(progress * 30 - 15)
            pan_y = int(progress * 20 - 10)
            
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > w and new_h > h:
                resized = cv2.resize(img, (new_w, new_h))
                start_x = max(0, min((new_w - w) // 2 + pan_x, new_w - w))
                start_y = max(0, min((new_h - h) // 2 + pan_y, new_h - h))
                return resized[start_y:start_y + h, start_x:start_x + w]
            return img
        
        elif effect == 'parallax_glow':
            # Simple parallax
            shift_x = int(np.sin(progress * np.pi * 2) * 15)
            shift_y = int(np.cos(progress * np.pi * 2) * 8)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        elif effect == 'zoom_out_sparkle':
            # Simple zoom out
            scale = max(0.7, 1.2 - (progress * 0.2))
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        elif effect == 'motion_blur_glow':
            # Simple blur
            blur_amount = max(1, int(15 * (1 - progress)))
            if blur_amount > 1:
                kernel_size = min(blur_amount, 9)
                return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            return img
        
        elif effect == 'cinematic_flare':
            # Simple brightness
            alpha = 0.4 + (progress * 0.6)
            return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        elif effect == 'wave_distortion':
            # Simple wave
            result = img.copy()
            wave_strength = 3
            for y in range(0, h, 4):
                offset = int(wave_strength * np.sin(2 * np.pi * y / 40 + frame_num * 0.2))
                if offset != 0:
                    M = np.float32([[1, 0, offset], [0, 1, 0]])
                    result[y:min(y+4, h)] = cv2.warpAffine(result[y:min(y+4, h)], M, (w, min(4, h-y)), borderMode=cv2.BORDER_REFLECT)
            return result
        
        elif effect == 'hologram':
            # Simple hologram
            result = img.copy()
            for y in range(0, h, 6):
                if (y + frame_num) % 12 < 6:
                    result[y:min(y+3, h), :] = result[y:min(y+3, h), :] * 0.8
            return result
        
        # Basic fallback effects
        elif effect == 'zoom':
            scale = 1.0 + (progress * 0.1)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return img
        
    except Exception as e:
        print(f"Effect '{effect}' failed: {e}")
        return img

def create_video(image_paths, output_path, duration_per_image, transition, effect, audio_path=None):
    try:
        import time
        
        # Optimized video settings for cloud deployment
        fps = 24  # Reduced FPS
        width, height = 720, 1280  # Reduced resolution
        frames_per_image = max(12, int(fps * duration_per_image))  # Minimum frames
        
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
                
                # Apply visual effects with subtle motion
                if effect != 'none':
                    try:
                        current_img = apply_effect(current_img, effect, frame, frames_per_image)
                        
                        # Add subtle camera shake for more realism
                        if effect in ['parallax_glow', 'zoom_glow', 'pan_zoom_particles'] and frame % 3 == 0:
                            shake_x = int(np.random.normal(0, 0.5))
                            shake_y = int(np.random.normal(0, 0.3))
                            if abs(shake_x) > 0 or abs(shake_y) > 0:
                                M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                                current_img = cv2.warpAffine(current_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
                    except Exception as e:
                        print(f"Effect error: {e}")
                        current_img = img.copy()
                
                # Apply enhanced transitions
                if transition == 'fade':
                    if i == 0 and frame < 20:  # Fade in first image
                        alpha = (frame / 20) ** 0.5  # Ease-in curve
                        current_img = cv2.convertScaleAbs(current_img, alpha=alpha, beta=0)
                    elif i == len(image_paths) - 1 and frame > frames_per_image - 20:  # Fade out last image
                        alpha = ((frames_per_image - frame) / 20) ** 0.5  # Ease-out curve
                        current_img = cv2.convertScaleAbs(current_img, alpha=alpha, beta=0)
                
                elif transition == 'slide':
                    if frame < 15:  # Enhanced slide in effect
                        progress = frame / 15
                        # Ease-out animation curve
                        eased_progress = 1 - (1 - progress) ** 3
                        shift = int((1 - eased_progress) * width)
                        M = np.float32([[1, 0, shift], [0, 1, 0]])
                        current_img = cv2.warpAffine(current_img, M, (width, height), borderMode=cv2.BORDER_REFLECT)
                
                elif transition == 'crossfade':
                    # Cross-fade between images
                    if i > 0 and frame < 20:
                        # Load previous image for crossfade
                        prev_img = cv2.imread(image_paths[i-1])
                        if prev_img is not None:
                            prev_img = cv2.resize(prev_img, (width, height))
                            alpha = frame / 20
                            current_img = cv2.addWeighted(prev_img, 1-alpha, current_img, alpha, 0)
                
                video_writer.write(current_img)
        
        video_writer.release()
        
        # Add audio using ffmpeg with enhanced encoding
        if audio_path and os.path.exists(audio_path):
            # Check if ffmpeg is available
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path:
                try:
                    # Calculate video duration for audio looping
                    total_duration = len(image_paths) * duration_per_image
                    
                    # Enhanced ffmpeg command with better audio handling
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_output,
                        '-stream_loop', '-1',  # Loop audio if needed
                        '-i', audio_path,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '128k',
                        '-ar', '44100',
                        '-ac', '2',
                        '-t', str(total_duration),  # Match video duration
                        '-shortest',
                        '-movflags', '+faststart',  # Web optimization
                        output_path
                    ]
                    
                    import subprocess
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Clean up temp file
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    
                    # Clean up downloaded audio if it was from URL
                    if 'downloaded_music_' in audio_path and os.path.exists(audio_path):
                        os.remove(audio_path)
                        
                    if result.returncode == 0 and os.path.exists(output_path):
                        return True
                    else:
                        print(f"FFmpeg error: {result.stderr}")
                        # Fallback: just use video without audio
                        if os.path.exists(temp_output):
                            os.rename(temp_output, output_path)
                        return os.path.exists(output_path)
                        
                except Exception as e:
                    print(f"FFmpeg processing error: {e}")
                    # Fallback: just use video without audio
                    if os.path.exists(temp_output):
                        os.rename(temp_output, output_path)
                    return os.path.exists(output_path)
            else:
                print("FFmpeg not found, creating video without audio")
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
    
    # Using royalty-free music sources
    sample_tracks = [
        {
            'id': 1,
            'name': 'Upbeat Summer Vibes',
            'artist': 'AudioLibrary',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-ukulele.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-ukulele.mp3',
            'genre': 'Pop'
        },
        {
            'id': 2,
            'name': 'Chill Lo-Fi Beat',
            'artist': 'BeatsLibrary',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-relaxing.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-relaxing.mp3',
            'genre': 'Lo-Fi'
        },
        {
            'id': 3,
            'name': 'Electronic Dance',
            'artist': 'EDMBeats',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-energy.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-energy.mp3',
            'genre': 'Electronic'
        },
        {
            'id': 4,
            'name': 'Acoustic Guitar Melody',
            'artist': 'GuitarTracks',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-acoustic.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-acoustic.mp3',
            'genre': 'Acoustic'
        },
        {
            'id': 5,
            'name': 'Hip Hop Instrumental',
            'artist': 'HipHopBeats',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-hip-hop.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-hip-hop.mp3',
            'genre': 'Hip Hop'
        },
        {
            'id': 6,
            'name': 'Trending Pop Beat',
            'artist': 'PopMusic',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-sunny.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-sunny.mp3',
            'genre': 'Pop'
        },
        {
            'id': 7,
            'name': 'Cinematic Epic',
            'artist': 'CinemaBeats',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-epic.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-epic.mp3',
            'genre': 'Cinematic'
        },
        {
            'id': 8,
            'name': 'Funky Groove',
            'artist': 'FunkMaster',
            'duration': 30,
            'url': 'https://www.bensound.com/bensound-music/bensound-funky.mp3',
            'preview_url': 'https://www.bensound.com/bensound-music/bensound-funky.mp3',
            'genre': 'Funk'
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

# if __name__ == '__main__':
#     app.run(debug=True)