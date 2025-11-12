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
                
                # Resize image to standard size
                img = Image.open(image_path)
                img = img.resize((1080, 1920), Image.Resampling.LANCZOS)
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
    """Apply visual effects to image"""
    try:
        h, w = img.shape[:2]
        progress = frame_num / max(total_frames, 1)
        
        if effect == 'zoom_glow':
            # Zoom with pulsing glow
            scale = 1.0 + (progress * 0.15) + np.sin(frame_num * 0.3) * 0.02
            angle = progress * 3 + np.sin(frame_num * 0.2) * 2
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, scale)
            result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Add glow effect
            blurred = cv2.GaussianBlur(result, (15, 15), 0)
            glow_intensity = 0.2 + 0.3 * abs(np.sin(frame_num * 0.4))
            result = cv2.addWeighted(result, 1.0, blurred, glow_intensity, 0)
            
            # Add particles
            overlay = result.copy()
            for i in range(15):
                seed = (frame_num + i * 17) % 1000
                np.random.seed(seed)
                x = int((np.sin(seed * 0.01 + frame_num * 0.1) * 0.5 + 0.5) * w)
                y = int((np.cos(seed * 0.013 + frame_num * 0.08) * 0.5 + 0.5) * h)
                x = max(5, min(x, w-5))
                y = max(5, min(y, h-5))
                size = max(1, int(3 + np.sin(frame_num * 0.2 + i) * 2))
                cv2.circle(overlay, (x, y), size, (255, 255, 255), -1)
            
            return cv2.addWeighted(result, 0.85, overlay, 0.15, 0)
        
        elif effect == 'pan_zoom_particles':
            # Ken Burns with particles
            scale = 1.0 + (progress * 0.25)
            pan_x = int(progress * 60 - 30 + np.sin(frame_num * 0.1) * 10)
            pan_y = int(progress * 40 - 20 + np.cos(frame_num * 0.12) * 8)
            
            new_w, new_h = max(w, int(w * scale)), max(h, int(h * scale))
            if new_w > w or new_h > h:
                resized = cv2.resize(img, (new_w, new_h))
                start_x = max(0, min((new_w - w) // 2 + pan_x, new_w - w))
                start_y = max(0, min((new_h - h) // 2 + pan_y, new_h - h))
                result = resized[start_y:start_y + h, start_x:start_x + w]
            else:
                result = img.copy()
            
            # Add light rays
            overlay = np.zeros_like(result)
            for i in range(3):
                angle = frame_num * 2 + i * 60
                start_x = max(0, min(int(w * 0.1 + np.sin(np.radians(angle)) * w * 0.3), w-1))
                start_y = max(0, min(int(h * 0.1 + np.cos(np.radians(angle)) * h * 0.3), h-1))
                end_x = max(0, min(int(w * 0.9 - np.sin(np.radians(angle)) * w * 0.2), w-1))
                end_y = max(0, min(int(h * 0.9 - np.cos(np.radians(angle)) * h * 0.2), h-1))
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), (50, 50, 100), 2)
            
            overlay = cv2.GaussianBlur(overlay, (21, 21), 0)
            return cv2.addWeighted(result, 0.9, overlay, 0.1, 0)
        
        elif effect == 'parallax_glow':
            # Parallax with edge glow
            shift_x = int(np.sin(progress * np.pi * 3 + frame_num * 0.1) * 25)
            shift_y = int(np.cos(progress * np.pi * 2 + frame_num * 0.08) * 15)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Add edge glow
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges = cv2.dilate(edges, np.ones((3,3)), iterations=1)
            edges = cv2.GaussianBlur(edges, (7, 7), 0)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            return cv2.addWeighted(result, 0.85, edges_colored, 0.15, 0)
        
        elif effect == 'zoom_out_sparkle':
            # Zoom out with sparkles
            scale = max(0.5, 1.3 - (progress * 0.3))
            rotation = np.sin(frame_num * 0.1) * 3
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, rotation, scale)
            result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Add sparkles
            overlay = np.zeros_like(result)
            for i in range(20):
                seed = (frame_num + i * 23) % 500
                np.random.seed(seed)
                x = np.random.randint(5, w-5)
                y = np.random.randint(5, h-5)
                if np.random.random() > 0.7:
                    sparkle_size = np.random.randint(1, 4)
                    intensity = np.random.randint(150, 255)
                    cv2.circle(overlay, (x, y), sparkle_size, (intensity, intensity, intensity), -1)
            
            overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
            return cv2.addWeighted(result, 0.9, overlay, 0.1, 0)
        
        elif effect == 'motion_blur_glow':
            # Motion blur with glow
            blur_amount = max(1, int(25 * (1 - progress)))
            if blur_amount > 1:
                kernel_size = min(blur_amount, 15)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1.0 / kernel_size
                result = cv2.filter2D(img, -1, kernel)
            else:
                result = img.copy()
            
            # Add glow
            glow_intensity = max(0, 0.4 * (1 - progress))
            if glow_intensity > 0:
                blurred = cv2.GaussianBlur(result, (15, 15), 0)
                result = cv2.addWeighted(result, 1.0, blurred, glow_intensity, 0)
            
            return result
        
        elif effect == 'cinematic_flare':
            # Cinematic with lens flares
            alpha = 0.3 + (progress * 0.7)
            beta = max(0, int(30 * (1 - progress)))
            result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Add lens flares
            overlay = np.zeros_like(result)
            flare_x = max(30, min(int(w * (0.2 + progress * 0.6)), w-30))
            flare_y = max(30, min(int(h * (0.3 + np.sin(frame_num * 0.1) * 0.2)), h-30))
            
            cv2.circle(overlay, (flare_x, flare_y), 30, (100, 150, 255), -1)
            cv2.circle(overlay, (flare_x, flare_y), 50, (50, 100, 200), 2)
            
            overlay = cv2.GaussianBlur(overlay, (21, 21), 0)
            return cv2.addWeighted(result, 0.8, overlay, 0.2, 0)
        
        elif effect == 'wave_distortion':
            # Wave distortion
            wave_strength = 5 + 3 * abs(np.sin(frame_num * 0.2))
            result = img.copy()
            
            for y in range(0, h, 2):
                offset = int(wave_strength * np.sin(2 * np.pi * y / 50 + frame_num * 0.3))
                if offset != 0:
                    M = np.float32([[1, 0, offset], [0, 1, 0]])
                    result[y:min(y+2, h)] = cv2.warpAffine(result[y:min(y+2, h)], M, (w, min(2, h-y)), borderMode=cv2.BORDER_REFLECT)
            
            return result
        
        elif effect == 'hologram':
            # Hologram effect
            result = img.copy()
            
            # Add scan lines
            for y in range(0, h, 4):
                if (y + frame_num) % 8 < 4:
                    result[y:min(y+2, h), :] = result[y:min(y+2, h), :] * 0.7
            
            # Add hologram tint
            tint = np.zeros_like(result)
            tint[:,:,1] = 50  # Green tint
            tint[:,:,0] = 30  # Blue tint
            
            return cv2.addWeighted(result, 0.8, tint, 0.2, 0)
        
        # Fallback effects
        elif effect == 'zoom':
            scale = 1.0 + (progress * 0.15)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 0, scale)
            return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        elif effect == 'pan_zoom':
            scale = 1.0 + (progress * 0.2)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > w and new_h > h:
                resized = cv2.resize(img, (new_w, new_h))
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                return resized[start_y:start_y + h, start_x:start_x + w]
        
        return img
        
    except Exception as e:
        print(f"Effect '{effect}' failed: {e}")
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

if __name__ == '__main__':
    app.run(debug=True)