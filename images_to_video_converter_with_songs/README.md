# 📱 Instagram Reel Maker

Create stunning Instagram-style reels from your images with background music! This web application converts multiple images into a vertical video (9:16 aspect ratio) perfect for Instagram Reels, TikTok, and YouTube Shorts.

## ✨ Features

### 🎬 Video Creation
- **Multiple Image Upload**: Drag & drop or browse multiple images
- **Instagram Format**: Automatic 1080x1920 (9:16) aspect ratio
- **Smooth Transitions**: Fade effects between images
- **Custom Duration**: Set how long each image appears (0.5-10 seconds)

### 🎵 Audio Integration
- **Background Music**: Add MP3, WAV, AAC, or M4A files
- **Auto-Loop**: Music automatically loops to match video length
- **Audio Sync**: Perfect synchronization with image transitions

### 🎨 User Experience
- **Drag & Drop Interface**: Easy file uploading
- **Real-time Preview**: See your images before creating video
- **Mobile Responsive**: Works on all devices
- **Instagram-like UI**: Familiar social media design

### 📱 Output Features
- **High Quality**: 1080p resolution at 30fps
- **Optimized Format**: MP4 with H.264 codec
- **Ready to Share**: Perfect for Instagram, TikTok, YouTube Shorts
- **Instant Download**: Get your video immediately

## 🚀 Installation

1. **Install Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   ```
   http://localhost:5000
   ```

## 📖 How to Use

### Step 1: Upload Images
- Drag and drop 2 or more images
- Supports: JPG, PNG, GIF, BMP
- Images are automatically resized to Instagram format

### Step 2: Add Music (Optional)
- Upload your favorite song
- Supports: MP3, WAV, AAC, M4A
- Music will loop to match video duration

### Step 3: Customize Settings
- **Duration**: How long each image shows (0.5-10 seconds)
- **Transition**: Choose fade effect or no transition

### Step 4: Create & Download
- Click "Create Reel" and wait for processing
- Preview your video in the browser
- Download the MP4 file to your device

## 🎯 Perfect For

- **Instagram Reels**: Vertical format optimized for Instagram
- **TikTok Videos**: Perfect aspect ratio and quality
- **YouTube Shorts**: High-quality vertical videos
- **Social Media**: Any platform supporting vertical video
- **Personal Projects**: Slideshows, memories, presentations

## 📁 Project Structure

```
images_to_video_converter_with_songs/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   ├── index.html     # Upload interface
│   └── result.html    # Video preview & download
├── uploads/           # Temporary image storage
├── videos/            # Generated video files
└── music/             # Uploaded audio files
```

## 🛠️ Technical Details

- **Backend**: Flask (Python)
- **Video Processing**: MoviePy + OpenCV
- **Image Processing**: Pillow (PIL)
- **Audio Processing**: MoviePy AudioFileClip
- **Format**: MP4 (H.264 video, AAC audio)
- **Resolution**: 1080x1920 (Instagram Reel format)
- **Frame Rate**: 30fps

## 🎨 Customization Options

- **Image Duration**: 0.5 to 10 seconds per image
- **Transitions**: Fade in/out effects
- **Audio**: Background music with auto-looping
- **Quality**: High-definition 1080p output

## 📱 Social Media Ready

The generated videos are optimized for:
- ✅ Instagram Reels (9:16 aspect ratio)
- ✅ TikTok (vertical format)
- ✅ YouTube Shorts (1080x1920)
- ✅ Facebook Stories
- ✅ Snapchat

## 💡 Pro Tips

1. **Image Quality**: Use high-resolution images for best results
2. **Consistent Style**: Use similar lighting/style for cohesive reels
3. **Music Choice**: Pick music that matches your content mood
4. **Duration**: 1-3 seconds per image works best for engagement
5. **File Size**: Keep total upload under 100MB for faster processing

## 🔧 Requirements

- Python 3.7+
- 2GB+ RAM (for video processing)
- Modern web browser
- Internet connection (for initial setup)

Transform your photos into engaging social media content with this powerful, easy-to-use reel maker! 🎬✨