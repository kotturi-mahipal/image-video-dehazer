# Image and Video Dehazing Application

This application provides a simple web interface for dehazing images and videos. It uses OpenCV for image and video processing and includes a placeholder for object detection using YOLO.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kotturi-mahipal/image-video-dehazer.git
   cd image-video-dehazing
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create necessary folders:**
   - `templates`: This folder will contain the HTML templates for the web interface.
     - Create `index.html`, `show_image.html`, and `show_video.html` inside this folder.
   - `static`: This folder will store uploaded images and processed outputs.
     - Create `uploads` and `output` subfolders inside this folder.

4. **Download YOLOv3 weights and configuration:**
   - Download `yolov3.weights` and `yolov3.cfg` from the official YOLO website or a trusted source.
   - Place these files in the root directory of the project.

5. **Update the code:**
   - In `app.py`, update the paths to the YOLOv3 weights and configuration files in the `detect_objects` function.
   - You may also need to adjust other paths based on your specific setup.

## Running the Application

Start the Flask development server:

```bash
flask run
```

Open a web browser and go to the provided address (usually http://127.0.0.1:5000/).

Use the forms on the webpage to upload images and videos for processing.

## Additional Notes

- The `video_dehaze` function uses the XVID codec. You might need to install the necessary codecs on your system.
- For production environments, consider using a WSGI server like Gunicorn or uWSGI instead of the Flask development server.
