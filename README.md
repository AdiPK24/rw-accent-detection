# English Accent Detector

This tool analyzes videos to detect the speaker's English accent, providing a confidence score and explanation. It's designed to help in hiring processes by evaluating candidates' English accent characteristics.

## Features

- Accepts YouTube, Loom, or direct MP4 video URLs
- Detects 16 different English accent varieties
- Provides confidence scores for accent detection
- Includes explanations of accent characteristics
- Visual representation of accent probabilities
- Hiring recommendations based on confidence levels

## Supported Accents

The tool can detect the following English accents:
- American (US)
- British (England)
- Australian
- Canadian
- Scottish
- Irish
- Welsh
- Northern Irish
- Indian
- Singaporean
- New Zealand
- South African
- Malaysian
- Hong Kong
- Filipino
- Bermudian

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Enter a video URL (YouTube, Loom, or direct MP4 link) in the input field

4. Click "Analyze Accent" to start the analysis

5. View the results, which include:
   - Detected accent
   - Confidence score
   - Accent explanation
   - Probability distribution across all accents
   - Hiring recommendation

## Technical Details

- Built with Streamlit for the web interface
- Uses SpeechBrain's accent classifier model
- Implements audio extraction from various video sources
- Provides real-time analysis and visualization

## Requirements

- Python 3.7+
- FFmpeg (for audio processing)
- Internet connection (for video downloading and model loading)

## Deployment

The application can be deployed on:
- Streamlit Cloud
- Hugging Face Spaces
- Heroku
- Any platform that supports Python web applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 