# 🎯 FolkIn — AI-Powered Focus Assistant

FolkIn is an AI-powered focus assistant designed to help students stay focused while studying. It uses real-time webcam monitoring to detect phone usage and provides audio alerts and AI-generated motivational messages when distractions are detected.

## ✨ Features

- **Real-Time Phone Detection:** Uses a YOLO object detection model to detect the `phone_in_hand` class through a webcam.
- **Smart Distraction Alerts:** Displays a warning when phone usage is detected and triggers an alert after the detection persists for approximately 1.5 seconds.
- **Audio Alert System:** Plays an audio warning using Pygame when a distraction is detected.
- **AI Focus Assistant:** Uses Qwen2.5-0.5B-Instruct with a LoRA adapter to generate contextual messages that encourage users to return to their study tasks.
- **Interactive Web Interface:** Provides a browser-based interface built with Gradio.
- **Task-Aware Messages:** Allows users to enter their current study task so AI-generated reminders can be more relevant.

## 🧠 How It Works

1. The user starts the application and allows webcam access.
2. YOLO analyzes webcam frames to identify whether a phone is being held.
3. If the detection persists, the interface displays a distraction warning.
4. After the detection exceeds the configured threshold, the application plays an audio alert.
5. Qwen generates a contextual reminder based on the detection event and the user's study task.
6. When the phone is no longer detected, the application resets the detection timer and stops the alert sound.

## 🛠️ Tech Stack

- Python
- Gradio
- Ultralytics YOLO
- OpenCV
- NumPy
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- Pygame

## 📁 Project Structure

```text
FolkIn/
├── app.py
├── alert.wav
├── README.md
├── requirements.txt
├── models/
│   ├── best.pt
│   └── folkin-qwen-lora/
├── scripts/
│   └── llm_assistant.py
│   └── webcam_detect.py
├── dataset/
└── notebooks/
```

The structure above is an example. Update it to match the actual files and folders in your repository.

## ⚙️ Requirements

- Python 3.10 or a compatible Python version supported by the selected dependencies
- A working webcam
- A microphone is not required for the current implementation
- Sufficient RAM and processing power to load the AI models
- Internet access for the initial download of the Qwen base model, unless it is already cached locally

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FolkIn.git
cd FolkIn
```

Replace `YOUR_USERNAME` with your GitHub username and `FolkIn` with the repository name if different.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created, install the project's dependencies first and generate a tested dependency list before sharing the repository.

### 4. Prepare the models

Ensure the following resources are available:

- `models/best.pt`: the trained YOLO model used for phone detection.
- `models/folkin-qwen-lora/`: the trained LoRA adapter files.
- The Qwen base model required by `scripts/llm_assistant.py`.

Large model files may need to be downloaded separately if they are not included in the repository.

### 5. Run FolkIn

```bash
python app.py
```

Open the local Gradio URL displayed in the terminal and allow webcam access.

## 🧪 Model Development and Evaluation

FolkIn combines computer vision and a language model:

- **YOLO:** Identifies phones held by the user in webcam frames.
- **Qwen2.5-0.5B-Instruct:** Generates contextual text reminders.
- **LoRA fine-tuning:** Adapts a small set of model parameters to the intended focus-assistant behavior.

To evaluate the language model adaptation, compare the base model and the LoRA-adapted model using the same test prompts. Assess response relevance, clarity, consistency, and suitability for the study context.

Record actual evaluation results before making claims about improvements.

## ⚠️ Limitations

- Detection accuracy depends on lighting, camera angle, object visibility, and the YOLO model's training data.
- False positives and false negatives may occur.
- The current implementation detects the `phone_in_hand` class; it does not independently verify whether the user is actively scrolling or using an application.
- AI-generated messages may vary in quality and should be evaluated before drawing conclusions about the fine-tuning results.
- Running both models simultaneously may require additional memory and processing resources.

## 🔒 Privacy

The application uses webcam frames for local detection in the current implementation. Review the model and application code before deployment to verify whether any images, video frames, or generated messages are transmitted or stored elsewhere.

## 📌 Project Status

FolkIn is a prototype under active development. Features and model performance may change as the dataset, detection model, and LoRA adapter are improved.

## 👥 Credits and License

Include the original dataset sources, dataset licenses, model references, and any required attribution here.

Check the license terms of each dataset, pretrained model, and audio asset before redistributing them.

---

**FolkIn — Lock in, stay focused, and make progress.**
