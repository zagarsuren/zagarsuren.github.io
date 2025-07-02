## VisionAid-VQA: Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms

### Project Overview:

VisionAid-VQA is an inclusive AI system built to empower visually impaired users by providing accessible, real-time Visual Question Answering (VQA). The project fine-tunes state-of-the-art multimodal models—ViLT and Florence-2—on the VizWiz dataset, which consists of real-world image-question pairs submitted by blind users.
The system combines advanced vision-language reasoning with a user-friendly interface that enables users to upload images, ask natural language questions, and receive both textual and spoken answers.

### 🧠 Core Technologies:
- Vision-Language Models: ViLT, Florence-2
- Dataset: VizWiz VQA (Real user questions from blind participants)
- PyTorch, Transformers, Streamlit, gTTS (Google Text-to-Speech)
- BLEU-1 Scoring, Per-type Accuracy Evaluation
- Accessible GUI with image capture/upload, question input, and audio response

### Summary of Results:
Florence-2 significantly outperformed ViLT in most categories:
- Overall Accuracy: 58.21% vs. 29.01%
- Number Questions: 73.08% vs. 7.69%
- Open-ended: 59.21% vs. 20.22%
- BLEU-1 Scores (Answer fluency): Florence-2: 0.6386 vs. ViLT: 0.3017


### User Interface
This system allows users to upload or capture an image, ask a question about it, and receive **text and audio responses** using advanced **Visual Question Answering (VQA)** models. The system is designed for accessibility, especially supporting visually impaired users.

![img](https://github.com/zagarsuren/visionaid-vqa/blob/main/assets/demo/app.jpeg?raw=true)

- 🔍 Supports VQA models:
  - `vilt_finetuned_vizwiz` (Transformer-based vision language model finetuned with VizWiz)
  - `florence2-finetuned` (Unified vision language model finetuned with VizWiz)
- 📷 Accepts image input from upload or camera
- ❓ Accepts natural language questions 
- 🔊 Converts text answers to speech using `gTTS`
- 🎧 Auto-plays audio response in the app

### High Level Design
![img](https://i.imgur.com/5StHHvp.jpeg)

### 🧠 Models

1) Vision Language Transformer (ViLT)

![img](https://i.imgur.com/lfZ68DA.jpeg)

2) Florence-2

![img](https://i.imgur.com/iTxz7OZ.jpeg)

**Model weights:**

- `ViLT` → `/models/vilt_finetuned_vizwiz`. ViLT model weight can be found at: [https://huggingface.co/Zagarsuren/vilt-finetuned-vizwiz](https://huggingface.co/Zagarsuren/vilt-finetuned-vizwiz)
- `Florence2Model` → `/models/florence2-finetuned` Florence-2 model weight can be found at: [https://huggingface.co/Zagarsuren/florence2-finetuned-vizwiz](https://huggingface.co/Zagarsuren/florence2-finetuned-vizwiz)

### License
This project is licensed under the MIT License.