# VisionAid-VQA: Inclusive Visual Question Answering Using Deep Learning and Multimodal Attention Mechanisms

Visual Question Answering (VQA) is a complex multimodal task that requires the integration
of visual perception and natural language understanding to generate relevant answers. This
work focuses on enhancing accessibility for visually impaired users by adapting VQA
technology to their specific needs. We propose VisionAid-VQA, a user-centric system built by
fine-tuning two state-of-the-art vision-language models—ViLT and Florence-2—on a
representative subset of the VizWiz dataset, which contains real-world visual questions
submitted by blind users. The models were evaluated based on their ability to produce accurate,
context-aware responses. To ensure practical usability, we developed an interactive graphical
user interface (GUI) that enables image upload, question input, and answer retrieval.
Experimental results show that Florence-2 significantly outperforms ViLT, achieving 58.21%
overall accuracy versus 29.01%, including substantial improvements in number-based (73.08%
vs. 7.69%), open-ended (59.21% vs. 20.22%), and unanswerable (60.61% vs. 0%) question
categories, while ViLT performs better on yes/no questions (71.32% vs. 48.84%). In addition,
Florence-2 significantly outperforms ViLT in language generation quality, achieving an
average BLEU-1 score of 0.6386 compared to ViLT’s 0.3017. These findings underscore the
significance of domain-specific fine-tuning and model architecture in developing inclusive AI
systems that promote accessibility.

---

## 🧠 Features

This system allows users to upload or capture an image, ask a question about it, and receive **text and audio responses** using advanced **Visual Question Answering (VQA)** models. The system is designed for accessibility, especially supporting visually impaired users.

![img](https://github.com/zagarsuren/visionaid-vqa/blob/main/assets/demo/app.jpeg?raw=true)

- 🔍 Supports VQA models:
  - `vilt_finetuned_vizwiz` (Transformer-based vision language model finetuned with VizWiz)
  - `florence2-finetuned` (Unified vision language model finetuned with VizWiz)
- 📷 Accepts image input from upload or camera
- ❓ Accepts natural language questions 
- 🔊 Converts text answers to speech using `gTTS`
- 🎧 Auto-plays audio response in the app

## High Level Design
![img](https://i.imgur.com/5StHHvp.jpeg)

## 🧠 Models

1) Vision Language Transformer (ViLT)

![img](https://i.imgur.com/lfZ68DA.jpeg)

2) Florence-2

![img](https://i.imgur.com/iTxz7OZ.jpeg)

**Model weights:**

- `ViLT` → `/models/vilt_finetuned_vizwiz`. ViLT model weight can be found at: [https://huggingface.co/Zagarsuren/vilt-finetuned-vizwiz](https://huggingface.co/Zagarsuren/vilt-finetuned-vizwiz)
- `Florence2Model` → `/models/florence2-finetuned` Florence-2 model weight can be found at: [https://huggingface.co/Zagarsuren/florence2-finetuned-vizwiz](https://huggingface.co/Zagarsuren/florence2-finetuned-vizwiz)

## License
This project is licensed under the MIT License.