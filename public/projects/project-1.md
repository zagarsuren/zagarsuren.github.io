# CodeWhisperer 🚀

**Overview**

CodeWhisperer is a powerful online code editor built with React and Tailwind CSS. It supports real-time code execution, syntax highlighting, and more.

## Features

- Real-time editor
- Multi-language support
- Sleek UI

## Tech Stack

- React
- Tailwind CSS
- Monaco Editor

## Code

```Python
class VizWizDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, answer2id, max_length=40):
        with open(annotation_file, "r") as f:
            self.samples = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.answer2id = answer2id
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)
```

```bash
!pip install -r requirements.txt
```

> Test