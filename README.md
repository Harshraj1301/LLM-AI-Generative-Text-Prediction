
# LLM-AI-Generative-Text-Prediction

This repository contains code and resources for distinguishing AI-generated texts from human-written texts using BERT (Bidirectional Encoder Representations from Transformers). This project was developed as part of a Kaggle competition.

## Repository Structure

```
LLM-AI-Generative-Text-Prediction/
│
├── .DS_Store
├── .gitattributes
├── Gen Text Prediction Through Bert.ipynb
└── README.md
```

- `.DS_Store`: MacOS system file.
- `.gitattributes`: Configuration file to ensure consistent handling of files across different operating systems.
- `Gen Text Prediction Through Bert.ipynb`: Jupyter Notebook containing the code for building, training, and evaluating the text prediction model.
- `README.md`: This file. Provides an overview of the project and instructions for getting started.

## Introduction

As part of a Kaggle competition, I developed a machine learning model to distinguish AI-generated texts from human-written texts using BERT. The project involved the following key steps:

### Data Preparation

- Collected and integrated datasets, including Mistral AI-generated text datasets and the provided competition datasets.
- Conducted data cleaning and preprocessing, including text stemming and removal of punctuation and stopwords.

### Model Development

- Implemented BERT for sequence classification using a pre-trained BERT model.
- Fine-tuned the model on the combined training dataset, which included both human-written and AI-generated texts.

### Training and Evaluation

- Trained the model using a balanced dataset to address class imbalance.
- Achieved a training accuracy of 72% and a testing accuracy of 58%.
- Evaluated model performance using metrics such as accuracy and loss.

### Deployment

- Prepared a submission file for the competition by predicting the classification of texts in the test dataset.

## Key Technologies

- **NLP**: BERT, Tokenization, Text Preprocessing
- **Libraries**: Pandas, NumPy, PyTorch, Transformers
- **Data Visualization**: Matplotlib, Seaborn
- **Model Evaluation**: Accuracy, Confusion Matrix, Loss Calculation

## Getting Started

To get started with this project, follow the steps below:

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Harshraj1301/LLM-AI-Generative-Text-Prediction.git
```

2. Navigate to the project directory:

```bash
cd LLM-AI-Generative-Text-Prediction
```

3. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Usage

1. Open the Jupyter Notebook:

```bash
jupyter notebook "Gen Text Prediction Through Bert.ipynb"
```

2. Follow the instructions in the notebook to run the code cells and perform text classification using the BERT model.

### Code Explanation

The notebook `Gen Text Prediction Through Bert.ipynb` includes the following steps:

1. **Data Loading and Exploration**: Load and explore the datasets to understand their structure and content.
2. **Data Preprocessing**: Clean and preprocess the text data, including tokenization and text normalization.
3. **Model Implementation**: Implement the BERT model for sequence classification.
4. **Model Training**: Train the BERT model on the preprocessed text data.
5. **Model Evaluation**: Evaluate the model's performance using accuracy, loss, and confusion matrix.
6. **Text Prediction**: Use the trained model to classify new texts and generate the competition submission file.

Here are the contents of the notebook:

Importing Libraries

Loading the datasets

## Code Cells

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
snowball = SnowballStemmer(language='english')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as XGB
from sklearn.metrics import confusion_matrix
```

```python
# Load the datasets
train_essays = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
test_essays = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
prompts = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv')
```

```python
dataset_1_loc ='/kaggle/input/mistral-datasets/Mistral7B_CME_v6.csv'
aug_data1 = pd.read_csv(dataset_1_loc)
aug_data1 = aug_data1[aug_data1["prompt_id"]==2]
aug_data1["prompt_id"]=aug_data1['prompt_id']-2
aug_data1
```

```python
dataset_2_loc = '/kaggle/input/mistral-datasets/Mistral7B_CME_v7.csv'
aug_data2 = pd.read_csv(dataset_2_loc)
aug_data2 = aug_data2[aug_data2["prompt_id"]==12]
aug_data2["prompt_id"]=aug_data2['prompt_id']-11
aug_data2
```

```python
aug_data_mistral = pd.concat([aug_data1,aug_data2],axis=0)
aug_data_mistral
```

```python
aug_data_mistral = aug_data_mistral.drop(columns= ['prompt_name'])
aug_data_mistral
```

```python
train_csv= train_essays.drop(columns=['id'])
train_csv
```

```python
final_data = pd.concat([train_csv,aug_data_mistral],axis=0)
final_data
```

```python
classes = final_data.groupby('generated').count()['text']
plt.title('Class imbalance solved')
plt.pie(classes, labels=['generated by AI','not generated by AI'],colors=['orange','pink'],shadow=True,autopct='%0.2f%%')
```

```python
final_data['text'].index = np.arange(0,2778)
final_data['text']
```

```python
stemtext = []
len_text = []
para = final_data['text'].tolist()
for paragraph in para:
    char = [char for char in paragraph if char not in string.punctuation]
    word = "".join(char).split(" ")
    words = [word.lower() for word in word if word not in stopwords.words('english')]
    stemwords = [SnowballStemmer('english').stem(word) for word in words]
    len_text.append(len(stemwords))
    stemtext.append(" ".join(stemwords))
```

```python
final_data['text']=stemtext
final_data
```

```python
test_data = test_essays
train_data = final_data
```

```python
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# Path to the local directory containing tokenizer files
local_bert_directory = '/kaggle/input/local-bert/'

# Initialize the tokenizer from the local directory
tokenizer = BertTokenizer.from_pretrained(local_bert_directory)

# Initialize the tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EssayDataset(Dataset):
    def __init__(self, essays, targets, tokenizer, max_len):
        self.essays = essays
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, item):
        essay = str(self.essays[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'essay_text': essay,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

# Define max token length
MAX_LEN = 256

# Create the dataset
train_dataset = EssayDataset(
    essays=train_data['text'].to_numpy(),
    targets=train_data['generated'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

```

```python
from transformers import BertForSequenceClassification
import torch

# Path where the model directory was transferred
transferred_model_directory = '/kaggle/input/model-bert/'

# Load the model
model = BertForSequenceClassification.from_pretrained(transferred_model_directory)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

```python
from transformers import AdamW
from torch.utils.data import DataLoader

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss().to(device)
```

```python
from tqdm import tqdm

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Training loop
EPOCHS = 2

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        len(train_dataset)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

```

```python
from sklearn.model_selection import train_test_split

# Split the data
train_data, val_data = train_test_split(train_essays, test_size=0.1)

# Create datasets for training and validation
train_dataset = EssayDataset(
    essays=train_data['text'].to_numpy(),
    targets=train_data['generated'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = EssayDataset(
    essays=val_data['text'].to_numpy(),
    targets=val_data['generated'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# Create data loaders
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=16)
```

```python
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluate the model
val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    len(val_dataset)
)

print(f'Validation loss {val_loss}, accuracy {val_acc}')
```

```python
class TestEssayDataset(Dataset):
    def __init__(self, essays, tokenizer, max_len):
        self.essays = essays
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, item):
        essay = str(self.essays[item])

        encoding = self.tokenizer.encode_plus(
            essay,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'essay_text': essay,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Create the test dataset
test_dataset = TestEssayDataset(
    essays=test_essays['text'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# DataLoader for the test data
test_data_loader = DataLoader(test_dataset, batch_size=16)
```

```python
# Predicting on test data
model.eval()
test_predictions = []
with torch.no_grad():
    for d in test_data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        test_predictions.extend(preds.tolist())

# Prepare submission file
sample_submission = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
sample_submission['generated'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
```

## Results

The notebook includes the results of the text prediction tasks, showcasing the performance of the BERT model on distinguishing AI-generated texts from human-written texts.

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project was created as part of a Kaggle competition by Harshraj Jadeja.
- Thanks to the open-source community for providing valuable resources and libraries for NLP and deep learning.

---

Feel free to modify this `README.md` file as per your specific requirements and project details.
