# SMS Spam Detection using Machine Learning and Deep Learning

This project focuses on detecting spam messages in SMS text using a variety of machine learning and deep learning techniques. It leverages popular Python libraries including Scikit-learn, XGBoost, and Hugging Face Transformers to preprocess data, train models, and evaluate performance.

## Features

- Text preprocessing (cleaning, tokenization, etc.)
- Exploratory Data Analysis (EDA) and visualization
- Feature extraction using TF-IDF
- Model training using:
  - Random Forest
  - Support Vector Machines (SVM)
  - XGBoost
  - BERT-based Transformers
- Performance evaluation using metrics like accuracy, F1-score, precision, recall, ROC-AUC

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- Transformers (Hugging Face)
- PyTorch
- Pandas, NumPy, Matplotlib, Seaborn
- Google Colab (for notebook execution and data storage)

## Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost transformers wordcloud tqdm
```

### Usage

1. Clone this repository or upload the notebook to Google Colab.
2. Mount your Google Drive (if using Colab) to access the dataset.
3. Run the cells sequentially to perform:
   - Data loading and preprocessing
   - EDA and visualization
   - Model training and evaluation

## Dataset

The dataset should be a CSV file named `Spam_SMS.csv`, containing SMS messages labeled as spam or ham (not spam). Typical columns include:

- `label`: Indicates whether the message is 'spam' or 'ham'
- `message`: The text content of the SMS

## Results

Model performance is evaluated using a confusion matrix and classification report. The best-performing model can be saved and reused for predictions on new data.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- [Hugging Face](https://huggingface.co/)
- [Scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)