import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from transformers import BertConfig, BertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import wandb
import numpy as np

# Initialize wandb
wandb.init(
    project="bert-biencoder-regression"
)


# Load dataset
dataset = load_dataset("minoosh/Annotated_story_pairs2")


# Initialize bi-encoder model (e.g., BERT as a sentence encoder)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)


# Tokenize both text1 and text2 independently
def preprocess_function(examples):
    text1_encodings = tokenizer(examples['text1'], truncation=True, padding=True, max_length=512)
    text2_encodings = tokenizer(examples['text2'], truncation=True, padding=True, max_length=512)
    return {
        'input_ids_text1': text1_encodings['input_ids'],
        'attention_mask_text1': text1_encodings['attention_mask'],
        'input_ids_text2': text2_encodings['input_ids'],
        'attention_mask_text2': text2_encodings['attention_mask'],
        'labels': examples['label']
    }



# Apply tokenization
tokenized_train = dataset['train'].map(preprocess_function, batched=True)
#tokenized_test = dataset['test'].map(preprocess_function, batched=True)
tokenized_val = dataset['validation'].map(preprocess_function, batched=True)

# Remove unnecessary columns and set format for PyTorch
columns_to_keep = ['input_ids_text1', 'attention_mask_text1', 'input_ids_text2', 'attention_mask_text2', 'labels']
tokenized_train.set_format(type='torch', columns=columns_to_keep)
#tokenized_test.set_format(type='torch', columns=columns_to_keep)
tokenized_val.set_format(type='torch', columns=columns_to_keep)



# Define a custom collator to handle text1 and text2 encoding
class BiEncoderCollator:
    def __call__(self, features):
        batch = {
            'input_ids_text1': torch.stack([f['input_ids_text1'] for f in features]),
            'attention_mask_text1': torch.stack([f['attention_mask_text1'] for f in features]),
            'input_ids_text2': torch.stack([f['input_ids_text2'] for f in features]),
            'attention_mask_text2': torch.stack([f['attention_mask_text2'] for f in features]),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.float)
        }
        return batch


collator = BiEncoderCollator()


# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    labels = labels.squeeze()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    pearson_corr, _ = pearsonr(predictions, labels)
    spearman_corr, _ = spearmanr(predictions, labels)
    cosine_sim = torch.nn.functional.cosine_similarity(torch.tensor(predictions), torch.tensor(labels), dim=0).mean().item()

    return {
        "mse": mse,
        "mae": mae,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "cosine_sim": cosine_sim  # Optional metric for similarity tasks
    }


# Define a custom BiEncoder model
class BiEncoderModel(torch.nn.Module):
    def __init__(self, base_model, config=None, loss_fn="mse"):
        super(BiEncoderModel, self).__init__()
        self.base_model = base_model
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.loss_fn = loss_fn
        self.config = config


    def forward(self, input_ids_text1, attention_mask_text1, input_ids_text2, attention_mask_text2, labels=None):
        # Encode text1 and text2 separately
        outputs_text1 = self.base_model(input_ids_text1, attention_mask=attention_mask_text1)
        outputs_text2 = self.base_model(input_ids_text2, attention_mask=attention_mask_text2)


        # Extract [CLS] token embeddings (first token)
        cls_embedding_text1 = outputs_text1.last_hidden_state[:, 0, :]
        cls_embedding_text2 = outputs_text2.last_hidden_state[:, 0, :]


        # Calculate cosine similarity between the two embeddings
        cos_sim = self.cos(cls_embedding_text1, cls_embedding_text2)

        loss = None
        if labels is not None:
            if self.loss_fn == "mse":
                loss_fct = torch.nn.MSELoss()  # Mean Squared Error Loss
            elif self.loss_fn == "mae":
                loss_fct = torch.nn.L1Loss()  # Mean Absolute Error Loss
            elif self.loss_fn == "contrastive":
                loss_fct = self.contrastive_loss
            elif self.loss_fn == "cosine_embedding":
                loss_fct = torch.nn.CosineEmbeddingLoss()  # Cosine Embedding Loss


            if self.loss_fn == "cosine_embedding":
                labels_cosine = 2 * (labels > 0.5).float() - 1  # Convert labels to binary for cosine embedding loss
                loss = loss_fct(cls_embedding_text1, cls_embedding_text2, labels_cosine)
            else:
                loss = loss_fct(cos_sim, labels)

        return {"loss": loss, "logits": cos_sim}


    def contrastive_loss(self, cos_sim, labels, margin=0.5):
        loss = torch.mean((1 - labels) * torch.pow(cos_sim, 2) + labels * torch.pow(torch.clamp(margin - cos_sim, min=0.0), 2))
        return loss



# Initialize the Bi-Encoder model with a specific loss function
def train_biencoder(loss_fn):
    # Load pre-trained BERT configuration and model
    config = BertConfig.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    # Initialize your custom BiEncoderModel with the BERT model and config
    bi_encoder_model = BiEncoderModel(base_model=bert_model, config=config, loss_fn=loss_fn)
    #bi_encoder_model = BiEncoderModel(base_model, loss_fn)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./output/bert-reg-biencoder-{loss_fn}",
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        logging_dir='./logs',           # Directory for logs
        logging_steps=10,               # Log every 10 steps
        per_device_train_batch_size=wandb.config['batch_size'],
        per_device_eval_batch_size=wandb.config['batch_size'],
        num_train_epochs=wandb.config['epochs'],
        warmup_steps=100,
        learning_rate=wandb.config['learning_rate'],
        weight_decay=0.01,
        report_to="wandb",
        save_strategy="epoch",          # Save checkpoints at the end of each epoch
        load_best_model_at_end=True,
        push_to_hub=True,
        save_total_limit=2              # Keep only the 2 most recent checkpoints

    )


    # Define the Trainer
    trainer = Trainer(
        model=bi_encoder_model,             # Custom BiEncoder model
        args=training_args,                 # Training arguments
        train_dataset=tokenized_train,      # Training dataset
        eval_dataset=tokenized_val,         # Validation dataset
        data_collator=collator,             # Custom collator for handling bi-encoder inputs
        compute_metrics=compute_metrics     # Function to compute metrics
    )

    # Train the model
    trainer.train()


    # Evaluate the model on the test set
    #trainer.evaluate(tokenized_test)

    # Save the model to Hugging Face Hub
    trainer.save_model(f"./output/bert-reg-biencoder-{loss_fn}")
    trainer.push_to_hub(f"minoosh/bert-reg-biencoder-{loss_fn}")



    # Finish wandb run
    wandb.finish()

    return trainer