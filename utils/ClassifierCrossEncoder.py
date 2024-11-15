import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb

# Initialize wandb
wandb.init(
    project="bert-crossencoder-classification"
)

# Load dataset
dataset = load_dataset("minoosh/EPITOME_pairs")

# Initialize the tokenizer and model for cross-encoder setup
model_name = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess data for the cross-encoder model by concatenating text1 and text2 with [SEP]
def preprocess_function(examples):
    # Concatenate both texts with a [SEP] token in between
    encodings = tokenizer(examples['text1'], examples['text2'], truncation=True, padding=True, max_length=512)
    encodings['labels'] = examples['label']  # Add labels
    return encodings

# Apply tokenization
tokenized_train = dataset['train'].map(preprocess_function, batched=True)
#tokenized_test = dataset['test'].map(preprocess_function, batched=True)
tokenized_val = dataset['validation'].map(preprocess_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
#tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])



# Define compute_metrics function for classification evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Custom Cross-Encoder model class for classification
class CrossEncoderModel(torch.nn.Module):
    def __init__(self, model_name, num_classes=4, loss_fn="cross_entropy"):
        super(CrossEncoderModel, self).__init__()
        # Load model config
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.loss_fn = loss_fn


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Output logits for classification

        loss = None
        if labels is not None:
            if self.loss_fn == "cross_entropy":
                loss_fct = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
                loss = loss_fct(logits, labels)
            elif self.loss_fn == "focal_loss":
                # Focal loss implementation for handling class imbalance
                alpha = 0.25
                gamma = 2.0
                ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
                pt = torch.exp(-ce_loss)  # Probability of the true class
                loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
            elif self.loss_fn == "kl_divergence":
                # KL Divergence for soft-label classification
                kl_div = torch.nn.KLDivLoss(reduction="batchmean")
                soft_labels = torch.nn.functional.one_hot(labels, num_classes=self.config.num_labels).float()
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = kl_div(log_probs, soft_labels)

            else:
                raise ValueError(f"Unsupported loss function: {self.loss_fn}")

        return {"loss": loss, "logits": logits}


# Function to initialize and train the cross-encoder model
def train_crossencoder(loss_fn):
    model = CrossEncoderModel(model_name=model_name, loss_fn=loss_fn)

    # Set up TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./output/bert-clf-crossencoder-{loss_fn}",
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        per_device_train_batch_size=wandb.config['batch_size'],
        per_device_eval_batch_size=wandb.config['batch_size'],
        num_train_epochs=wandb.config['epochs'],
        warmup_steps=100,
        learning_rate=wandb.config['learning_rate'],
        weight_decay=0.01,
        report_to="wandb",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        save_total_limit=2
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the test set
    #trainer.evaluate(tokenized_test)
    
    trainer.model = trainer.model.model

    # Save and push the model to the Hugging Face Hub
    trainer.save_model(f"./output/bert-clf-crossencoder-{loss_fn}")
    trainer.push_to_hub(f"minoosh/bert-clf-crossencoder-{loss_fn}")

    # End the wandb run
    wandb.finish()
    
    return trainer