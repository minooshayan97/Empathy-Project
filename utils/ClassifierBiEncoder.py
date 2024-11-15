import torch

from datasets import load_dataset

from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer

from transformers import BertConfig, BertModel

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb

import numpy as np



# Initialize wandb

wandb.init(

    project="bert-biencoder-classification"

)



# Load dataset

dataset = load_dataset("minoosh/EPITOME_pairs")



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

tokenized_test = dataset['test'].map(preprocess_function, batched=True)

tokenized_val = dataset['validation'].map(preprocess_function, batched=True)



# Remove unnecessary columns and set format for PyTorch

columns_to_keep = ['input_ids_text1', 'attention_mask_text1', 'input_ids_text2', 'attention_mask_text2', 'labels']

tokenized_train.set_format(type='torch', columns=columns_to_keep)

tokenized_test.set_format(type='torch', columns=columns_to_keep)

tokenized_val.set_format(type='torch', columns=columns_to_keep)



# Define a custom collator to handle text1 and text2 encoding

class BiEncoderCollator:

    def __call__(self, features):

        batch = {

            'input_ids_text1': torch.nn.utils.rnn.pad_sequence(

                [torch.tensor(f['input_ids_text1']) for f in features], batch_first=True),

            'attention_mask_text1': torch.nn.utils.rnn.pad_sequence(

                [torch.tensor(f['attention_mask_text1']) for f in features], batch_first=True),

            'input_ids_text2': torch.nn.utils.rnn.pad_sequence(

                [torch.tensor(f['input_ids_text2']) for f in features], batch_first=True),

            'attention_mask_text2': torch.nn.utils.rnn.pad_sequence(

                [torch.tensor(f['attention_mask_text2']) for f in features], batch_first=True),

            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)  # Change to long for classification

        }

        return batch



collator = BiEncoderCollator()



# Define the compute_metrics function for classification with precision and recall

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)

    f1 = f1_score(labels, preds, average="weighted")

    precision = precision_score(labels, preds, average="weighted")

    recall = recall_score(labels, preds, average="weighted")

    return {

        "accuracy": accuracy,

        "f1": f1,

        "precision": precision,

        "recall": recall,

    }



# Define a custom BiEncoder model with options for different loss functions

class BiEncoderModel(torch.nn.Module):

    def __init__(self, base_model, config=None, num_classes=4, loss_fn="cross_entropy"):

        super(BiEncoderModel, self).__init__()

        self.base_model = base_model

        self.config = config  # Add this line to set the config attribute

        self.classifier = torch.nn.Linear(base_model.config.hidden_size * 2, num_classes)  # Updated for 4 classes

        self.loss_fn = loss_fn



    def forward(self, input_ids_text1, attention_mask_text1, input_ids_text2, attention_mask_text2, labels=None):

        # Encode text1 and text2 separately

        outputs_text1 = self.base_model(input_ids_text1, attention_mask=attention_mask_text1)

        outputs_text2 = self.base_model(input_ids_text2, attention_mask=attention_mask_text2)



        # Extract [CLS] token embeddings (first token)

        cls_embedding_text1 = outputs_text1.last_hidden_state[:, 0, :]

        cls_embedding_text2 = outputs_text2.last_hidden_state[:, 0, :]



        # Concatenate embeddings and apply classifier

        concatenated_embeddings = torch.cat([cls_embedding_text1, cls_embedding_text2], dim=1)

        logits = self.classifier(concatenated_embeddings)



        loss = None

        if labels is not None:

            if self.loss_fn == "cross_entropy":

                loss_fct = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for classification

                loss = loss_fct(logits, labels)

            elif self.loss_fn == "focal_loss":

                # Focal loss implementation

                alpha = 0.25

                gamma = 2.0

                ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)

                pt = torch.exp(-ce_loss)  # Probability of the true class

                loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()

            elif self.loss_fn == "kl_divergence":

                # KL Divergence for soft-label classification

                kl_div = torch.nn.KLDivLoss(reduction="batchmean")

                soft_labels = torch.nn.functional.one_hot(labels, num_classes=self.classifier.out_features).float()

                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                loss = kl_div(log_probs, soft_labels)

            else:

                raise ValueError(f"Unsupported loss function: {self.loss_fn}")



        return {"loss": loss, "logits": logits}



# Initialize the Bi-Encoder model with specified loss function

def train_biencoder(loss_fn="cross_entropy"):

    # Load pre-trained BERT configuration and model

    config = BertConfig.from_pretrained(model_name)

    bert_model = BertModel.from_pretrained(model_name)



    # Initialize your custom BiEncoderModel with the BERT model, config, and loss function

    bi_encoder_model = BiEncoderModel(base_model=bert_model, config=config, loss_fn=loss_fn)



    # Define TrainingArguments

    training_args = TrainingArguments(

        output_dir=f"./output/bert-clf-biencoder-{loss_fn}",

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



    #trainer.model = trainer.model.base_model



    # Save and push the model to the Hugging Face Hub

    trainer.save_model(f"./output/bert-clf-biencoder-{loss_fn}")

    trainer.push_to_hub(f"minoosh/bert-clf-biencoder-{loss_fn}")



    # Finish wandb run

    wandb.finish()



    return trainer