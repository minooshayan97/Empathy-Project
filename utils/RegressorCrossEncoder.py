import torch

from datasets import load_dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, TrainingArguments, Trainer

from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats import pearsonr, spearmanr

import wandb



# Initialize wandb

wandb.init(

    project="bert-crossencoder-regression"

)



# Load dataset

dataset = load_dataset("minoosh/Annotated_story_pairs2")



# Initialize the tokenizer and model for cross-encoder setup

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)



# Preprocess data for the cross-encoder model by concatenating text1 and text2 with [SEP]

def preprocess_function(examples):

    # Concatenate both texts with a [SEP] token in between

    encodings = tokenizer(examples['text1'], examples['text2'], truncation=True, padding=True, max_length=512)

    encodings['labels'] = examples['label']

    return encodings



# Apply tokenization

tokenized_train = dataset['train'].map(preprocess_function, batched=True)

tokenized_test = dataset['test'].map(preprocess_function, batched=True)

tokenized_val = dataset['validation'].map(preprocess_function, batched=True)



# Set format for PyTorch

tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])



# Define compute_metrics function for regression evaluation

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



# Custom Cross-Encoder model class with config

class CrossEncoderModel(torch.nn.Module):

    def __init__(self, model_name, loss_fn="mse"):

        super(CrossEncoderModel, self).__init__()

        # Load model config

        self.config = AutoConfig.from_pretrained(model_name, num_labels=1)  # Specify 1 output for regression

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

        self.loss_fn = loss_fn



    def forward(self, input_ids, attention_mask, labels=None):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits.squeeze()  # Output logits for regression



        loss = None

        if labels is not None:

            if self.loss_fn == "mse":

                loss_fct = torch.nn.MSELoss()

            elif self.loss_fn == "mae":

                loss_fct = torch.nn.L1Loss()

            elif self.loss_fn == "cosine_embedding":
                loss_fct = torch.nn.CosineEmbeddingLoss()
                labels_cosine = 2 * (labels > 0.5).float() - 1  # Convert to binary for cosine embedding loss
            
                # Make sure to provide a target similarity score (1 for similar, -1 for dissimilar)
                # Assuming you need to compute the target based on labels
                target = labels_cosine  # This can also be -1 or 1 depending on your implementatio
            elif self.loss_fn == "contrastive":
                loss_fct = self.contrastive_loss
            else:
                raise ValueError(f"Unknown loss function: {self.loss_fn}")

            if self.loss_fn == "cosine_embedding":
                loss = loss_fct(logits, target)  # Compute loss
                print("Logits shape:", logits.shape)
                print("Labels cosine shape:", labels_cosine.shape)
            else:
                loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}



    def contrastive_loss(self, logits, labels, margin=0.5):

        positive_pairs = labels * torch.pow(1 - logits, 2)  # For similar pairs (y=1)

        negative_pairs = (1 - labels) * torch.pow(torch.clamp(margin - logits, min=0.0), 2)  # For dissimilar pairs (y=0)

        return torch.mean(positive_pairs + negative_pairs)



# Function to initialize and train the cross-encoder model

def train_crossencoder(loss_fn):

    # Initialize the cross-encoder model with the specified loss function

    model = CrossEncoderModel(model_name=model_name, loss_fn=loss_fn)



    # Set up TrainingArguments

    training_args = TrainingArguments(

        output_dir=f"./output/bert-reg-crossencoder-{loss_fn}",

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

    trainer.save_model(f"./output/bert-reg-crossencoder-{loss_fn}")

    trainer.push_to_hub(f"minoosh/bert-reg-crossencoder-{loss_fn}")



    # End the wandb run

    wandb.finish()