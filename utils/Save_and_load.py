import os

import json

from huggingface_hub import HfApi

from transformers import AutoModel, AutoConfig, AutoTokenizer, BertConfig



def save_and_push_to_hub(trainer, repo_id, token=None):

    """

    Save and push BiEncoder model to Hugging Face Hub

    """

    api = HfApi()

    

    try:

        temp_save_path = f"temp_save_{repo_id.split('/')[-1]}"

        os.makedirs(temp_save_path, exist_ok=True)

        

        print(f"Saving model to {temp_save_path}...")

        

        # 1. Save the base model configuration

        base_config = trainer.model.base_model.config.to_dict()

        base_config["model_type"] = "bert"  # Ensure we're using BERT as base

        base_config["architectures"] = ["BertModel"]

        

        with open(os.path.join(temp_save_path, "config.json"), 'w') as f:

            json.dump(base_config, f)

            

        # 2. Save model weights

        torch.save(trainer.model.state_dict(), os.path.join(temp_save_path, "pytorch_model.bin"))

        

        # 3. Save tokenizer

        print("Saving tokenizer...")

        if hasattr(trainer, 'tokenizer'):

            trainer.tokenizer.save_pretrained(temp_save_path)

        

        # 4. Create model card

        model_card = f"""---

language: en

tags:

- bert

- classification

- pytorch

pipeline_tag: text-classification

---



# BiEncoder Classification Model



This model is a BiEncoder architecture based on BERT for text pair classification.



## Model Details

- Base Model: bert-base-uncased

- Architecture: BiEncoder with BERT base

- Number of classes: {trainer.model.classifier.out_features}



## Usage



```python

from transformers import AutoTokenizer

import torch



# Load tokenizer

tokenizer = AutoTokenizer.from_pretrained("{repo_id}")



# Load model weights

state_dict = torch.load("pytorch_model.bin")



# Initialize model (you'll need the BiEncoderModel class)

model = BiEncoderModel(

    base_model=AutoModel.from_pretrained("bert-base-uncased"),

    num_classes={trainer.model.classifier.out_features}

)

model.load_state_dict(state_dict)

```

"""

        with open(os.path.join(temp_save_path, "README.md"), 'w') as f:

            f.write(model_card)

        

        # 5. Push to hub

        print(f"Pushing to hub at {repo_id}...")

        api.upload_folder(

            folder_path=temp_save_path,

            repo_id=repo_id,

            token=token

        )

        

        print(f"Successfully pushed model to {repo_id}")

        

    except Exception as e:

        print(f"Error during push to hub: {str(e)}")

        raise

    finally:

        if os.path.exists(temp_save_path):

            import shutil

            shutil.rmtree(temp_save_path)



def load_from_hub(repo_id, num_classes=4):

    """

    Load BiEncoder model from Hugging Face Hub

    """

    try:

        print(f"Loading model from {repo_id}...")

        

        # 1. Initialize base model with BERT

        base_model = AutoModel.from_pretrained("bert-base-uncased")

        

        # 2. Create BiEncoder model

        model = BiEncoderModel(

            base_model=base_model,

            num_classes=num_classes

        )

        

        # 3. Load state dict

        state_dict = torch.hub.load_state_dict_from_url(

            f"https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin",

            map_location="cpu"

        )

        model.load_state_dict(state_dict)

        

        # 4. Load tokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo_id)

        

        # 5. Create trainer

        trainer = Trainer(

            model=model,

            data_collator=BiEncoderCollator(),

            compute_metrics=compute_metrics

        )

        

        print("Model loaded successfully!")

        return trainer, model, tokenizer

        

    except Exception as e:

        print(f"Error loading model from hub: {str(e)}")

        raise