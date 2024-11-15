#!huggingface-cli login --token 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

from datasets import Dataset
import pandas as pd

'''df = pd.read_csv('Annotated_story_pairs2.csv')
df['label'] = df['Rate']
df['text1'] = df['Shared_stories']
df['text2'] = df['Response_strories']'''

df = pd.read_csv('EPITOME_pairs2.csv')
df['label'] = df['level']
df['text1'] = df['seeker_post']
df['text2'] = df['response_post']


df = df[['text1', 'text2', 'label']]
ds = Dataset.from_pandas(df)

# Split dataset into train and validation sets
ds = ds.train_test_split(test_size=0.2, seed=42)
ds_ = ds['test'].train_test_split(test_size=0.5, seed=42)

train_ds = ds['train']
test_ds = ds_['train']
val_ds = ds_['test']

'''train_ds.push_to_hub("minoosh/Annotated_story_pairs2", split='train')
test_ds.push_to_hub("minoosh/Annotated_story_pairs2", split='test')
val_ds.push_to_hub("minoosh/Annotated_story_pairs2", split='validation')'''

train_ds.push_to_hub("minoosh/EPITOME_pairs", split='train')
test_ds.push_to_hub("minoosh/EPITOME_pairs", split='test')
val_ds.push_to_hub("minoosh/EPITOME_pairs", split='validation')


