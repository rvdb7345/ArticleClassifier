'''Code based on: https://www.kaggle.com/code/vpkprasanna/bert-model-with-0-845-accuracy/notebook'''

import copy
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

sys.path.append("/home/jovyan/20230406_ArticleClassifier/ArticleClassifier")

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname('data_loader.py'), os.path.pardir)))
from src.general.utils import cc_path
from src.data_preparation.text_embedding.embedding_models.bert_finetuning_model import BertClassifier
from src.data_preparation.text_embedding.bert_utils import generate_litcovid_embedding_text, \
    load_canary_data, load_litcovid_data, generate_dataloader_objects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = torch.round(torch.sigmoid(eval_preds.predictions))
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
def initialize_model(dataloaders, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """

    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(dataloaders['train']) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    # Specify loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    return bert_classifier, optimizer, loss_fn, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, dataloaders, optimizer, scheduler, loss_fn, num_labels, epochs=4, evaluation=False, dataset_name='litcovid'):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    max_val_accuracy = 0
    not_improved = 0
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(dataloaders['train'])):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels.float())
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20--50000 batches
            if (step % 50000 == 0 and step != 0) or (step == len(dataloaders['train']) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(dataloaders['train'])

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy, _ = evaluate(model, dataloaders['val'])

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.5f} | {time_elapsed:^9.2f}")
            print("-" * 70)

            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
                torch.save(model.bert, cc_path(f'models/embedders/scibert_{dataset_name}_best_iter.pt'))
                not_improved = 0
            else:
                not_improved += 1

            if not_improved == 5:
                break
        print("\n")

    print("Training complete!")

    evaluate(best_model, dataloaders['test'], loss_fn, num_labels)

    return best_model

def evaluate(model, dataloader, loss_fn, num_labels=7):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    list_val_f1_micro = []
    list_val_f1_macro = []
    val_loss = []

    predictions = []
    all_labels = []

    # For each batch in our validation set...
    for batch in tqdm(dataloader):
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels.float())
        val_loss.append(loss.item())

        # Append batch predictions to the list
        predictions.append(logits)
        all_labels.append(b_labels)

    # Combine predictions for all batches into a single tensor
    predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate f1 scores
    f1_micro, f1_macro = accuracy_thresh(predictions.view(-1, num_labels), all_labels.view(-1, num_labels))

    # Compute the average loss over the validation set.
    val_loss = np.mean(val_loss)
    # Calculate the average f1 scores
    val_f1_micro = f1_micro.mean().item()
    val_f1_macro = f1_macro.mean().item()

    return val_loss, val_f1_micro, val_f1_macro


def accuracy_thresh(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_pred[(y_pred > thresh)] = 1
    y_pred[(y_pred < thresh)] = 0
    return f1_score(y_true.byte().cpu(), y_pred.cpu(), average='micro'), f1_score(y_true.byte().cpu(), y_pred.cpu(),
                                                                                  average='macro')
    # return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()
def scibert_finetuning(dataset_to_run):
    # Load the pre-trained SciBERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

    if dataset_to_run == 'canary':
        label_columns, processed_df, puis = load_canary_data()
        processed_df = generate_litcovid_embedding_text(processed_df)
        num_labels = 52
    elif dataset_to_run == 'litcovid':
        label_columns, processed_df, puis = load_litcovid_data()
        processed_df = generate_litcovid_embedding_text(processed_df)
        num_labels = 7
    else:
        assert False, f'{dataset_to_run} not recognized as known dataset.'

    dataloaders, datasets = generate_dataloader_objects(tokenizer, label_columns, processed_df, puis,
                                                        batch_size=32)

    set_seed(42)  # Set seed for reproducibility
    bert_classifier, optimizer, loss_fn, scheduler = initialize_model(dataloaders, epochs=15)
    best_model = train(bert_classifier, dataloaders, optimizer, scheduler, loss_fn, num_labels,
                       epochs=15, evaluation=True, dataset_name=dataset_to_run)

    torch.save(best_model.bert, cc_path(f'models/embedders/litcovid_finetuned_bert_20e_3lay_meta.pt'))

if __name__ == '__main__':
    dataset_to_run = 'litcovid'
    scibert_finetuning(dataset_to_run)
