from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# TODOS:
#clean up code
#separate visualizations from actual ml stuff for clarity
#look into tokenizer replacement


def prepare_data(df):
    #as usual in machine learning, we have to split up the data into training and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    return train_df, val_df

#dataloaders split up the data set into batches, which allows for better efficiency & processing
#note that batch size is defaulted to 16, you can increase or decrease this to get better speed or accuracy, respectively
def create_dataloaders(texts, labels, tokenizer, batch_size=16):
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader

#here the actual training takes place, keep in mind a lot of this code is for visualizing the training progress
def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5) #lr = learning rate, can adjust this as needed
    best_val_accuracy = 0
    
    #it is reccomended to do 3-5 epochs when fine-tuning bert, we default to 3 epochs
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_pbar = tqdm(train_loader, desc="Training") #this is making the progress bar
        
        for batch in train_pbar:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            optimizer.zero_grad()
            
            #get the outputs 
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ) 
            
            #get the loss (which helps us understand how much error we have)
            loss = outputs.loss
            
            # calculate accuracy & total loss
            total_train_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # update progress bar
            train_pbar.set_description(
                f"Training - Loss: {loss.item():.4f}, Acc: {100 * train_correct/train_total:.2f}%"
            )
            
            #backpropogate aka: update the model with what we've learned
            loss.backward() 
            optimizer.step()
        
        # validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        print("\nValidating...")
        val_pbar = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                #again, getting the outputs of the model
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                #again, we get the loss
                loss = outputs.loss
                total_val_loss += loss.item()
                
                #calculating how well our predictions did
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                val_pbar.set_description(
                    f"Validation - Loss: {loss.item():.4f}, Acc: {100 * val_correct/val_total:.2f}%"
                )
        
        # print epoch results
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Training Loss: {total_train_loss/len(train_loader):.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {total_val_loss/len(val_loader):.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        # save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model!")

def predict_text(texts, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        outputs = model(**encodings)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return predictions.cpu().numpy()

def main():
    pd.set_option('display.max_colwidth', None)
    df = pd.read_pickle("./bert_corpus.pickle")

    device = torch.device('cuda') #CHANGE THIS LINE from 'cuda' to 'cpu' if you are not running on a gpu!

    # load model and tokenizer
    model_name = "pranaydeeps/Ancient-Greek-BERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ) .to(device)

    # load saved model if available
    model_path = 'best_model.pt' 
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No saved model found. Training from scratch.")

        # prepare data
        train_df, val_df = prepare_data(df)
        
        # create dataloaders
        train_loader = create_dataloaders(
            train_df['text'],
            (train_df['label'] == 'poetry').astype(int).values,
            tokenizer
        )
        
        val_loader = create_dataloaders(
            val_df['text'],
            (val_df['label'] == 'poetry').astype(int).values,
            tokenizer
        )

        train_model(model, train_loader, val_loader, device)

    #sappho
    new_texts = ["ποικιλόθρον᾿ ἀθανάτ᾿ Αφρόδιτα,","παῖ Δίος δολόπλοκε, λίσσομαί σε,","μή μ᾿ ἄσαισι μηδ᾿ ὀνίαισι δάμνα,","πότνια, θῦμον,"]

    predictions = predict_text(new_texts, model, tokenizer, device)
    
    for text, pred in zip(new_texts, predictions):
        prose_prob, poetry_prob = pred
        predicted_class = "Poetry" if poetry_prob > 0.5 else "Prose"
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {predicted_class}")
        print(f"Poetry probability: {poetry_prob:.2%}")
        print(f"Prose probability: {prose_prob:.2%}\n")

main()
