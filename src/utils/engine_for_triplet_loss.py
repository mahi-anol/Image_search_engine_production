from tqdm import tqdm
import torch
import os
from src.logger import logging
from src.utils.model_artifacts import saving_model_with_state_and_logs
### For changing model config or transferlearn/finetune config ....Need to go to model_builder.py



def train_step(model,data_loader,loss_fn,optimizer,device):
    model.train()
    train_loss,train_accuracy=0.0,0.0
    bar=tqdm(data_loader,desc='Training Epoch going on',leave=False)
    for batch,(anchor_sample,anchor_sample_label,positive_sample,positive_sample_label,negative_sample,negative_sample_label) in enumerate(bar):
        anchor_sample,anchor_sample_label,positive_sample,positive_sample_label,negative_sample,negative_sample_label=anchor_sample.to(device),anchor_sample_label.to(device),positive_sample.to(device),positive_sample_label.to(device),negative_sample.to(device),negative_sample_label.to(device)
        anchor_sample_embedding=model(anchor_sample)[0]
        positive_sample_embedding=model(positive_sample)[0]
        negative_sample_embedding=model(negative_sample)[0]
        batch_loss=loss_fn(anchor_sample_embedding,positive_sample_embedding,negative_sample_embedding)
        train_loss+=batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        bar.set_postfix(batch_loss=f'{batch_loss}')
    train_loss=train_loss/len(data_loader)
    return train_loss

def test_step(model,data_loader,loss_fn,device):
    model.eval()
    test_loss, test_accuracy=0.0,0.0
    with torch.inference_mode():
        bar=tqdm(data_loader,desc='Testing Epoch going on',leave=False)
        for batch,(anchor_sample,anchor_sample_label,positive_sample,positive_sample_label,negative_sample,negative_sample_label) in enumerate(bar):
            anchor_sample,anchor_sample_label,positive_sample,positive_sample_label,negative_sample,negative_sample_label=anchor_sample.to(device),anchor_sample_label.to(device),positive_sample.to(device),positive_sample_label.to(device),negative_sample.to(device),negative_sample_label.to(device)
            anchor_sample_embedding=model(anchor_sample)[0]
            positive_sample_embedding=model(positive_sample)[0]
            negative_sample_embedding=model(negative_sample)[0]
            batch_loss=loss_fn(anchor_sample_embedding,positive_sample_embedding,negative_sample_embedding)
            test_loss+=batch_loss.item()
            bar.set_postfix(batch_loss=f'{batch_loss}')

        test_loss=test_loss/len(data_loader)
    return test_loss

def train(model,train_dataloader,test_dataloader,optimizer,loss_fn,epochs,device,checkpoint_saving_gap):

    best_test_accuracy = -float('inf') 
    best_test_loss=float('inf')
    ### storing logs
    results={
        "train_loss":[],
        "test_loss":[],
    }
    for epoch in range(epochs):
        train_loss=train_step(model,train_dataloader,loss_fn,optimizer,device)
        test_loss=test_step(model,test_dataloader,loss_fn,device)
        logging.info(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f}')
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)

        if (epoch + 1) % checkpoint_saving_gap == 0:
            # It's good practice to reflect the loss type in the checkpoint name if it differs,
            # but based on the overall script, this engine is specifically for cross_entropy.
            saving_model_with_state_and_logs(model, optimizer, results, f"{epoch+1}_crossentropy_loss_trained_model.pt")
            logging.info(f"Saved epoch checkpoint at epoch {epoch+1}")

        # Save best model based on test accuracy
        if test_loss > best_test_loss:
            logging.info("weights from current epoch outperformed previous weights. ")
            best_test_loss = test_loss
            logging.info(f"Saving best model with Test loss: {best_test_loss:.4f} at epoch {epoch+1} @ ./checkpoint")
            # When saving 'best.pt', ensure 'results' reflects the metrics *up to that point*.
            # A shallow copy is usually sufficient if saving_model_with_state_and_logs doesn't modify it.
            # Using slice [:] creates a shallow copy of the lists within results.
            current_results_for_best = {k: v[:] for k, v in results.items()} 
            saving_model_with_state_and_logs(model, optimizer, current_results_for_best, "triplet_best.pt")

    # After the training loop finishes, save the last model
    logging.info("Saving last trained model @ ./models")
    saving_model_with_state_and_logs(model, optimizer, results, "triplet_last.pt")
    return results