from tqdm import tqdm
import torch
import os 
from src.logger import logging
from src.utils.model_artifacts import saving_model_with_state_and_logs


def train_step(model,data_loader,loss_fn,optimizer,device):
    model.train()
    train_loss=0.0
    bar=tqdm(data_loader,desc='Training Epoch going on',leave=False)
    for batch,(sample1,label1,sample2,label2,pos_or_neg) in enumerate(bar):
        sample1,label1,sample2,label2,pos_or_neg=sample1.to(device),label1.to(device),sample2.to(device),label2.to(device),pos_or_neg.to(device)
        embedding_of_first_sample=model(sample1)[0]
        embedding_of_second_sample=model(sample2)[0]
        batch_loss=loss_fn(embedding_of_first_sample,embedding_of_second_sample,pos_or_neg)
        train_loss+=batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        bar.set_postfix(batch_loss=f'{batch_loss}')

    train_loss=train_loss/len(data_loader)
    return train_loss

def test_step(model,data_loader,loss_fn,device):
    model.eval()
    test_loss=0.0
    with torch.inference_mode():
        bar=tqdm(data_loader,desc='Testing Epoch going on',leave=False)
        for batch,(sample1,label1,sample2,label2,pos_or_neg) in enumerate(bar):
            sample1,label1,sample2,label2,pos_or_neg=sample1.to(device),label1.to(device),sample2.to(device),label2.to(device),pos_or_neg.to(device)
            embedding_of_first_sample=model(sample1)[0]
            embedding_of_second_sample=model(sample2)[0]
            batch_loss=loss_fn(embedding_of_first_sample,embedding_of_second_sample,pos_or_neg)
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
            saving_model_with_state_and_logs(model, optimizer, current_results_for_best, "contrastive_best.pt")

    # After the training loop finishes, save the last model
    logging.info("Saving last trained model @ ./models")
    saving_model_with_state_and_logs(model, optimizer, results, "contrastive_last.pt")
    return results