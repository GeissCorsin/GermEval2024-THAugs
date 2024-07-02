import numpy as np
from sklearn.metrics import f1_score
import torch

def evaluate_model(model, val_loader, device, tasks=None):
    """
    Evaluates the given model on the validation loader for specified tasks.

    Parameters:
        model (torch.nn.Module): The model to evaluate.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device): Device to perform computations on.
        tasks (list, optional): List of task names to evaluate. Defaults to all tasks.

    Returns:
        tuple: F1 scores for each task, average F1 score, targets, predictions, probabilities.
    """
    if tasks is None:
        tasks = ["bin_maj", "bin_one", "bin_all", "multi_maj", "disagree_bin"]

    model.eval()
    preds = {task: [] for task in tasks}
    targets = {task: [] for task in tasks}
    probs = {task: [] for task in tasks}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            # Ensure logits is always a tuple
            if isinstance(logits, torch.Tensor):
                logits = (logits,)

            task_logits = {}
            task_preds = {}
            
            for i, task in enumerate(tasks):
                if task in ["bin_maj", "bin_one", "bin_all", "disagree_bin"]:
                    task_logits[task] = torch.sigmoid(logits[i])
                    task_preds[task] = task_logits[task].round()
                elif task == "multi_maj":
                    task_logits[task] = logits[i]
                    task_preds[task] = torch.argmax(task_logits[task], dim=1)
                else:
                    raise ValueError(f"Unsupported task: {task}")

                preds[task].extend(task_preds[task].cpu().numpy())
                targets[task].extend(labels[:, get_task_index(task)].cpu().numpy())
                probs[task].extend(task_logits[task].cpu().numpy())

    # Compute F1 macro scores for all tasks
    f1_scores = {task: f1_score(targets[task], preds[task], average='macro') for task in tasks}
    
    # Calculate the final score as the unweighted average of all F1 macro scores
    avg_f1 = np.mean(list(f1_scores.values()))

    return list(f1_scores.values()), avg_f1, targets, preds, probs

def get_task_index(task_name):
    """
    Returns the index of the task in the labels tensor.

    Parameters:
        task_name (str): The name of the task.

    Returns:
        int: The index corresponding to the task.
    """
    task_indices = {
        "bin_maj": 0,
        "bin_one": 1,
        "bin_all": 2,
        "multi_maj": 3,
        "disagree_bin": 4
    }
    return task_indices[task_name]