from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F
import torch

METRICS = {
    'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
    'euclidean': lambda gallery, query: euclidean_dist(query, gallery),
    'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
    'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
}

# def draw_confusion(label_y, pre_y, path):
#     confusion = confusion_matrix(label_y, pre_y)
#     print(confusion)
    
def draw_confusion(label_y, pre_y):
    cm = confusion_matrix(label_y, pre_y)
    # Calculate the confusion matrix

    # Print False Negatives for each class
    print("False Negatives for each class:")
    for i, label in enumerate(set(label_y)):
        false_negatives = sum(cm[i, :]) - cm[i, i]
        true_positives = cm[i, i]
        if (false_negatives + true_positives) > 0:
            fnr = false_negatives / (false_negatives + true_positives)
        else:
            fnr = 0.0
        print(f"Class {label}: {fnr:.4f}")
    print(cm)

def write_result(fin, label_y, pre_y):
    accuracy = accuracy_score(label_y, pre_y)
    precision = precision_score(label_y, pre_y)
    recall = recall_score(label_y, pre_y)
    f1 = f1_score(label_y, pre_y)
    print('  -- test result: ')
    print('    -- accuracy: ', accuracy)
    fin.write('    -- accuracy: ' + str(accuracy) + '\n')
    print('    -- recall: ', recall)
    fin.write('    -- recall: ' + str(recall) + '\n')
    print('    -- precision: ', precision)
    fin.write('    -- precision: ' + str(precision) + '\n')
    print('    -- f1 score: ', f1)
    fin.write('    -- f1 score: ' + str(f1) + '\n\n')
    report = classification_report(label_y, pre_y)
    fin.write(report)
    fin.write('\n\n')
    return f1, accuracy

def cal_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def print_model_info_pytorch(model):
    """
    Prints all the model parameters, both trainable and non-trainable, and calculates the model size.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        else:
            non_trainable_params += param_size

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    # Calculate model size (assuming 32-bit floats for parameters)
    model_size = total_params * 4 / (1024 ** 2)  # Size in MB (4 bytes per float)
    print(f"Model size: {model_size:.2f} MB")
    
def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def classify_feats(prototypes, classes, feats, targets, metric='euclidean', sigma=1.0):
    # Classify new examples with prototypes and return classification error
    # dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
    # Calculate distances from query embeddings to prototypes
    # dist = torch.cdist(feats, prototypes)
    # dist = euclidean_dist(feats, prototypes)

    dist = METRICS[metric](prototypes, feats)
    preds = F.log_softmax(-dist, dim=1)
    labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)

    with torch.no_grad():
        acc = (preds.argmax(dim=1) == labels).float().mean()
        f1 = f1_score(labels.cpu().numpy(), preds.argmax(dim=1).cpu().numpy(), average='weighted')
    return preds, labels, acc, f1