from transformers import AutoImageProcessor, ResNetForImageClassification, AutoFeatureExtractor
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch
import torch.nn as nn
import process_images
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from skopt import BayesSearchCV
# from skopt.space import Real
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

class ResNetWithMLP(nn.Module):
    def __init__(self, base_model_name="microsoft/resnet-50", mlp_dims=[512, 128, 2]):
        super().__init__()
        self.base_model = ResNetForImageClassification.from_pretrained(base_model_name)

        # Determine base output dim
        classifier = self.base_model.classifier
        if isinstance(classifier, nn.Sequential):
            for layer in reversed(classifier):
                if isinstance(layer, nn.Linear):
                    base_output_dim = layer.in_features  # input to classifier, not out_features
                    break
        elif isinstance(classifier, nn.Linear):
            base_output_dim = classifier.in_features
        else:
            raise ValueError(f"Unsupported classifier type: {type(classifier)}")

        # Replace classifier with Identity
        self.base_model.classifier = nn.Identity()

        # Build MLP
        layers = []
        in_dim = base_output_dim
        for dim in mlp_dims[:-1]:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        layers.append(nn.Linear(in_dim, mlp_dims[-1]))
        
        # Final Output lawyer
        #layers.append(nn.Softmax(dim=1))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        features = outputs.logits if hasattr(outputs, "logits") else outputs
        features = features.view(features.size(0), -1)
        logits = self.mlp_head(features)
        return logits

def evaluate_model(model, loader, device):
    """
    Run inference on a dataloader and return (logits, labels) as numpy arrays.
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for px, lbl in loader:
            px, lbl = px.to(device), lbl.to(device)
            out = model(px)
            logits_list.append(out.cpu())
            labels_list.append(lbl.cpu())

    logits_np = torch.cat(logits_list).numpy()
    labels_np = torch.cat(labels_list).numpy()
    return logits_np, labels_np

def train_one_fold(train_loader, val_loader, device, num_epochs, model, patience=10, lr=1e-4):
    """
    if use_MLP:
        model = ResNetWithMLP().to(device)
    else:
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc, best_val_loss = 0, float("inf")
    best_state, counter = None, 0

    for epoch in range(num_epochs):
        # ── Training loop ──
        model.train()
        running_loss = 0.0
        for px, lbl in train_loader:
            px, lbl = px.to(device), lbl.to(device)
            out = model(px)

            # Handle HuggingFace vs custom model outputs
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validation loop ──
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for px, lbl in val_loader:
                px, lbl = px.to(device), lbl.to(device)
                out = model(px)
                logits = out.logits if hasattr(out, "logits") else out

                loss = criterion(logits, lbl)
                val_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

        val_acc = correct / total
        print(f"E{epoch+1:02d}  train-loss {running_loss:.4f}  val-loss {val_loss:.4f}  val-acc {val_acc:.4f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop on epoch {epoch+1}")
                break
    model.load_state_dict(best_state)  # Reload best state
    
    return model #, best_val_acc, best_val_loss

def get_feature_extractor(model):
    # take everything except the last FC layer
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    return feature_extractor

def extract_features(model, loader, device):
    model.eval()
    feats_list, labels_list = [], []

    with torch.no_grad():
        for px, lbl in loader:
            px, lbl = px.to(device), lbl.to(device)
            out = model(px)

            # last_hidden_state is [B, C, H, W] for HuggingFace ResNet
            feats = out.last_hidden_state   # [B, C, H, W]

            # Apply global average pooling → [B, C]
            feats = feats.mean(dim=[2, 3])

            feats_list.append(feats.cpu())
            labels_list.append(lbl.cpu())

    feats_np = torch.cat(feats_list).numpy()   # [N, C]
    labels_np = torch.cat(labels_list).numpy() # [N]

    return feats_np, labels_np



def train_one_fold_svm(train_loader, val_loader, device, num_epochs, model, patience=10, lr=1e-4):
    """
    Train a torchvision ResNet model with early stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc, best_val_loss = 0, float("inf")
    best_state, counter = None, 0

    for epoch in range(num_epochs):
        # ── Training loop ──
        model.train()
        running_loss = 0.0
        for px, lbl in train_loader:
            px, lbl = px.to(device), lbl.to(device)
            out = model(px)              # torchvision → logits directly
            loss = criterion(out, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validation loop ──
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for px, lbl in val_loader:
                px, lbl = px.to(device), lbl.to(device)
                out = model(px)          # logits
                loss = criterion(out, lbl)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)

        val_acc = correct / total
        print(f"E{epoch+1:02d}  train-loss {running_loss:.4f}  val-loss {val_loss:.4f}  val-acc {val_acc:.4f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stop on epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # Reload best state

    return model

def preprocess_images_svm(image_list, labels, image_size=224):
    """
    Process images into tensors using torchvision transforms.
    Converts to 3-channel tensors normalized to ImageNet stats.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensors = [transform(img) for img in image_list]
    images_tensor = torch.stack(tensors)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor


# ────────────────────────────────
# Extract logits (torchvision)
# ────────────────────────────────
def extract_logits_svm(dataloader, device, model):
    logits_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)                 # logits directly
            logits = outputs.cpu().numpy()

            logits_list.append(logits)
            labels_list.append(labels.cpu().numpy())

            del images, labels, outputs
            torch.cuda.empty_cache()

    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)

def testCNN(model, X_test, y_test):
    """
    Test the trained CNN on test data.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CNN/ResNet model
    processor : transformers.AutoImageProcessor
        Preprocessing pipeline
    test_data : dict
        Test image data {"X": [PIL images], "y": labels}

    Returns
    -------
    accuracy : float
        Classification accuracy on test set
    auc : float
        AUC score (only for binary classification)
    f1 : float
        F1 score (weighted average)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess test images
    #inputs_test = processor(X_test, return_tensors="pt")
    #pixel_values = inputs_test["pixel_values"].to(device)
    #y_test = torch.tensor(test_data["y"]).to(device)

    with torch.no_grad():
        outputs = model(pixel_values=X_test.to(device))
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = torch.argmax(logits, dim=1)

    # Move to CPU for metric calculations
    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_prob = torch.softmax(logits, dim=1).detach().cpu().numpy()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # F1 Score (weighted for multi-class)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # AUC Score
    try:
        if y_prob.shape[1] == 2:  # Binary classification
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:  # Multiclass
            print("multiclass error")
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        auc = float('nan')  # e.g. if only one class is present in y_true

    print(f"Test Accuracy: {accuracy:.3%}")
    print(f"Test F1 Score: {f1:.3f}")
    print(f"Test AUC Score: {auc:.3f}")

    return accuracy, auc, f1

def preprocess_images(processor, image_list, labels):
    """Process images in batches to reduce memory usage."""
    tensors = []
    for img in image_list:
        inputs = processor(img, return_tensors="pt")["pixel_values"]
        tensors.append(inputs.squeeze(0))  # Remove batch dim
    images_tensor = torch.stack(tensors) 

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return images_tensor, labels_tensor

def extract_logits(dataloader, device, model):
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits.cpu().numpy()

            logits_list.append(logits)
            labels_list.append(labels.cpu().numpy())

            del images, labels, outputs  # Free memory
            torch.cuda.empty_cache()

    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)

def fitSVM(x_train, y_train, _C=None, _gamma=None, bestParam=None, kernel='rbf', cv_folds=10):
    """
    Compute accuracy using SVM with cross-validation.

    Parameters
    ----------
    x_train : numpy array
        Training features
    x_test : numpy array
        Testing features
    y_train : numpy array
        Training labels
    y_test : numpy array
        Testing labels
    _C : float, optional
        SVM regularization parameter
    _gamma : float, optional
        Kernel coefficient for RBF kernel
    bestParam : dict, optional
        Dictionary of best parameters from optimization
    kernel : str, optional
        SVM kernel type
    cv_folds : int, optional
        Number of cross-validation folds (default: 5)

    Returns
    -------
    accuracy : float
        Mean accuracy across cross-validation folds
    """
    if bestParam is None:
        C = _C
        gamma = _gamma
    else: 
        C = bestParam[kernel][0] 
        gamma = bestParam[kernel][1]  
    
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    
    svm.fit(x_train, y_train)

    return svm

def testSVM(svm, x_test, y_test):
    y_pred = svm.predict(x_test)
    
    # Perform test
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_auc = roc_auc_score(y_test, svm.predict_proba(x_test)[:,1]) 

    return test_accuracy, test_auc, test_f1


def bayes_opti(X_train, y_train, kernel='rbf', cv=5, n_iter=30,
               scoring='accuracy', random_state=None, verbose=False, n_jobs=-1):
    """
    Perform Bayesian optimization for SVM hyperparameters using scikit-optimize.
    
    Args:
        X_train (array): Training features, shape (n_samples, n_features).
        y_train (array): Training labels, shape (n_samples,).
        kernel (str): Kernel type for SVM ('rbf', 'poly', 'linear', etc.).
        cv (int): Number of folds for cross-validation.
        n_iter (int): Number of iterations for Bayesian search.
        scoring (str): Metric for model evaluation.
        random_state (int): Random seed.
        verbose (bool): Print best results if True.
        n_jobs (int): Number of parallel jobs for CV.
    """
    param_space = {
        'C': Real(1e-2, 1e2, prior='log-uniform'),
        'gamma': Real(1e-3, 1.0, prior='log-uniform')
    }

    opt = BayesSearchCV(
        estimator=SVC(kernel=kernel, random_state=random_state),
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state
    )

    opt.fit(X_train, y_train)

    if verbose:
        print("Best SVM parameters:", opt.best_params_)
        print("Best CV score:", opt.best_score_)

    return opt.best_params_


if __name__=='__main__':
    pass
