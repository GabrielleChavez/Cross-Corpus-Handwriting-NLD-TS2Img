from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import io
import re

random.seed(42)
np.random.seed(42)

def group_data(dataset, file_name):
    if (dataset == "Parkinson_Drawings"):
        group_id = "PK_" + file_name[3:-4]
    elif (dataset == "HandPD"):
        group_id = "HP_" + "".join(file_name.split("-")[0])
    elif (dataset == "NewHandPD"):
        group_id = "NHP_" + "".join(file_name.split("-")[-1])
    else:
        print("Invalid dataset name. Only valid datasets are: ")
        print("\tParkinson_Drawings")
        print("\tHandPD")
        print("\tNewHandPD")

        return None

    return group_id

def loadImages(d, spiralType, dataset, directory = "Data/"):
    # ResNet50 expects ImageNet normalization

    count = 1
    data = {} # dictionary key = patient and value = image

    # directory = "/projects/NLS_ADPIE/data/" + d
    # directory = "Data/" + d
    directory += d
    for photo in os.listdir(directory):
        filepath = os.path.join(directory, photo)
        group_id = group_data(dataset, photo)
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = setLuminosity(img)
            img = img.resize((224, 224))
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            data[f'{spiralType}_{count}'] = {"image": img, "label": spiralType, "groupID": group_id}
            count += 1

    return data

def setLuminosity(img, target_lum=275):
    """
    Adjusts image so its mean luminosity matches target_lum.
    
    Parameters:
        image_path (str): Path to the input image.
        target_lum (float): Target average luminosity (0–255).
        save_path (str, optional): If given, saves adjusted image here.
    
    Returns:
        PIL.Image: Adjusted image.
    """
    # Open image and convert to RGB
    arr = np.array(img, dtype=np.float32)

    # Convert to grayscale (luminosity = weighted sum of RGB)
    # Rec. 709 luma coefficients
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    current_lum = np.mean(lum)

    # Compute scaling factor
    scale = target_lum / current_lum if current_lum > 0 else 1.0

    # Apply scaling to RGB channels
    adjusted = arr * scale
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(adjusted, mode="RGB")

    return out_img

def gatherData(dataset):
    typesOfData = {
        "Park_Healthy" : "Parkinson_Drawings/drawings/spiral/training/healthy/",
        "Park_Park": "Parkinson_Drawings/drawings/spiral/training/parkinson/",
        "Hand_Park" : "HandPD/Spiral_HandPD/SpiralPatients/",
        "Hand_Healthy" :  "HandPD/Spiral_HandPD/SpiralControl/",
        "New_Healthy" : "NewHandPD/HealthySpiral/",
        "New_Park" :"NewHandPD/PatientSpiral/",
        }
    d = "/projects/NLS_ADPIE/data/"
    if (dataset == "Parkinson_Drawings"):
        return loadImages(typesOfData["Park_Healthy"], "CTL", dataset, d) | loadImages(typesOfData["Park_Park"], "PD", dataset, d)
    elif (dataset == "HandPD"):
        return loadImages(typesOfData["Hand_Healthy"], "CTL",dataset, d) | loadImages(typesOfData["Hand_Park"],"PD", dataset, d)
    elif (dataset == "NewHandPD"):
        return loadImages(typesOfData["New_Healthy"], "CTL",dataset, d) | loadImages(typesOfData["New_Park"], "PD", dataset, d)
    elif (dataset == "PaHaW"):
        return getPaHaW(d)
    elif (dataset == "spirals"):
        return gatherDataNLS("spirals")
    else:
        print("Invalid dataset name. Only valid datasets are: ")
        print("\tParkinson_Drawings")
        print("\tHandPD")
        print("\tNewHandPD")
        print("\tPaHaW")
        return None
    
def gatherDataNLS(task):
    if (task == "points"):
        return getNLS(r"(point_DOM|point_NONDOM|point_sustained)")
    elif (task == "spirals"):
        return getNLS(r"(spiral_DOM|spiral_NONDOM|spiral_pataka)")
    elif (task == "numbers"):
        return getNLS(r"(numbers)")
    elif (task == "writing"):
        return getNLS(r"(copytext|copyreadtext|freewrite)")
    elif (task == "drawing"):
        return getNLS(r"(drawclock|copycube|copymage)")
    elif (task == "all"):
        keywords = r"point_DOM|point_NONDOM|point_sustained|spiral_DOM|spiral_NONDOM|spiral_pataka|numbers|"
        keywords += r"copytext|copyreadtext|freewrite|drawclock|copycube|copymage"
        return getNLS(keywords)
    elif (task == "all_pahaw"):
        keywords = r"point_DOM|point_NONDOM|point_sustained|spiral_DOM|spiral_NONDOM|spiral_pataka|numbers|"
        keywords += r"copytext|copyreadtext|freewrite|drawclock|copycube|copymage"
        new_dict = getNLS(keywords) | getPaHaW("/projects/NLS_ADPIE/data/")
        return new_dict
    else:
        return getNLS(task)

def xy2img(df, hasBS=False, isPoint=False):
    """
    Function to convert x,y coordinates to an image
    Parameters
    ----------
    df : pandas dataframe
        Dataframe with x,y coordinates
    hasBS : bool
        Boolean to determine if the data has a button press
    Returns
    -------
    img : PIL image
        Image with x,y coordinates
    """
    # Extract X, Y, and Pressure values
    X = df['X'].astype(float).values
    Y = df['Y'].astype(float).values
    P = df['P'].astype(float).values

    # Normalize Pressure values to [0, 1]
    P_min, P_max = np.min(P), np.max(P)
    P_s = ((P - P_min) / (P_max - P_min)) ** 0.5 if (P_max > P_min) else np.ones_like(P) 
    #P_s = 0.5 + 6*(P/P_max)**2 if (P_max > P_min) else np.ones_like(P)
    #P_a = ((P - P_min) / (P_max - P_min)) if (P_max > P_min) else np.ones_like(P)

    # If hasBS is True, filter points where BS == 1
    if hasBS and 'BS' in df.columns:
        S = df['BS'].astype(bool).values
        X, Y, P_s = X[S], Y[S], P_s[S]
        del S

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)  # High-resolution image
    ax.axis("off")  # Hide axis
    
    if isPoint:
        mask = P != 0
        C = ['black' if m else 'blue' for m in mask]
        ax.set_xlim(12000,18000)
        ax.set_ylim(6000,10000)
        ax.scatter(X, Y, c=C, alpha=0.5, s=1)
        del mask, C
    else:
        ax.set_xlim(np.min(X) - 5, np.max(X) + 5)  # Add some padding
        ax.set_ylim(np.min(Y) - 5, np.max(Y) + 5)
         # Scatter plot with transparency based on P
        # ax.scatter(X, Y, c='black', alpha=P, s=1)
        ax.scatter(X, Y, c='black', alpha=0.1, s=P_s)


    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    
    # Free Matplotlib figure memory
    plt.close(fig)  
    del fig, ax  # Delete figure objects explicitly

    # Convert buffer to PIL image
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    
    # Free buffer memory
    buf.close()
    del buf  

    # Free NumPy arrays
    del X, Y, P_s  

    return img

def getPaHaW(data_name= "/../../projects/NLS_ADPIE/data/"):
    """
    Function to load PaHaW data
    Parameters
    ----------
    data_name : str
        Path to the data
    Returns
    -------
    pahaw_data : dict
        Dictionary with keys as image names and values as images
    """
    # First Gather Disease info for each patient
    file_path = "/projects/NLS_ADPIE/data/PaHaW/PaHaW_files/corpus_PaHaW.xlsx"
    df = pd.read_excel(file_path)
    df['Disease'] = df['Disease'].replace("H", "CTL")
    
    ID_TYPE = pd.Series(df['Disease'].values, index=df['ID']).to_dict()

    # Second Iterate through folders to get svc files
    directory = "/projects/NLS_ADPIE/data/PaHaW/PaHaW_public/"
    pahaw_data = {}
    column_names =['Y', 'X', 'T', 'BS', 'Az', 'Al', 'P']
    countP = 1
    countC = 1
    for folder in sorted(os.listdir(directory)): # Iterate through folders
        folder_path = os.path.join(directory, folder)
        name = ID_TYPE[int(folder)] + '_' + str(int(folder))
        spiralType = findLabel(name)

        for svc_file in sorted(os.listdir(folder_path)):
            group_id = "PW_" + "".join(svc_file.split("_")[0])
            if spiralType=="PD":
                svc_paht = os.path.join(folder_path, svc_file)
                df = pd.read_csv(svc_paht, sep=' ', skiprows=1, header=None, names=column_names)
                img = xy2img(df,hasBS=True)
                pahaw_data[f'{spiralType}__{countP}'] = {"image": img, "label": spiralType, "groupID" : group_id}
                countP+=1
            elif spiralType=="CTL":
                svc_paht = os.path.join(folder_path, svc_file)
                df = pd.read_csv(svc_paht, sep=' ', skiprows=1, header=None, names=column_names)
                img = xy2img(df,hasBS=True)
                pahaw_data[f'{spiralType}__{countC}'] = {"image": img, "label": spiralType, "groupID" : group_id}
                countC+=1
    return pahaw_data

def getNLS(_labels):
    """
    Function to load NLS data
    Parameters
    ----------
    data_name : str
        Path to the data
    Returns
    -------
    nls_data : dict
        Dictionary with keys as image names and values as images
    """
    # First Gather Disease info for each patient
    # First Gather Disease info for each patient
    file_path = "/projects/NLS_ADPIE/data/NLS/handwriting/clean/0.metadata.csv"
    df = pd.read_csv(file_path)
    # Goal is to only extract PD and control
    df = df[(df['label'] == 'CTL') | (df['label'] == 'PD')]
    # Create Dictionary
    ID_TYPE = pd.Series(df['label'].values, index=df['ID']).to_dict()

    # Second Iterate through folders to get svc files
    directory = "/projects/NLS_ADPIE/data/NLS/handwriting/clean/"
    nls_data = {}
    valid_labels = r"CTL|PD"
    countP = 1
    countC = 1
    isPoint = "point" in _labels
    for folder in sorted(os.listdir(directory)): # Iterate through folders
        if folder in ID_TYPE.keys() and (ID_TYPE[folder] in valid_labels):
            folder_path = os.path.join(directory, folder)
            taskType = ID_TYPE[folder] # Determines the disease type (AD, PD, PDM, CTL, etc.) of each fie
            for svc_file in sorted(os.listdir(folder_path)): 
                if re.search(_labels, svc_file) and taskType=='PD': # True if label is PD
                    svc_paht = os.path.join(folder_path, svc_file)
                    df = pd.read_csv(svc_paht)
                    img = xy2img(df, isPoint=isPoint)
                    groupID = "NLS" + "_".join(svc_file.split("_")[:2])
                    nls_data[f'{taskType}_{countP}'] = {"image": img, "label": "PD", "groupID": groupID}
                    countP += 1 
                elif re.search(_labels, svc_file) and taskType=='CTL': # True if label is CTL
                    svc_paht = os.path.join(folder_path, svc_file)
                    df = pd.read_csv(svc_paht)
                    img = xy2img(df, isPoint=isPoint)
                    groupID = "NLS" + "_".join(svc_file.split("_")[:2])
                    nls_data[f'{taskType}_{countC}'] = {"image": img, "label": "CTL", "groupID": groupID}
                    countC += 1 
    return nls_data

def findLabel(filename):
    if ("PD" in filename):
        return "PD"
    elif("CTL" in filename):
        return "CTL"
    else:
        print("Error in Finding Label: ", filename)
        os.exit()
        
def partitionData(data, doSplit=False, test_size=0.2):
    X = [d["image"] for d in data.values()]
    y = [d["label"] for d in data.values()]
    map_labels = lambda labels: np.array([{"CTL": 0, "PD": 1}[label] for label in labels])
    y =  map_labels(y)

    groups = [d["groupID"] for d in data.values()]

    if doSplit:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        d_train = {"X": X_train, "y": map_labels(y_train)}
        d_test = {"X": X_test, "y": map_labels(y_test)}
        del map_labels
        return d_train, d_test
    else:
        return {"X": X, "y": y, "groups": groups}


if __name__ == "__main__": 
    pass  
 
