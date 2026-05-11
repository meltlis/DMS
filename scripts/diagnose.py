import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

def analyze():
    # 1. Load data
    with open("reports/raw_features.json") as f:
        video_data = json.load(f)
        
    import csv
    video_gt = {}
    with open("reports/accuracy_eval.csv") as f:
        for row in csv.DictReader(f):
            stem = row['video'].split('\\')[-1]
            try:
                video_gt[stem] = int(row['gt_sleepy'])
            except:
                pass
                
    X = []
    y = []
    video_names = []
    
    for vp, feats in video_data.items():
        if vp not in video_gt: continue
        
        # extract sequence level summary statistics!
        ears = [f["ear"] for f in feats]
        mars = [f["mar"] for f in feats]
        pitches = [f["pitch"] for f in feats]
        yaws = [f["yaw"] for f in feats]
        
        # basic stats
        feat_vector = [
            np.mean(ears), np.min(ears), np.percentile(ears, 10), np.var(ears),
            np.mean(mars), np.max(mars), np.percentile(mars, 90), np.var(mars),
            np.mean(pitches), np.var(pitches), np.min(pitches),
            np.mean(yaws), np.var(yaws)
        ]
        
        X.append(feat_vector)
        y.append(video_gt[vp])
        video_names.append(vp)
        
    X = np.array(X)
    y = np.array(y)
    
    # 2. Can anything separate them?
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    train_acc = accuracy_score(y, clf.predict(X))
    print(f"Logistic Regression Train Acc: {train_acc:.4f}")
    
    svc = SVC(kernel='linear')
    svc.fit(X, y)
    svc_acc = accuracy_score(y, svc.predict(X))
    print(f"SVC Linear Train Acc: {svc_acc:.4f}")
    
    svc_rbf = SVC(kernel='rbf')
    svc_rbf.fit(X, y)
    svc_acc_rbf = accuracy_score(y, svc_rbf.predict(X))
    print(f"SVC RBF Train Acc: {svc_acc_rbf:.4f}")

if __name__ == "__main__":
    analyze()