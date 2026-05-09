import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

#EDA

#Correlation Heatmap

df = pd.read_csv('heart_disease.csv')
num_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()

#Imbalance Bar Chart

plt.figure(figsize=(8, 5))
c_count = df['num'].value_counts().sort_index()
sns.barplot(x=c_count.index, y=c_count.values, hue=c_count.index, palette='viridis', legend=False)
plt.title('Class Imbalance')
plt.xlabel('Severity Class')
plt.ylabel('Patients')
plt.savefig('class_imbalance.png', bbox_inches='tight')
plt.close()

#Pre Processing

df = df.drop(columns=['id'])
for i in df.select_dtypes(include=['number']).columns:
    df[i] = df[i].fillna(df[i].median())
for i in df.select_dtypes(include=['object']).columns:
    df[i] = df[i].fillna(df[i].mode()[0])
x = pd.get_dummies(df.drop('num', axis=1), drop_first=True)
y = df['num']

#Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)

# Model Training

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}
accuracies = {}
with open('model_reports.txt', 'w') as f:
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train_scale, y_train)
        y_pred = model.predict(x_test_scale)
        y_prob = model.predict_proba(x_test_scale)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        f.write(f"========== {name} ==========\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n")
        f.write(f"AUC Score: {auc:.4f}\n")
        f.write("Classification Report (Precision/Recall):\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))
        f.write("\n\n")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'CM_{name.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), hue=list(accuracies.keys()), palette='magma', legend=False)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontweight='bold')
plt.savefig('accuracy.png', bbox_inches='tight')
plt.close()

# ROC Curves

plt.figure(figsize=(8, 6))
for name, model in models.items():
    # Binarize the target to easily graph "No Disease (0)" vs "Any Disease (1-4)"
    y_test_bin = (y_test > 0).astype(int)
    y_train_bin = (y_train > 0).astype(int)
    
    # Fit strictly for the ROC plot
    model.fit(x_train_scale, y_train_bin)
    y_prob_positive = model.predict_proba(x_test_scale)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_prob_positive)
    plt.plot(fpr, tpr, label=f'{name}')

plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guessing')
plt.title('ROC Curve (No Disease vs Any Disease)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc.png', bbox_inches='tight')
plt.close()

#K Means
x_scale = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scale)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scale)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1', legend='full')
plt.title('K Means Visualization')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig('kmeans.png', bbox_inches='tight')
plt.close()