import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns

def load_images(folder_path, label):
    """Charge les images"""
    images, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpeg', '.jpg', '.png')):
            try:
                img = Image.open(os.path.join(folder_path, filename)).convert('L')
                img = img.resize((64, 64))
                img_array = np.array(img) / 255.0
                images.append(img_array.flatten())
                labels.append(label)
            except:
                pass
    return np.array(images), np.array(labels)


# Charger les données
print("\n📥 Chargement des images...")
normal_imgs, normal_labels = load_images("val/NORMAL", 0)
pneumonia_imgs, pneumonia_labels = load_images("val/PNEUMONIA", 1)

X = np.vstack([normal_imgs, pneumonia_imgs])
y = np.concatenate([normal_labels, pneumonia_labels])

print(f"✅ {len(normal_imgs)} images NORMAL")
print(f"✅ {len(pneumonia_imgs)} images PNEUMONIA")
print(f"✅ Total: {len(X)} images de {64}x{64} pixels")

# Visualiser quelques exemples
print("\n📊 Création de visualisations...")
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(4):
    # NORMAL
    axes[0, i].imshow(normal_imgs[i].reshape(64, 64), cmap='gray')
    axes[0, i].set_title('NORMAL', color='green')
    axes[0, i].axis('off')
    
    # PNEUMONIA
    axes[1, i].imshow(pneumonia_imgs[i].reshape(64, 64), cmap='gray')
    axes[1, i].set_title('PNEUMONIA', color='red')
    axes[1, i].axis('off')

plt.suptitle('Exemples de radiographies', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('test_examples.png', dpi=200)
print("✅ Sauvegardé: test_examples.png")

# Split train/test sur val (8 images pour train, 8 pour test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print(f"\n🔄 Split: {len(X_train)} train, {len(X_test)} test")

# Normaliser
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner un modèle simple
print("\n🤖 Entraînement Random Forest...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

# Prédiction
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Métriques
acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"✅ Accuracy: {acc*100:.1f}%")
    print(f"✅ ROC-AUC: {auc:.3f}")
except:
    print(f"✅ Accuracy: {acc*100:.1f}%")
    print("⚠️  ROC-AUC: Impossible à calculer (trop peu de données)")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['NORMAL', 'PNEUMONIA'],
           yticklabels=['NORMAL', 'PNEUMONIA'])
plt.title('Matrice de Confusion (Test sur VAL)')
plt.savefig('test_confusion.png', dpi=200)
print("✅ Sauvegardé: test_confusion.png")

print("\n" + "="*60)
print("✅ TEST TERMINÉ AVEC SUCCÈS !")
print("="*60)
print("\n💡 PROCHAINES ÉTAPES:")
print("1. Si ça marche → Téléchargez train.zip et test.zip")
print("2. Décompressez-les dans le même dossier")
print("3. Utilisez zoidberg_simple.py avec toutes les données")
print("="*60)