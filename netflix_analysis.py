# -*- coding: utf-8 -*-
# =============================================================================
# Netflix Data Science Project
# Pengantar Sains Data — Universitas Brawijaya
# M. Alhafiz Arya Wardhana (245150407111038)
# Ananda Tiara Pramitha (245150401111013)
# =============================================================================

# %% [markdown]
# # Analisis Data Netflix Movies & TV Shows
# **Mata Kuliah:** Pengantar Sains Data  
# **Nama:** M. Alhafiz Arya Wardhana (245150407111038) & Ananda Tiara Pramitha (245150401111013)  
# **Universitas Brawijaya — Sistem Informasi**
# 
# Project ini mencakup full pipeline data science:
# 1. Import & Load Data
# 2. Exploratory Data Analysis (EDA)
# 3. Preprocessing & Feature Engineering
# 4. Supervised Learning — Classification (Movie vs TV Show)
# 5. Unsupervised Learning — Clustering Konten
# 6. Kesimpulan & Insight

# %% [markdown]
# ---
# ## TAHAP 1: Import Library & Load Data

# %%
# Import semua library yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Konfigurasi tampilan
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_palette('husl')

# Buat folder output jika belum ada
os.makedirs('output', exist_ok=True)

print("[OK] Semua library berhasil di-import!")

# %%
# Load dataset
df = pd.read_csv('netflix_titles.csv')

print(f"[DATA] Dataset berhasil dimuat!")
print(f"   Jumlah baris : {df.shape[0]}")
print(f"   Jumlah kolom : {df.shape[1]}")

# %%
# Informasi dasar dataset
print("=" * 60)
print("INFO DATASET")
print("=" * 60)
df.info()

# %%
# Tampilkan 5 baris pertama
print("\n[LIST] 5 Baris Pertama Dataset:")
df.head()

# %%
# Statistik deskriptif
print("\n[CHART] Statistik Deskriptif:")
df.describe(include='all')

# %%
# Cek missing values
print("\n[?] Missing Values per Kolom:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Jumlah': missing, 'Persentase (%)': missing_pct})
missing_df = missing_df[missing_df['Jumlah'] > 0].sort_values('Jumlah', ascending=False)
print(missing_df)

# %%
# Cek duplikat
print(f"\n[CYCLE] Jumlah data duplikat: {df.duplicated().sum()}")

# %% [markdown]
# ---
# ## TAHAP 2: Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 2.1 Distribusi Tipe Konten (Movie vs TV Show)

# %%
fig, ax = plt.subplots(figsize=(8, 8))
type_counts = df['type'].value_counts()
colors = ['#3B5998', '#2ECC71']
explode = (0.03, 0.03)

wedges, texts, autotexts = ax.pie(
    type_counts.values, 
    labels=type_counts.index,
    autopct='%1.1f%%', 
    colors=colors, 
    explode=explode,
    startangle=90,
    textprops={'fontsize': 14, 'fontweight': 'bold'}
)
for t in autotexts:
    t.set_color('white')
    t.set_fontsize(16)

ax.set_title('Distribusi Tipe Konten Netflix', fontsize=18, fontweight='bold', pad=20)

# Tambahkan jumlah absolut di legend
legend_labels = [f'{idx}: {val:,}' for idx, val in zip(type_counts.index, type_counts.values)]
ax.legend(legend_labels, loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('output/distribusi_tipe_konten.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Movie: {type_counts.get('Movie', 0):,} | TV Show: {type_counts.get('TV Show', 0):,}")

# %% [markdown]
# ### 2.2 Distribusi Rating

# %%
fig, ax = plt.subplots(figsize=(12, 6))
rating_counts = df['rating'].value_counts().head(10)

bars = ax.bar(rating_counts.index, rating_counts.values, color=sns.color_palette('Reds_r', len(rating_counts)))
ax.set_title('Distribusi Rating Konten Netflix (Top 10)', fontsize=16, fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Jumlah Konten')

# Tambahkan label di atas bar
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/distribusi_rating.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.3 Top 10 Negara Produksi

# %%
# Pecah negara (beberapa konten punya multi-country)
country_data = df['country'].dropna().str.split(',').explode().str.strip()
country_counts = country_data.value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(country_counts.index[::-1], country_counts.values[::-1], 
               color=sns.color_palette('viridis', len(country_counts)))

ax.set_title('Top 10 Negara Produksi Konten Netflix', fontsize=16, fontweight='bold')
ax.set_xlabel('Jumlah Konten')

for bar in bars:
    width = bar.get_width()
    ax.text(width + 10, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('output/top_negara.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.4 Top 10 Genre Paling Populer

# %%
# Pecah genre (listed_in bisa berisi multi-genre)
genre_data = df['listed_in'].dropna().str.split(',').explode().str.strip()
genre_counts = genre_data.value_counts().head(10)

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], 
               color=sns.color_palette('coolwarm', len(genre_counts)))

ax.set_title('Top 10 Genre Paling Populer di Netflix', fontsize=16, fontweight='bold')
ax.set_xlabel('Jumlah Konten')

for bar in bars:
    width = bar.get_width()
    ax.text(width + 10, bar.get_y() + bar.get_height()/2.,
            f'{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('output/top_genre.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.5 Tren Produksi Konten per Tahun

# %%
# Tren berdasarkan release_year (2008 - 2021 agar relevan)
trend_data = df[df['release_year'] >= 2008].groupby(['release_year', 'type']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 7))
trend_data.plot(kind='bar', stacked=True, ax=ax, color=['#3B5998', '#2ECC71'], width=0.8)

ax.set_title('Tren Produksi Konten Netflix per Tahun', fontsize=16, fontweight='bold')
ax.set_xlabel('Tahun Rilis')
ax.set_ylabel('Jumlah Konten')
ax.legend(title='Tipe', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/tren_produksi.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2.6 Distribusi Durasi Film (Movie only)

# %%
# Filter hanya Movie dan ekstrak durasi numerik
movies = df[df['type'] == 'Movie'].copy()
movies['duration_min'] = movies['duration'].str.replace(' min', '').astype(float)

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(movies['duration_min'].dropna(), bins=30, color='#E74C3C', edgecolor='white', alpha=0.85)
ax.axvline(movies['duration_min'].median(), color='black', linestyle='--', linewidth=2, 
           label=f"Median: {movies['duration_min'].median():.0f} menit")

ax.set_title('Distribusi Durasi Film di Netflix', fontsize=16, fontweight='bold')
ax.set_xlabel('Durasi (menit)')
ax.set_ylabel('Frekuensi')
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('output/distribusi_durasi.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"[DATA] Statistik Durasi Film:")
print(f"   Mean   : {movies['duration_min'].mean():.1f} menit")
print(f"   Median : {movies['duration_min'].median():.1f} menit")
print(f"   Min    : {movies['duration_min'].min():.0f} menit")
print(f"   Max    : {movies['duration_min'].max():.0f} menit")

# %% [markdown]
# ### 2.7 Konten Ditambahkan per Tahun (date_added)

# %%
df_dated = df.dropna(subset=['date_added']).copy()
df_dated['date_added'] = pd.to_datetime(df_dated['date_added'].str.strip())
df_dated['year_added'] = df_dated['date_added'].dt.year

yearly_added = df_dated.groupby('year_added').size()
yearly_added = yearly_added[yearly_added.index >= 2015]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(yearly_added.index, yearly_added.values, marker='o', linewidth=2.5, 
        color='#E74C3C', markersize=8)

for x, y in zip(yearly_added.index, yearly_added.values):
    ax.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                xytext=(0, 12), ha='center', fontweight='bold', fontsize=11)

ax.set_title('Konten Ditambahkan ke Netflix per Tahun', fontsize=16, fontweight='bold')
ax.set_xlabel('Tahun')
ax.set_ylabel('Jumlah Konten')

plt.tight_layout()
plt.savefig('output/konten_per_tahun.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ---
# ## TAHAP 3: Data Preprocessing & Feature Engineering

# %% [markdown]
# ### 3.1 Handling Missing Values

# %%
print("=" * 60)
print("SEBELUM PREPROCESSING")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Isi missing value untuk kolom dengan banyak null
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

# Hapus baris dengan missing value kecil
df = df.dropna(subset=['date_added', 'rating', 'duration'])

print(f"\n{'=' * 60}")
print("SETELAH HANDLING MISSING VALUES")
print(f"{'=' * 60}")
print(f"Shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")

# %% [markdown]
# ### 3.2 Konversi Tipe Data

# %%
# Konversi date_added ke datetime
df['date_added'] = pd.to_datetime(df['date_added'].str.strip())

# Ekstrak tahun dan bulan ditambahkan
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Pisahkan duration jadi angka dan tipe
df['duration_int'] = df['duration'].str.extract(r'(\d+)').astype(int)
df['duration_type'] = df['duration'].str.extract(r'(\D+)').iloc[:, 0].str.strip()

print("[OK] Konversi tipe data selesai!")
print(f"\nContoh duration_int: {df['duration_int'].head(3).tolist()}")
print(f"Contoh duration_type: {df['duration_type'].unique()}")

# %% [markdown]
# ### 3.3 Feature Engineering

# %%
# Jumlah genre per konten
df['genre_count'] = df['listed_in'].str.split(',').apply(len)

# Apakah director diketahui
df['has_director'] = (df['director'] != 'Unknown').astype(int)

# Negara utama (negara pertama sebelum koma)
df['country_main'] = df['country'].str.split(',').str[0].str.strip()

# Panjang deskripsi
df['description_length'] = df['description'].str.len()

# Panjang judul
df['title_length'] = df['title'].str.len()

# Genre utama (genre pertama)
df['genre_main'] = df['listed_in'].str.split(',').str[0].str.strip()

print("[OK] Feature engineering selesai!")
print(f"\nFitur baru yang dibuat:")
print(f"  - genre_count       : rata-rata {df['genre_count'].mean():.1f} genre per konten")
print(f"  - has_director      : {df['has_director'].sum():,} konten punya director")
print(f"  - country_main      : {df['country_main'].nunique()} negara unik")
print(f"  - description_length: rata-rata {df['description_length'].mean():.0f} karakter")
print(f"  - title_length      : rata-rata {df['title_length'].mean():.0f} karakter")

# %% [markdown]
# ### 3.4 Encoding

# %%
# Label encode type
df['type_encoded'] = (df['type'] == 'Movie').astype(int)

# Label encode rating
le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'])

# One-hot encode negara utama (top 10 saja)
top10_countries = df['country_main'].value_counts().head(10).index.tolist()
df['country_main_grouped'] = df['country_main'].apply(lambda x: x if x in top10_countries else 'Other')
country_dummies = pd.get_dummies(df['country_main_grouped'], prefix='country')

# One-hot encode genre utama (top 10 saja)
top10_genres = df['genre_main'].value_counts().head(10).index.tolist()
df['genre_main_grouped'] = df['genre_main'].apply(lambda x: x if x in top10_genres else 'Other')
genre_dummies = pd.get_dummies(df['genre_main_grouped'], prefix='genre')

# Gabungkan ke dataframe
df = pd.concat([df, country_dummies, genre_dummies], axis=1)

print("[OK] Encoding selesai!")
print(f"   Kolom country one-hot: {len(country_dummies.columns)}")
print(f"   Kolom genre one-hot  : {len(genre_dummies.columns)}")

# %% [markdown]
# ### 3.5 Validasi Akhir

# %%
print("=" * 60)
print("VALIDASI AKHIR DATASET")
print("=" * 60)
print(f"Shape final         : {df.shape}")
print(f"Missing values total: {df.isnull().sum().sum()}")
print(f"Duplikat            : {df.duplicated().sum()}")
print(f"\nKolom dataset final ({len(df.columns)} kolom):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col} ({df[col].dtype})")

# %% [markdown]
# ---
# ## TAHAP 4: Supervised Learning — Classification (Movie vs TV Show)

# %% [markdown]
# ### 4.1 Persiapan Data untuk Classification

# %%
# Pilih fitur untuk classification
feature_cols = (['release_year', 'duration_int', 'genre_count', 
                 'description_length', 'title_length', 'has_director', 
                 'rating_encoded'] + 
                list(country_dummies.columns) + 
                list(genre_dummies.columns))

X = df[feature_cols]
y = df['type_encoded']  # 1 = Movie, 0 = TV Show

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Data siap untuk classification!")
print(f"   Training set : {X_train.shape[0]:,} sampel")
print(f"   Testing set  : {X_test.shape[0]:,} sampel")
print(f"   Jumlah fitur : {X_train.shape[1]}")
print(f"   Proporsi target (train):")
print(f"     Movie   (1): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
print(f"     TV Show (0): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")

# %% [markdown]
# ### 4.2 Training 3 Model Classification

# %%
# Definisikan 3 model
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=15),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")
    
    # Training
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': acc,
        'y_pred': y_pred,
        'report': classification_report(y_test, y_pred, target_names=['TV Show', 'Movie'])
    }
    
    print(f"\n[DATA] Accuracy: {acc*100:.2f}%")
    print(f"\nClassification Report:")
    print(results[name]['report'])

# %% [markdown]
# ### 4.3 Confusion Matrix Visualisasi

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['TV Show', 'Movie'],
                yticklabels=['TV Show', 'Movie'])
    axes[idx].set_title(f'{name}\nAccuracy: {res["accuracy"]*100:.2f}%', 
                        fontsize=13, fontweight='bold')
    axes[idx].set_ylabel('Aktual')
    axes[idx].set_xlabel('Prediksi')

plt.suptitle('Confusion Matrix — Perbandingan 3 Model', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/confusion_matrix_semua.png', dpi=150, bbox_inches='tight')
plt.show()

# Simpan juga per model
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['TV Show', 'Movie'],
                yticklabels=['TV Show', 'Movie'])
    ax.set_title(f'Confusion Matrix — {name}\nAccuracy: {res["accuracy"]*100:.2f}%', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Aktual')
    ax.set_xlabel('Prediksi')
    plt.tight_layout()
    
    filename = name.lower().replace(' ', '_')
    plt.savefig(f'output/confusion_matrix_{filename}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("[OK] Semua confusion matrix tersimpan di folder output/")

# %% [markdown]
# ### 4.4 Tabel Perbandingan Model

# %%
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy (%)': [f"{res['accuracy']*100:.2f}" for res in results.values()]
}).sort_values('Accuracy (%)', ascending=False)

print("\n[DATA] TABEL PERBANDINGAN MODEL CLASSIFICATION")
print("=" * 45)
print(comparison_df.to_string(index=False))

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n[TROPHY] Model terbaik: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")

# Visualisasi perbandingan
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] * 100 for m in model_names]
colors_bar = ['#2ECC71' if m == best_model_name else '#3B5998' for m in model_names]

bars = ax.bar(model_names, accuracies, color=colors_bar, edgecolor='white', linewidth=2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_title('Perbandingan Accuracy Model Classification', fontsize=16, fontweight='bold')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('output/perbandingan_model.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4.5 Feature Importance (Random Forest)

# %%
rf_model = results['Random Forest']['model']
importances = rf_model.feature_importances_
feat_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(feat_importance['Feature'][::-1], feat_importance['Importance'][::-1],
               color=sns.color_palette('RdYlGn_r', len(feat_importance)))

ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
ax.set_xlabel('Importance')

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
            f'{width:.4f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("[DATA] Top 5 Fitur Terpenting:")
for i, row in feat_importance.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

# %% [markdown]
# ---
# ## TAHAP 5: Unsupervised Learning — Clustering Konten

# %% [markdown]
# ### 5.1 Persiapan Data Clustering

# %%
# Pilih fitur numerik untuk clustering
cluster_features = ['release_year', 'duration_int', 'genre_count', 
                    'description_length', 'title_length']

X_cluster = df[cluster_features].copy()

# Normalisasi dengan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"[OK] Data clustering siap!")
print(f"   Jumlah sampel: {X_scaled.shape[0]:,}")
print(f"   Jumlah fitur : {X_scaled.shape[1]}")
print(f"\nStatistik fitur (setelah scaling):")
print(pd.DataFrame(X_scaled, columns=cluster_features).describe().round(2))

# %% [markdown]
# ### 5.2 Elbow Method & Silhouette Score

# %%
# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("[SEARCH] Menghitung Elbow Method & Silhouette Score...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    print(f"   k={k}: Inertia={kmeans.inertia_:,.0f}, Silhouette={sil_score:.4f}")

# Visualisasi Elbow Method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_title('Elbow Method', fontsize=16, fontweight='bold')
ax1.set_xlabel('Jumlah Cluster (k)')
ax1.set_ylabel('Inertia')
ax1.set_xticks(list(K_range))

# Silhouette Score
ax2.plot(K_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
ax2.set_title('Silhouette Score', fontsize=16, fontweight='bold')
ax2.set_xlabel('Jumlah Cluster (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_xticks(list(K_range))

# Tandai k optimal
optimal_k = list(K_range)[np.argmax(silhouette_scores)]
ax2.axvline(optimal_k, color='green', linestyle='--', linewidth=2, 
            label=f'Optimal k={optimal_k}')
ax2.legend(fontsize=12)

plt.suptitle('Penentuan Jumlah Cluster Optimal', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('output/elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n[TROPHY] K optimal berdasarkan Silhouette Score: k = {optimal_k}")

# Simpan juga terpisah
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_title('Elbow Method', fontsize=16, fontweight='bold')
ax1.set_xlabel('Jumlah Cluster (k)')
ax1.set_ylabel('Inertia')
ax1.set_xticks(list(K_range))
plt.tight_layout()
plt.savefig('output/elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(K_range, silhouette_scores, 'rs-', linewidth=2, markersize=8)
ax2.set_title('Silhouette Score', fontsize=16, fontweight='bold')
ax2.set_xlabel('Jumlah Cluster (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_xticks(list(K_range))
ax2.axvline(optimal_k, color='green', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
ax2.legend(fontsize=12)
plt.tight_layout()
plt.savefig('output/silhouette_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ### 5.3 K-Means Clustering

# %%
# Terapkan K-Means dengan k optimal
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"[OK] K-Means Clustering selesai (k={optimal_k})")
print(f"\nDistribusi Cluster:")
cluster_dist = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_dist.items():
    print(f"   Cluster {cluster_id}: {count:,} konten ({count/len(df)*100:.1f}%)")

# %% [markdown]
# ### 5.4 Visualisasi Cluster dengan PCA

# %%
# Reduksi dimensi ke 2D menggunakan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='Set1',
                     alpha=0.5, s=15, edgecolors='none')

# Tambahkan centroid
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', 
           s=200, linewidths=2, edgecolors='white', zorder=5, label='Centroid')

ax.set_title(f'Visualisasi Cluster (PCA 2D) — k={optimal_k}', fontsize=16, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')

legend = ax.legend(*scatter.legend_elements(), title="Cluster", fontsize=11)
ax.add_artist(legend)
ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('output/pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"[DATA] PCA Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# %% [markdown]
# ### 5.5 Analisis Karakteristik Cluster

# %%
# Rata-rata fitur per cluster
print("[DATA] RATA-RATA FITUR PER CLUSTER")
print("=" * 70)
cluster_analysis = df.groupby('cluster')[cluster_features].mean().round(2)
print(cluster_analysis.to_string())

# %%
# Distribusi type per cluster
print("\n[DATA] DISTRIBUSI TIPE KONTEN PER CLUSTER")
print("=" * 50)
type_cluster = pd.crosstab(df['cluster'], df['type'], normalize='index').round(3) * 100
print(type_cluster.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
type_cluster.plot(kind='bar', ax=ax, color=['#3B5998', '#2ECC71'])
ax.set_title('Distribusi Tipe Konten per Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster')
ax.set_ylabel('Persentase (%)')
ax.legend(title='Tipe')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/tipe_per_cluster.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Genre dominan per cluster
print("\n[DATA] TOP 3 GENRE PER CLUSTER")
print("=" * 60)
for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    genres = cluster_data['listed_in'].str.split(',').explode().str.strip()
    top_genres = genres.value_counts().head(3)
    print(f"\nCluster {cluster_id} ({len(cluster_data):,} konten):")
    for genre, count in top_genres.items():
        print(f"   * {genre}: {count} ({count/len(cluster_data)*100:.1f}%)")

# %% [markdown]
# ### 5.6 Interpretasi Cluster

# %%
print("\n" + "=" * 70)
print("[SEARCH] INTERPRETASI CLUSTER")
print("=" * 70)

for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    avg_year = cluster_data['release_year'].mean()
    avg_dur = cluster_data['duration_int'].mean()
    movie_pct = (cluster_data['type'] == 'Movie').mean() * 100
    top_genre = (cluster_data['listed_in'].str.split(',').explode().str.strip()
                 .value_counts().index[0])
    top_country = cluster_data['country_main'].value_counts().index[0]
    
    print(f"\n[PIN] Cluster {cluster_id} ({len(cluster_data):,} konten):")
    print(f"   Rata-rata tahun rilis : {avg_year:.0f}")
    print(f"   Rata-rata durasi      : {avg_dur:.0f}")
    print(f"   Persentase Movie      : {movie_pct:.1f}%")
    print(f"   Genre dominan         : {top_genre}")
    print(f"   Negara dominan        : {top_country}")
    
    # Auto interpretasi
    if avg_year >= 2017:
        era = "konten baru/modern"
    elif avg_year >= 2010:
        era = "konten era pertengahan"
    else:
        era = "konten lama/klasik"
    
    if avg_dur > 100:
        dur_desc = "berdurasi panjang"
    elif avg_dur > 50:
        dur_desc = "berdurasi menengah"
    else:
        dur_desc = "berdurasi pendek"
    
    content_type = "didominasi Movie" if movie_pct > 60 else "didominasi TV Show" if movie_pct < 40 else "campuran Movie & TV Show"
    
    print(f"   -> Interpretasi: {era}, {dur_desc}, {content_type}")

# %% [markdown]
# ---
# ## TAHAP 6: Kesimpulan & Insight

# %%
print("\n" + "=" * 70)
print("[NOTE] KESIMPULAN & INSIGHT")
print("=" * 70)

print("""
===================================================================
[DATA] TEMUAN UTAMA EDA
===================================================================
1. Netflix memiliki total {total:,} konten, didominasi oleh Movie ({movie_pct:.1f}%)
2. Rating paling umum adalah TV-MA (konten dewasa), menunjukkan target 
   audiens utama Netflix adalah penonton dewasa
3. Genre terpopuler: International Movies, Dramas, dan Comedies
4. Produksi konten melonjak pesat sejak 2015, puncak di 2017-2019
5. United States mendominasi produksi konten Netflix

===================================================================
[AI] HASIL CLASSIFICATION (Movie vs TV Show)
===================================================================
- Model terbaik: {best_model} ({best_acc:.2f}%)
- Fitur terpenting: duration_int (durasi adalah pembeda utama antara 
  Movie dan TV Show)
- Classification berjalan baik karena Movie dan TV Show memiliki 
  karakteristik yang cukup berbeda

===================================================================
[TARGET] HASIL CLUSTERING
===================================================================
- Jumlah cluster optimal: k={k_opt}
- Clustering berhasil mengelompokkan konten Netflix berdasarkan 
  kombinasi tahun rilis, durasi, dan kompleksitas genre

===================================================================
[IDEA] REKOMENDASI / ACTIONABLE INSIGHTS
===================================================================
1. Netflix dapat memperbanyak konten International & Drama karena 
   genre ini paling populer
2. Fokus pada konten dewasa (TV-MA) karena mendominasi katalog, 
   tetapi juga diversifikasi ke konten keluarga
3. Kolaborasi dengan lebih banyak negara non-US untuk memperkaya 
   variasi konten global
4. Durasi film 90-120 menit adalah sweet spot yang paling umum — 
   bisa dijadikan acuan produksi
5. Pertumbuhan konten melambat sejak 2020, perlu strategi baru 
   untuk mempertahankan pertumbuhan katalog
""".format(
    total=len(df),
    movie_pct=(df['type'] == 'Movie').mean() * 100,
    best_model=best_model_name,
    best_acc=results[best_model_name]['accuracy'] * 100,
    k_opt=optimal_k
))

# %%
print("\n[OK] Project Pengantar Sains Data — Netflix Analysis SELESAI!")
print(f"[FOLDER] Semua visualisasi tersimpan di folder 'output/'")
print(f"[DATA] Total visualisasi: {len([f for f in os.listdir('output') if f.lower().endswith('.png')])} file PNG")

# Buat dashboard HTML untuk semua visualisasi
image_cards = [
    ('Distribusi Tipe Konten', 'distribusi_tipe_konten.png'),
    ('Distribusi Rating', 'distribusi_rating.png'),
    ('Top 10 Negara Produksi', 'top_negara.png'),
    ('Top 10 Genre Paling Populer', 'top_genre.png'),
    ('Tren Produksi Konten per Tahun', 'tren_produksi.png'),
    ('Distribusi Durasi Film (Movie)', 'distribusi_durasi.png'),
    ('Konten Ditambahkan per Tahun', 'konten_per_tahun.png'),
    ('Confusion Matrix Semua Model', 'confusion_matrix_semua.png'),
    ('Confusion Matrix Decision Tree', 'confusion_matrix_decision_tree.png'),
    ('Confusion Matrix Random Forest', 'confusion_matrix_random_forest.png'),
    ('Confusion Matrix Logistic Regression', 'confusion_matrix_logistic_regression.png'),
    ('Perbandingan Akurasi Model', 'perbandingan_model.png'),
    ('Feature Importance Random Forest', 'feature_importance.png'),
    ('Elbow Method', 'elbow_method.png'),
    ('Silhouette Scores', 'silhouette_scores.png'),
    ('PCA Cluster Visualization', 'pca_clusters.png'),
    ('Distribusi Tipe Konten per Cluster', 'tipe_per_cluster.png')
]

dashboard_html = os.path.join('output', 'dashboard.html')
with open(dashboard_html, 'w', encoding='utf-8') as f:
    f.write(f"""<!DOCTYPE html>
<html lang='id'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Dashboard Netflix Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
        h1, h2 { color: #222; }
        .summary { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }
        .card { background: #ffffff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }
        .card h3 { margin: 0; padding: 16px; font-size: 18px; background: #2b6cb0; color: white; }
        .card img { width: 100%; display: block; }
        .footer { margin-top: 32px; font-size: 14px; color: #555; text-align: center; }
        a.button { display:inline-block; margin-top:12px; padding:10px 18px; background:#2b6cb0; color:white; text-decoration:none; border-radius:8px; }
    </style>
</head>
<body>
    <div class='container'>
        <h1>Dashboard Netflix Analysis</h1>
        <div class='summary'>
            <h2>Ringkasan Hasil</h2>
            <p><strong>Total visualisasi:</strong> {len([f for f in os.listdir('output') if f.lower().endswith('.png')])}</p>
            <p><strong>Model terbaik:</strong> {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}% akurasi)</p>
            <p><strong>Cluster optimal:</strong> k = {optimal_k}</p>
            <p><strong>Proporsi Movie:</strong> {(df['type'] == 'Movie').mean()*100:.1f}%</p>
            <p><strong>Proporsi TV Show:</strong> {(df['type'] == 'TV Show').mean()*100:.1f}%</p>
            <a class='button' href='dashboard.html'>Refresh halaman ini setelah menjalankan ulang script</a>
        </div>
        <div class='grid'>
""")
    for title, img in image_cards:
        if os.path.exists(os.path.join('output', img)):
            f.write(f"        <div class='card'>\n")
            f.write(f"            <h3>{title}</h3>\n")
            f.write(f"            <img src='{img}' alt='{title}'>\n")
            f.write(f"        </div>\n")
    f.write("""
        </div>
        <div class='footer'>
            <p>Dashboard otomatis dibuat oleh script <strong>netflix_analysis.py</strong>.</p>
        </div>
    </div>
</body>
</html>
""")

print(f"[WEB] Dashboard HTML telah dibuat: {dashboard_html}")
