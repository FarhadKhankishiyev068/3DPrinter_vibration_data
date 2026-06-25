import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, entropy
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from gemma_client import OllamaClient
from rul_forecaster import RULForecaster

# --- Configuration: set these env vars or edit the defaults ---
data_dir = os.environ.get('DATA_DIR', './data')
output_dir = os.environ.get('OUTPUT_DIR', './output')
os.makedirs(output_dir, exist_ok=True)

config_path = os.path.join(data_dir, 'device_config.json')

with open(config_path, 'r') as f:
    config = json.load(f)

conf = None
for comp in config['devices'][0]['components']:
    if 'iis3dwb_acc' in comp:
        conf = comp['iis3dwb_acc']
        break

measodr = int(conf['measodr'])
sensitivity = conf['sensitivity']
dim = conf['dim']
dt = np.dtype(np.int16) if conf['data_type'] == 'int16' else np.float32

def apply_filters(data, fs):
    nyq = 0.5 * fs
    low_cut = 10.0 / nyq
    b_high, a_high = butter(4, low_cut, btype='high')
    
    high_cut = 10000.0 / nyq
    b_low, a_low = butter(4, high_cut, btype='low')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        y = filtfilt(b_high, a_high, data[:, i])
        y = filtfilt(b_low, a_low, y)
        filtered[:, i] = y
    return filtered

def extract_features(data_chunk, fs):
    features = []
    for axis in range(data_chunk.shape[1]):
        ax_data = data_chunk[:, axis]
        rms = np.sqrt(np.mean(ax_data**2))
        var = np.var(ax_data)
        sk = skew(ax_data)
        kurt = kurtosis(ax_data)
        
        fft_vals = np.abs(np.fft.rfft(ax_data)) / len(ax_data)
        fft_vals[0] = 0
        
        spec_entropy = entropy(fft_vals + 1e-12)
        
        peak_indices = np.argsort(fft_vals)[-3:][::-1]
        peak_freq_1 = peak_indices[0] * (fs / len(ax_data))
        peak_mag_1 = fft_vals[peak_indices[0]]
        
        peak_freq_2 = peak_indices[1] * (fs / len(ax_data)) if len(peak_indices) > 1 else 0
        peak_mag_2 = fft_vals[peak_indices[1]] if len(peak_indices) > 1 else 0
        
        peak_freq_3 = peak_indices[2] * (fs / len(ax_data)) if len(peak_indices) > 2 else 0
        peak_mag_3 = fft_vals[peak_indices[2]] if len(peak_indices) > 2 else 0
        
        features.extend([rms, var, sk, kurt, spec_entropy, 
                         peak_freq_1, peak_mag_1, 
                         peak_freq_2, peak_mag_2, 
                         peak_freq_3, peak_mag_3])
    return features

max_seconds = 2000
window_size = int(measodr * 1.0) # 1.0s windows

print("Extracting and filtering data...")
files = {'iis3dwb_acc_23.dat': 0, 'iis3dwb_acc_23ang.dat': 1}
features_list = []

for fname, label in files.items():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        continue
    raw = np.fromfile(fpath, dtype=dt, count=int(measodr * dim * max_seconds))
    actual_samples = len(raw) // dim
    data = raw[:actual_samples*dim].reshape(-1, dim) * sensitivity
    
    data = apply_filters(data, measodr)
    
    n_windows = data.shape[0] // window_size
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = data[start:end]
        
        feat = extract_features(window_data, measodr)
        row = {'label': label}
        idx = 0
        for axis_name in ['X', 'Y', 'Z']:
            row[f'{axis_name}_rms'] = feat[idx]
            row[f'{axis_name}_var'] = feat[idx+1]
            row[f'{axis_name}_skew'] = feat[idx+2]
            row[f'{axis_name}_kurt'] = feat[idx+3]
            row[f'{axis_name}_entropy'] = feat[idx+4]
            row[f'{axis_name}_peak1_freq'] = feat[idx+5]
            row[f'{axis_name}_peak1_mag'] = feat[idx+6]
            row[f'{axis_name}_peak2_freq'] = feat[idx+7]
            row[f'{axis_name}_peak2_mag'] = feat[idx+8]
            row[f'{axis_name}_peak3_freq'] = feat[idx+9]
            row[f'{axis_name}_peak3_mag'] = feat[idx+10]
            idx += 11
        features_list.append(row)

df = pd.DataFrame(features_list)

print("Saving Correlation Matrix...")
corr_features = ['label', 'X_rms', 'Y_rms', 'Z_rms', 'X_var', 'Y_var', 'Z_var']
corr_df = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(corr_features)))
ax.set_yticks(range(len(corr_features)))
ax.set_xticklabels(corr_features, rotation=45)
ax.set_yticklabels(corr_features)
plt.title('Correlation Matrix (Angle vs RMS/Var)', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

print("Scaling and PCA Orthogonalization (Full Data Visualization)...")
X = df.drop('label', axis=1)
y = df['label']

scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

pca_full = PCA(n_components=0.95)
X_pca_full = pca_full.fit_transform(X_scaled_full)
print(f"PCA reduced dimensions to: {X_pca_full.shape[1]}")

print("Saving PCA Visualization...")
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], c=y, cmap='coolwarm', alpha=0.8, edgecolors='k')
plt.title('PCA: 0-Degree vs 90-Degree Separability (2000s data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['0-Degree', '90-Degree'])
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'pca_clusters.png'))
plt.close()

print("Training XGBoost with 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
all_y_te = []
all_preds = []

for train_index, test_index in skf.split(X, y):
    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
    
    scaler_cv = StandardScaler()
    X_tr_sc = scaler_cv.fit_transform(X_tr)
    X_te_sc = scaler_cv.transform(X_te)
    
    pca_cv = PCA(n_components=0.95)
    X_tr_pca = pca_cv.fit_transform(X_tr_sc)
    X_te_pca = pca_cv.transform(X_te_sc)
    
    xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss')
    xgb_clf.fit(X_tr_pca, y_tr)
    
    preds = xgb_clf.predict(X_te_pca)
    cv_scores.append(accuracy_score(y_te, preds))
    all_y_te.extend(y_te)
    all_preds.extend(preds)

xgb_acc = np.mean(cv_scores)
print(f"XGBoost 5-Fold CV Mean Accuracy: {xgb_acc:.4f}")

print("Saving CV Accuracy Visualization...")
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color='g')
plt.axhline(y=xgb_acc, color='r', linestyle='--', label=f'Mean Accuracy: {xgb_acc:.4f}')
plt.title('XGBoost 5-Fold Cross-Validation Accuracy')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.05)
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cv_accuracy.png'))
plt.close()

print("Saving Confusion Matrix Visualization...")
cm = confusion_matrix(all_y_te, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0-Degree', '90-Degree'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('XGBoost 5-Fold Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

print("Forecasting with TimesFM...")
rul_model = RULForecaster()
rul_model.load_model()

fig, axs = plt.subplots(3, 1, figsize=(12, 15))
axes_names = ['X', 'Y', 'Z']
forecast_results = []

for idx, axis_name in enumerate(axes_names):
    hist_0 = df[df['label'] == 0][f'{axis_name}_rms'].tolist()
    hist_1 = df[df['label'] == 1][f'{axis_name}_rms'].tolist()
    
    thresh_0 = max(hist_0) * 1.5
    ticks_0, slope_0 = rul_model.forecast_rul(hist_0, thresh_0)
    
    thresh_1 = max(hist_1) * 1.5
    ticks_1, slope_1 = rul_model.forecast_rul(hist_1, thresh_1)
    
    print(f"TimesFM Forecast {axis_name} 0-Degree: {ticks_0:.1f} ticks")
    print(f"TimesFM Forecast {axis_name} 90-Degree: {ticks_1:.1f} ticks")
    forecast_results.append(f"- {axis_name}-Axis: 0-Degree={ticks_0:.1f} ticks, 90-Degree={ticks_1:.1f} ticks")
    
    axs[idx].plot(hist_0, label=f'Historical {axis_name}_rms (0-Degree)', color='blue')
    if slope_0 > 0:
        forecast_line_0 = [hist_0[-1] + i * slope_0 for i in range(int(ticks_0))]
        axs[idx].plot(range(len(hist_0), len(hist_0) + int(ticks_0)), forecast_line_0, label='Forecast (0-Degree)', color='blue', linestyle='--')
    axs[idx].axhline(y=thresh_0, color='blue', linestyle=':', label='Threshold (0-Degree)')
    
    axs[idx].plot(hist_1, label=f'Historical {axis_name}_rms (90-Degree)', color='orange')
    if slope_1 > 0:
        forecast_line_1 = [hist_1[-1] + i * slope_1 for i in range(int(ticks_1))]
        axs[idx].plot(range(len(hist_1), len(hist_1) + int(ticks_1)), forecast_line_1, label='Forecast (90-Degree)', color='orange', linestyle='--')
    axs[idx].axhline(y=thresh_1, color='orange', linestyle=':', label='Threshold (90-Degree)')
    
    axs[idx].set_title(f'TimesFM {axis_name}_rms Forecast Trend: 0-Degree vs 90-Degree')
    axs[idx].set_xlabel('Time (windows)')
    axs[idx].set_ylabel(f'{axis_name}_rms Amplitude')
    axs[idx].legend()
    axs[idx].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'timesfm_forecast.png'))
plt.close()

forecast_str = "\n".join(forecast_results)

print("Querying Gemma 26b...")
prompt_text = f"""You are a vibration sensor analyst:
We collected 2000 seconds of 3D printer vibrations at 0-degree and 90-degree angles.
- XGBoost Mean CV Accuracy: {xgb_acc:.4f}

TimesFM Forecasts (Ticks until 1.5x max historical amplitude):
{forecast_str}

Analyze the class separability and what the TimesFM ticks indicate about axis stability across orientations. Keep it very brief."""

with open('gemma_prompt.txt', 'w') as f:
    f.write(prompt_text)

gemma = OllamaClient()
features_dict = {}
response = gemma.diagnose(features_dict, 'gemma_prompt.txt')

gemma_verdict = response.get('diagnosis', 'No response')
print("\n--- Gemma Output ---")
print(gemma_verdict)

output_data = {
    'xgb_accuracy': float(xgb_acc),
    'timesfm_forecast': forecast_results,
    'gemma_verdict': gemma_verdict,
    'features_3d': {
        'x_rms': df['X_rms'].tolist(),
        'y_rms': df['Y_rms'].tolist(),
        'z_rms': df['Z_rms'].tolist(),
        'label': df['label'].tolist()
    }
}
with open(os.path.join(output_dir, 'pipeline.json'), 'w') as f:
    json.dump(output_data, f, indent=4)
