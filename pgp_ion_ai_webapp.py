#!/usr/bin/env python3
"""
pGP Efflux Ratio & Ion Channel Activity Predictor - Yapay Zeka Web Uygulaması
ChEMBL verileri + AMPL ModelPipeline'den esinlenerek basit RF + Morgan FP ile eğitilmiş demo.
Kullanıcı dostu Streamlit arayüzü ile molekül tahmini + Uygulanabilirlik Alanı (AD) analizi.
"""

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# XGBoost (daha güçlü model için)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ----------------- AMPL Pipeline'den AD Fonksiyonları (kopyalandı ve uyarlandı) -----------------
def get_k_nearest_index(dist_matrix, k):
    """Returns the indexes of the nearest k neighbors."""
    result = []
    for i in range(len(dist_matrix)):
        kn_idx = np.argpartition(dist_matrix[i], k)[:k]
        result.append(kn_idx)
    return np.vstack(result)

def calc_AD_kmean_dist(train_dset, pred_dset, k=5, dist_metric="euclidean"):
    """
    AMPL pipeline'den uyarlanmış K-en yakın komşu mesafe tabanlı Uygulanabilirlik Alanı (AD) skoru.
    Düşük skor (< ~2) = eğitim verisine yakın, güvenilir tahmin.
    """
    from sklearn.metrics import pairwise_distances
    if len(train_dset) == 0 or len(pred_dset) == 0:
        return np.array([np.nan] * len(pred_dset)), None
    
    train_dset_pair_distance = pairwise_distances(X=train_dset, metric=dist_metric)
    train_kmean_dis = []
    kn_idxes = get_k_nearest_index(train_dset_pair_distance, k + 1)
    for i in range(len(train_dset_pair_distance)):
        dis = np.sum(train_dset_pair_distance[i][kn_idxes[i]]) / k
        train_kmean_dis.append(dis)
    
    from scipy import stats
    train_dset_distribution = stats.norm.fit(train_kmean_dis)
    
    pred_size = len(pred_dset)
    train_pred_dis = pairwise_distances(X=pred_dset, Y=train_dset, metric=dist_metric)
    train_pred_kn_idxes = get_k_nearest_index(train_pred_dis, k)
    pred_kmean_dis_score = np.zeros(pred_size)
    
    for i in range(pred_size):
        dists = train_pred_dis[i][train_pred_kn_idxes[i]]
        pred_km_dis = np.mean(dists)
        train_dset_std = train_dset_distribution[1] if train_dset_distribution[1] != 0 else 1e-6
        pred_kmean_dis_score[i] = max(1e-6, (pred_km_dis - train_dset_distribution[0]) / train_dset_std)
    return pred_kmean_dis_score, train_pred_kn_idxes

# ----------------- Yardımcı Fonksiyonlar -----------------
@st.cache_data(show_spinner=False)
def load_pgp_data():
    """pGP MDCK efflux ratio verisini yükle ve temizle."""
    # GitHub + Streamlit Cloud uyumlu relative path
    data_path = os.path.join(os.path.dirname(__file__), "data", "pGP_MDCK_efflux_ratio_chembl29.csv")
    if not os.path.exists(data_path):
        data_path = "data/pGP_MDCK_efflux_ratio_chembl29.csv"  # fallback
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['efflux_ratio', 'base_rdkit_smiles'])
    df = df.drop_duplicates(subset=['base_rdkit_smiles'])
    df['efflux_ratio'] = pd.to_numeric(df['efflux_ratio'], errors='coerce')
    df = df.dropna(subset=['efflux_ratio'])
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_ion_data():
    """KCNA5 / KCNH2 / SCN5A verisini yükle."""
    # GitHub + Streamlit Cloud uyumlu relative path
    data_path = os.path.join(os.path.dirname(__file__), "data", "KCNA5_KCNH2_SCN5A_data.csv")
    if not os.path.exists(data_path):
        data_path = "data/KCNA5_KCNH2_SCN5A_data.csv"  # fallback
    df = pd.read_csv(data_path)
    # compound_id aslında SMILES
    df = df.rename(columns={'compound_id': 'smiles'})
    df = df.dropna(subset=['smiles'])
    for col in ['target_KCNA5_standard_value', 'target_KCNH2_standard_value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.reset_index(drop=True)

def compute_morgan_fps(smiles_list, radius=2, nBits=2048):
    """RDKit Morgan (ECFP benzeri) parmak izi hesapla."""
    fps = []
    valid_idx = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(np.array(fp, dtype=np.float32))
            valid_idx.append(i)
        else:
            fps.append(np.zeros(nBits, dtype=np.float32))
    return np.array(fps), valid_idx


def compute_rdkit_descriptors(smiles_list):
    """RDKit 2D descriptor hesapla (daha güçlü feature için)."""
    all_desc = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_desc)
    
    descs = []
    valid_idx = []
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(s))
        if mol is not None:
            try:
                desc = calc.CalcDescriptors(mol)
                descs.append(desc)
                valid_idx.append(i)
            except:
                pass
    if len(descs) == 0:
        return np.array([]), []
    return np.array(descs, dtype=np.float32), valid_idx

def draw_molecule(smiles, size=(300, 300)):
    """SMILES'ten molekül resmi oluştur."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    return img

def get_ad_interpretation(score):
    """AD skoruna göre yorum."""
    if np.isnan(score):
        return "⚠️ AD hesaplanamadı"
    if score < 1.5:
        return "✅ **Güvenilir** - Eğitim verisine çok yakın (AD içinde)"
    elif score < 3.0:
        return "🟡 **Orta güvenilirlik** - Sınırda, dikkatli yorumla"
    else:
        return "🔴 **Düşük güvenilirlik** - Eğitim verisinden uzak (AD dışı)"

# ----------------- Model Eğitimi ve Kaydetme -----------------
MODEL_DIR = '/home/workdir/artifacts/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_save_pgp_model(df, n_estimators=150, feature_type="morgan", model_type="rf"):
    """
    pGP efflux için gelişmiş model eğit (Morgan veya RDKit Descriptor + RF/XGBoost).
    Bu fonksiyon mevcut app'i daha güçlü yapay zeka haline getirir.
    """
    if feature_type == "morgan":
        st.info("🔄 Model eğitiliyor... (Morgan Fingerprint + " + model_type.upper() + ")")
        X, valid_idx = compute_morgan_fps(df['base_rdkit_smiles'].values)
    else:
        st.info("🔄 Model eğitiliyor... (RDKit Descriptor + " + model_type.upper() + ") - Daha güçlü!")
        X, valid_idx = compute_rdkit_descriptors(df['base_rdkit_smiles'].values)
    
    y = df['efflux_ratio'].values[valid_idx]
    
    if len(X) < 50:
        st.error("Yetersiz veri!")
        return None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model_type == "rf" or not XGBOOST_AVAILABLE:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model_name = "rf"
    else:
        # XGBoost - genellikle daha güçlü
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model_name = "xgb"
    
    model.fit(X_train, y_train)
    
    # Test performansı
    y_pred = model.predict(X_test)
    metrics = {
        'R2 Skoru': round(r2_score(y_test, y_pred), 3),
        'MAE (Ortalama Mutlak Hata)': round(mean_absolute_error(y_test, y_pred), 3),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        'Eğitim örnek sayısı': len(X_train),
        'Test örnek sayısı': len(X_test),
        'Feature Tipi': feature_type,
        'Model Tipi': model_type
    }
    
    # Kaydet
    model_path = os.path.join(MODEL_DIR, f'pgp_efflux_{model_name}_{feature_type}_model.pkl')
    joblib.dump(model, model_path)
    np.save(os.path.join(MODEL_DIR, 'pgp_train_X.npy'), X_train)
    np.save(os.path.join(MODEL_DIR, 'pgp_train_y.npy'), y_train)
    
    return model, metrics, X_train, model_path

def load_pgp_model():
    """Kaydedilmiş pGP modelini yükle."""
    model_path = os.path.join(MODEL_DIR, 'pgp_efflux_rf_model.pkl')
    X_train_path = os.path.join(MODEL_DIR, 'pgp_train_X.npy')
    if os.path.exists(model_path) and os.path.exists(X_train_path):
        model = joblib.load(model_path)
        X_train = np.load(X_train_path)
        return model, X_train
    return None, None

def predict_pgp(smiles, model, X_train):
    """Yeni SMILES için efflux tahmini + AD skoru."""
    fps, _ = compute_morgan_fps([smiles])
    if fps.shape[0] == 0 or np.all(fps[0] == 0):
        return None, None, None
    
    pred = model.predict(fps)[0]
    
    # AD skoru (k=5)
    ad_scores, _ = calc_AD_kmean_dist(X_train, fps, k=5)
    ad_score = ad_scores[0] if len(ad_scores) > 0 else np.nan
    
    return float(pred), ad_score, fps

# ----------------- STREAMLIT ARAYÜZÜ -----------------
st.set_page_config(
    page_title="🧬 pGP & İyon Kanalı Yapay Zeka",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 pGP Efflux Oranı ve İyon Kanalı Aktivite Tahmin Yapay Zekası")
st.markdown("""
**ChEMBL29 verileri** üzerinde eğitilmiş Makine Öğrenmesi modeli.  
**AMPL ModelPipeline** (model_pipeline.py) içindeki Uygulanabilirlik Alanı (AD) hesaplamasından esinlenilmiştir.  
Basit ve hızlı demo için **Morgan parmak izi (ECFP)** + **Random Forest Regressor** kullanılmıştır.
""")

# Sidebar menü
menu = st.sidebar.radio(
    "📍 Menü",
    ["🏠 Ana Sayfa", "📊 Veri Keşfi (pGP)", "📊 Veri Keşfi (İyon Kanalları)", 
     "🧠 Model Eğitimi (pGP Efflux)", "🔮 Yeni Molekül Tahmini (pGP)", 
     "ℹ️ Teknik Bilgi & Kaynaklar"],
    index=0
)

# ----------------- ANA SAYFA -----------------
if menu == "🏠 Ana Sayfa":
    st.header("Hoş Geldiniz!")
    st.markdown("""
    Bu uygulama, ilaç keşfi alanında kritik öneme sahip iki hedefi tahmin etmek için tasarlanmıştır:
    
    1. **pGP (P-glikoprotein) MDCK Efflux Oranı**  
       - Düşük değer (< 2-3) → İyi BBB penetrasyonu (CNS ilaçları için avantaj)
       - Yüksek değer → Efflux pompası tarafından dışarı atılır
    
    2. **İyon Kanalları (KCNA5, KCNH2, SCN5A)**  
       - Kardiyovasküler ve nörolojik ilaç geliştirme için önemli hedefler
       - Değerler genellikle pIC50 (-log IC50) cinsindendir (yüksek = güçlü inhibitör)
    
    **Nasıl Kullanılır?**
    - Sol menüden "Model Eğitimi (pGP Efflux)" → Modeli eğit (ilk seferde ~30-60 sn)
    - Sonra "Yeni Molekül Tahmini (pGP)" ile SMILES girip tahmin al
    - AD skoru ile tahminin güvenilirliğini gör
    """)
    
    st.info("💡 İpucu: Modeli bir kez eğittikten sonra tahmin sayfası anında çalışır. Kaydedilen model artifacts/models klasöründe saklanır.")
    
    st.subheader("📁 Kullanılan Veriler")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("pGP Efflux Veri Seti", "~1,100 bileşik", "ChEMBL29")
    with col2:
        st.metric("İyon Kanalı Veri Seti", "~9,000 bileşik", "KCNA5/KCNH2/SCN5A")

# ----------------- VERİ KEŞFİ pGP -----------------
elif menu == "📊 Veri Keşfi (pGP)":
    st.header("📊 pGP MDCK Efflux Ratio Veri Seti Keşfi")
    df = load_pgp_data()
    
    st.subheader("Veri Özeti")
    st.write(f"**Toplam bileşik:** {len(df)}")
    st.dataframe(df[['base_rdkit_smiles', 'efflux_ratio', 'log_efflux_ratio']].head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Efflux Ratio Dağılımı")
        st.bar_chart(df['efflux_ratio'].value_counts(bins=20).sort_index())
    with col2:
        st.subheader("İstatistikler")
        st.write(df['efflux_ratio'].describe())
    
    st.caption("Not: Efflux_ratio < 1 genellikle iyi kabul edilir (düşük efflux).")

# ----------------- VERİ KEŞFİ İYON -----------------
elif menu == "📊 Veri Keşfi (İyon Kanalları)":
    st.header("📊 KCNA5 / KCNH2 / SCN5A Veri Seti")
    df_ion = load_ion_data()
    
    st.write(f"**Toplam satır:** {len(df_ion)}")
    
    target_choice = st.selectbox("Hedef seçin:", 
                                  ["target_KCNA5_standard_value", "target_KCNH2_standard_value"])
    
    subdf = df_ion.dropna(subset=[target_choice])
    st.write(f"**{target_choice} için geçerli veri:** {len(subdf)}")
    
    st.dataframe(subdf[['smiles', target_choice]].head(8), use_container_width=True)
    
    st.bar_chart(subdf[target_choice].value_counts(bins=15).sort_index())
    st.caption("Değerler pIC50 benzeri (yüksek = daha potent inhibitör). SCN5A verisi daha seyrek.")

# ----------------- MODEL EĞİTİMİ -----------------
elif menu == "🧠 Model Eğitimi (pGP Efflux)":
    st.header("🧠 pGP Efflux Tahmin Modeli Eğitimi")
    st.markdown("Random Forest + 2048-bit Morgan parmak izi ile regresyon modeli. AMPL pipeline'deki AD mantığı entegre edilmiştir.")
    
    df = load_pgp_data()
    
    st.subheader("Model Ayarları (Daha Güçlü Yapay Zeka İçin)")

col1, col2 = st.columns(2)
with col1:
    feature_type = st.selectbox(
        "Feature Tipi",
        ["morgan", "rdkit_descriptor"],
        index=0,
        help="rdkit_descriptor genellikle daha güçlü sonuç verir (Mordred'e yakın)"
    )
with col2:
    model_type = st.selectbox(
        "Model Tipi",
        ["rf", "xgb"] if XGBOOST_AVAILABLE else ["rf"],
        index=0,
        help="XGBoost genellikle Random Forest'tan daha iyi performans gösterir"
    )

if st.button("🚀 Modeli Eğit ve Kaydet", type="primary"):
    with st.spinner("Model eğitiliyor... Lütfen bekleyin"):
        result = train_and_save_pgp_model(df, feature_type=feature_type, model_type=model_type)
        if result[0] is not None:
            model, metrics, X_train, model_path = result
            st.success("✅ Model başarıyla eğitildi ve kaydedildi!")
            
            st.subheader("📈 Model Performans Metrikleri (Test Seti)")
            for k, v in metrics.items():
                st.metric(k, v)
            
            st.info(f"Model kaydedildi: {model_path}")
            
            # Feature Importance (eğer model destekliyorsa)
            if hasattr(model, 'feature_importances_'):
                st.subheader("🔍 En Önemli 15 Feature")
                importances = model.feature_importances_
                if feature_type == "morgan":
                    feat_names = [f"Bit_{i}" for i in range(len(importances))]
                else:
                    feat_names = [f"RDKit_{i}" for i in range(len(importances))]
                
                imp_df = pd.DataFrame({
                    'Feature': feat_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(15)
                st.dataframe(imp_df, use_container_width=True)

# ----------------- TAHMİN SAYFASI -----------------
elif menu == "🔮 Yeni Molekül Tahmini (pGP)":
    st.header("🔮 Yeni Molekül için pGP Efflux Tahmini")
    
    model, X_train = load_pgp_model()
    
    if model is None:
        st.warning("⚠️ Henüz model eğitilmedi! Lütfen sol menüden **🧠 Model Eğitimi (pGP Efflux)** sayfasına gidip modeli eğitin.")
        st.stop()
    
    st.success("✅ Eğitilmiş model yüklendi. SMILES girerek tahmin yapabilirsiniz.")
    
    smiles_input = st.text_input(
        "SMILES dizesini girin:",
        value="CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # Örnek: Morfin benzeri
        placeholder="Örnek: c1ccccc1 (benzen) veya tam SMILES"
    )
    
    if st.button("🔮 Tahmin Et", type="primary"):
        if not smiles_input.strip():
            st.error("Lütfen bir SMILES girin.")
        else:
            with st.spinner("Tahmin ve AD analizi yapılıyor..."):
                pred, ad_score, fps = predict_pgp(smiles_input, model, X_train)
            
            if pred is None:
                st.error("Geçersiz SMILES! Lütfen doğru bir yapı girin.")
            else:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Molekül Görselleştirmesi")
                    img = draw_molecule(smiles_input)
                    if img:
                        st.image(img, caption="Girilen Molekül", use_column_width=True)
                    else:
                        st.warning("Molekül çizilemedi.")
                
                with col2:
                    st.subheader("Tahmin Sonuçları")
                    st.metric("Tahmini Efflux Ratio", f"{pred:.2f}")
                    st.metric("Log10(Efflux)", f"{np.log10(max(pred, 0.01)):.2f}")
                    
                    st.markdown("**Uygulanabilirlik Alanı (AD) Skoru**")
                    st.metric("AD Skoru (düşük = daha güvenilir)", f"{ad_score:.2f}")
                    st.markdown(get_ad_interpretation(ad_score))
                    
                    if pred < 2.0:
                        st.success("✅ Düşük efflux → Muhtemelen iyi BBB penetrasyonu")
                    elif pred < 5.0:
                        st.info("🟡 Orta efflux")
                    else:
                        st.warning("🔴 Yüksek efflux → CNS için sorunlu olabilir")
    
    st.markdown("---")
    st.caption("Not: Bu demo modeli eğitim verisinin ~80% ile eğitilmiştir. Gerçek projelerde AMPL pipeline ile hiperparametre optimizasyonu, graph conv veya NN modelleri kullanılır.")

# ----------------- TEKNİK BİLGİ -----------------
elif menu == "ℹ️ Teknik Bilgi & Kaynaklar":
    st.header("ℹ️ Teknik Detaylar")
    
    st.markdown("""
    ### Kullanılan Teknolojiler
    - **Featurization**: RDKit Morgan Fingerprint (radius=2, 2048 bit) → ECFP benzeri
    - **Model**: scikit-learn RandomForestRegressor (n_estimators=150, max_depth=20)
    - **Uygulanabilirlik Alanı (AD)**: AMPL `model_pipeline.py` içindeki `calc_AD_kmean_dist` fonksiyonundan uyarlanmıştır (K=5 en yakın komşu Euclidean mesafe + z-score)
    - **Web Framework**: Streamlit
    - **Veri**: ChEMBL29 pGP MDCK efflux + KCNA5/KCNH2/SCN5A bioactivity
    
    ### AMPL Pipeline ile İlişki
    Bu uygulama, sağlanan `model_pipeline.py` dosyasındaki mantığı basitleştirerek web arayüzüne taşımaktadır:
    - Model eğitimi, veri yükleme, split, transformer mantığı esinlenilmiştir
    - AD analizi doğrudan pipeline kodundan alınmıştır
    - Gerçek AMPL ile tam uyumlu NN/GraphConv/XGBoost + hyperparam search + model tracker yapılabilir (daha ağır altyapı gerekir)
    
    ### Geliştirme Önerileri (İleri Seviye)
    - DeepChem + GraphConv veya AttentiveFP ile daha güçlü modeller
    - Multi-task öğrenme (pGP + ion kanalları birlikte)
    - SMILES augmentasyonu veya uncertainty estimation (ensemble)
    - Streamlit + RDKit + Plotly ile interaktif kimyasal alan görselleştirmesi
    """)
    
    st.subheader("Dosya Konumları")
    st.code("""
    artifacts/
    ├── pgp_ion_ai_webapp.py          # Bu uygulama
    └── models/
        ├── pgp_efflux_rf_model.pkl   # Eğitilmiş model
        ├── pgp_train_X.npy           # AD için eğitim parmak izleri
        └── pgp_train_y.npy
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using RDKit + scikit-learn + Streamlit\nInspired by ATOM/AMPL model_pipeline.py")