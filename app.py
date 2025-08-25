import streamlit as st
import time 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import tempfile

class HerbalPlantClassifier:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.class_names = ['daun_jambu_biji', 'daun_sirih', 'daun_sirsak', 'lidah_buaya']
        
        # Threshold untuk rejection logic
        self.confidence_threshold = 0.6  # Minimum confidence untuk menerima prediksi
        self.uncertainty_threshold = 0.25  # Maximum entropy untuk menolak prediksi ambiguous

        self.manfaat_database = {
            'daun_jambu_biji': """Daun jambu biji memiliki berbagai manfaat kesehatan:
            • Mengendalikan diabetes dengan menurunkan kadar gula darah
            • Membantu mengatasi diare dan gangguan pencernaan
            • Mengandung antioksidan tinggi untuk meningkatkan imunitas
            • Memiliki sifat anti-inflamasi dan antibakteri
            • Membantu menurunkan kolesterol""",
            
            'daun_sirih': """Daun sirih memiliki berbagai manfaat kesehatan:
            • Menjaga kesehatan mulut dan mencegah bau mulut
            • Memiliki sifat antiseptik dan antibakteri alami
            • Membantu penyembuhan luka dan peradangan
            • Mengatasi keputihan pada wanita
            • Membantu mengobati batuk dan masalah pernapasan""",
            
            'daun_sirsak': """Daun sirsak memiliki berbagai manfaat kesehatan:
            • Sebagai antioksidan kuat untuk melawan radikal bebas
            • Memiliki sifat anti-inflamasi dan anti-kanker
            • Membantu menurunkan tekanan darah tinggi
            • Meningkatkan sistem kekebalan tubuh
            • Membantu mengatasi insomnia dan stres""",
            
            'lidah_buaya': """Lidah buaya memiliki berbagai manfaat kesehatan:
            • Menyembuhkan luka bakar dan iritasi kulit
            • Melembapkan kulit kering dan mengatasi eksim
            • Membantu pencernaan dan mengatasi maag
            • Mengandung vitamin dan mineral penting
            • Memiliki sifat anti-aging untuk kesehatan kulit"""
        }

    def calculate_entropy(self, predictions):
        """Menghitung entropy untuk mengukur uncertainty"""
        # Tambahkan epsilon untuk menghindari log(0)
        epsilon = 1e-8
        predictions_safe = np.clip(predictions, epsilon, 1.0 - epsilon)
        entropy = -np.sum(predictions_safe * np.log(predictions_safe))
        return entropy

    def is_valid_prediction(self, predictions):
        """
        Logic untuk menentukan apakah prediksi valid atau tidak
        Menggunakan beberapa metrik:
        1. Maximum confidence threshold
        2. Entropy-based uncertainty
        3. Distribution balance check
        """
        max_confidence = np.max(predictions)
        entropy = self.calculate_entropy(predictions)
        
        # Normalisasi entropy (untuk 4 kelas, max entropy ≈ 1.386)
        max_entropy = np.log(len(self.class_names))
        normalized_entropy = entropy / max_entropy
        
        # Check 1: Maximum confidence harus di atas threshold
        confidence_check = max_confidence >= self.confidence_threshold
        
        # Check 2: Entropy tidak boleh terlalu tinggi (prediksi terlalu ambiguous)
        uncertainty_check = normalized_entropy <= self.uncertainty_threshold
        
        # Check 3: Gap antara prediksi tertinggi dan kedua tertinggi
        sorted_preds = np.sort(predictions)[::-1]  # Sort descending
        confidence_gap = sorted_preds[0] - sorted_preds[1]
        gap_check = confidence_gap >= 0.2  # Gap minimum 20%
        
        # Kombinasi semua check
        is_valid = confidence_check and uncertainty_check and gap_check
        
        return {
            'is_valid': is_valid,
            'max_confidence': max_confidence,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'confidence_gap': confidence_gap,
            'checks': {
                'confidence': confidence_check,
                'uncertainty': uncertainty_check,
                'gap': gap_check
            }
        }

    def predict_image(self, image):
        if self.model is None:
            raise ValueError("Model belum dilatih!")

        # Jika input adalah PIL Image, konversi langsung
        if isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            img = Image.open(image).convert('RGB')
            
        img = img.resize(self.input_shape[:2])
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predictions_flat = predictions[0]
        
        # Validasi prediksi menggunakan rejection logic
        validation_result = self.is_valid_prediction(predictions_flat)
        
        predicted_class_index = np.argmax(predictions_flat)
        confidence = predictions_flat[predicted_class_index]

        if validation_result['is_valid']:
            # Prediksi valid - return hasil normal
            predicted_class = self.class_names[predicted_class_index]
            manfaat = self.manfaat_database[predicted_class]
            
            result = {
                'status': 'valid',
                'predicted_class': predicted_class,
                'confidence': confidence,
                'manfaat': manfaat,
                'all_predictions': predictions_flat,
                'validation_details': validation_result
            }
        else:
            # Prediksi tidak valid - reject
            result = {
                'status': 'rejected',
                'predicted_class': None,
                'confidence': confidence,
                'manfaat': None,
                'all_predictions': predictions_flat,
                'validation_details': validation_result,
                'rejection_reason': self._get_rejection_reason(validation_result)
            }

        return result

    def _get_rejection_reason(self, validation_result):
        """Generate human-readable rejection reason"""
        reasons = []
        
        if not validation_result['checks']['confidence']:
            reasons.append(f"Confidence terlalu rendah ({validation_result['max_confidence']:.1%} < {self.confidence_threshold:.1%})")
        
        if not validation_result['checks']['uncertainty']:
            reasons.append(f"Prediksi terlalu ambiguous (entropy: {validation_result['normalized_entropy']:.2f})")
        
        if not validation_result['checks']['gap']:
            reasons.append(f"Gap confidence terlalu kecil ({validation_result['confidence_gap']:.1%})")
        
        return " | ".join(reasons)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def set_thresholds(self, confidence_threshold=None, uncertainty_threshold=None):
        """Update threshold values"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if uncertainty_threshold is not None:
            self.uncertainty_threshold = uncertainty_threshold

def inject_custom_css():
    """Inject custom CSS for styling"""
    st.markdown("""
    <style>
        /* Hide Streamlit default elements */
        .stDeployButton {display:none;}
        .stDecoration {display:none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom navbar styling */
        .custom-navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            color: white;
            text-decoration: none;
        }
        
        .navbar-brand h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
        }
        
        .navbar-nav {
            display: flex;
            align-items: center;
            gap: 2rem;
        }
        
        .nav-item {
            color: white;
            text-decoration: none;
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.3s ease;
            border: none;
            background: none;
        }
        
        .nav-item:hover {
            background-color: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
        }
        
        .nav-item.active {
            background-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Main content adjustment */
        .block-container {
            margin-top: 80px;
            padding-top: 1rem;
        }
        
        /* Card styling */
        .info-card {
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 1rem 0;
        }
        
        .result-card {
            background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin: 1rem 0;
            text-align: center;
        }
        
        .rejected-card {
            background: linear-gradient(135deg, #FFE8E8 0%, #FFCDD2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #f44336;
            margin: 1rem 0;
            text-align: center;
        }
        
        .benefit-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #28a745;
            margin: 1rem 0;
        }
        
        .threshold-card {
            background: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            border-left: 3px solid #ffc107;
            margin: 1rem 0;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .custom-navbar {
                padding: 0 1rem;
                height: 60px;
            }
            
            .navbar-brand h1 {
                font-size: 1.2rem;
            }
            
            .navbar-nav {
                gap: 1rem;
            }
            
            .nav-item {
                padding: 0.3rem 0.6rem;
                font-size: 0.9rem;
            }
            
            .block-container {
                margin-top: 70px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_navbar():
    """Create navbar using Streamlit components"""
    st.markdown("""
    <div class="custom-navbar">
        <div class="navbar-brand">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.17 22L6.58 17.58C8.13 15.45 10.23 13.4 13 12.5C13.91 12.22 14.87 12.03 15.87 11.94C19.25 11.61 22.31 13.19 22.97 14.4C23.14 14.7 23.13 15.05 22.94 15.34C22.75 15.63 22.4 15.82 22.04 15.82C21.17 15.82 20.25 15.26 19.1 14.5C18.22 13.94 17.2 13.3 16 13.3C14.93 13.3 14.22 13.94 14.22 14.7C14.22 15.46 14.93 16.1 16 16.1C17.2 16.1 18.22 15.46 19.1 14.9C20.25 14.14 21.17 13.58 22.04 13.58C22.4 13.58 22.75 13.77 22.94 14.06C23.13 14.35 23.14 14.7 22.97 15C22.31 16.21 19.25 17.79 15.87 17.46C14.87 17.37 13.91 17.18 13 16.9C10.23 16 8.13 13.95 6.58 11.82L5.17 7.4L3.82 2.66C5.9 7.83 8 14 17 8Z" fill="white"/>
            </svg>
            <h1>Tanaman Herbal</h1>
        </div>
        <div class="navbar-nav">
            <span class="nav-item active">🏠 Home</span>
            <a href="https://github.com/fathanbimo/Tanaman-Herbal" target="_blank" class="nav-item">📁 GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title="Klasifikasi Tanaman Herbal Indonesia",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject CSS dan create navbar
    inject_custom_css()
    create_navbar()

    # Header dengan styling yang lebih baik
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="color: #2E7D32; margin-bottom: 0.5rem; font-size: 2.5rem;">
            🌿 Klasifikasi Tanaman Herbal 
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Info card
    st.markdown("""
    <div class="info-card">
        <p style="margin: 0; color: #2E7D32; text-align: center;">
            📋 <strong>Untuk menggunakan aplikasi ini:</strong><br>
            Silakan download model terlebih dahulu di 
            <a href="https://github.com/fathanbimo/Tanaman-Herbal" target="_blank" 
               style="color: #2E7D32; text-decoration: none; font-weight: bold;">
               🔗 GitHub Repository kami
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Inisialisasi session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = HerbalPlantClassifier()

    # Tampilkan halaman prediksi
    prediction_page()

def prediction_page():
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar for threshold adjustment
    with st.sidebar:
        st.markdown("###Pengaturan Threshold")
        st.markdown("""
        <div class="threshold-card">
            <p><strong>Atur sensitivity detection:</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="Semakin tinggi, semakin ketat dalam menerima prediksi"
        )
        
        uncertainty_threshold = st.slider(
            "Maximum Uncertainty Threshold", 
            min_value=0.1,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Semakin rendah, semakin ketat dalam menolak prediksi ambiguous"
        )
        
        # Update thresholds
        st.session_state.classifier.set_thresholds(
            confidence_threshold=confidence_threshold,
            uncertainty_threshold=uncertainty_threshold
        )
        
        st.markdown("---")
        st.markdown("**Status:**")
        st.info(f"Min Confidence: {confidence_threshold:.0%}")
        st.info(f"Max Uncertainty: {uncertainty_threshold:.0%}")
    
    # Section 1: Upload Model
    with st.container():
        st.markdown("Upload file model yang sudah dilatih (.h5 atau .keras)")
        
        model_file = st.file_uploader(
            "Pilih file model:",
            type=['h5', 'keras'],
            key="model_upload",
            help="Upload file model TensorFlow/Keras yang sudah dilatih"
        )
    
    if model_file is not None:
        # Simpan model ke temporary file
        suffix = os.path.splitext(model_file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(model_file.read())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner("Memuat model..."):
                st.session_state.classifier.load_model(tmp_file_path)
            st.success("✅ Model berhasil dimuat!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return

    st.divider()

    # Section 2: Upload Image
    with st.container():
        st.markdown("Upload gambar tanaman herbal yang ingin diidentifikasi")
        
        uploaded_image = st.file_uploader(
            "Pilih gambar:",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="image_upload",
            help="Format yang didukung: JPG, JPEG, PNG, BMP"
        )

    if uploaded_image is not None:
        # Layout untuk gambar dan hasil
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📸 Gambar Input")
            image = Image.open(uploaded_image)
            st.image(
                image, 
                use_container_width=True, 
                caption="Gambar tanaman herbal yang akan dianalisis"
            )

        with col2:
            st.markdown("### Hasil Analisis")
            
            if st.session_state.classifier.model is not None:
                try:
                    with st.spinner("🔄 Menganalisis gambar..."):
                        time.sleep(1)  # Simulasi processing time
                        result = st.session_state.classifier.predict_image(image)
                    
                    if result['status'] == 'valid':
                        # Hasil prediksi valid
                        predicted_class = result['predicted_class'].replace('_', ' ').title()
                        confidence = result['confidence']
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h2 style="color: #2E7D32; margin: 0 0 0.5rem 0;">🌿 {predicted_class}</h2>
                            <p style="color: #2E7D32; margin: 0; font-size: 1.3rem;">
                                <strong>Tingkat Kepercayaan: {confidence:.1%}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar
                        st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
                        
                    else:
                        # Hasil prediksi ditolak
                        st.markdown(f"""
                        <div class="rejected-card">
                            <h2 style="color: #d32f2f; margin: 0 0 0.5rem 0;">Gambar Tidak Dikenali</h2>
                            <p style="color: #d32f2f; margin: 0; font-size: 1.1rem;">
                                <strong>Gambar ini bukan salah satu dari 4 tanaman herbal yang dapat diidentifikasi</strong>
                            </p>
                            <hr style="margin: 1rem 0; border-color: #d32f2f;">
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("⚠️ Silakan upload gambar daun sirsak, daun sirih, daun jambu biji, atau lidah buaya yang lebih jelas.")
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {e}")
                    return
            else:
                st.warning("⚠️ Model belum dimuat. Upload model terlebih dahulu.")
                return

        # Section 3: Manfaat (hanya tampil untuk prediksi valid)
        if uploaded_image is not None and st.session_state.classifier.model is not None and result['status'] == 'valid':
            st.divider()
            
            predicted_class = result['predicted_class'].replace('_', ' ').title()
            st.markdown("## Manfaat Kesehatan")
            st.markdown(f"""
            <div class="benefit-card">
                <h4 style="color: #2E7D32; margin-top: 0;">Manfaat {predicted_class}:</h4>
                {result['manfaat'].replace('•', '<br>• ')}
            </div>
            """, unsafe_allow_html=True)
        
        # Section 4: Detail Prediksi (tampil untuk semua hasil)
        if uploaded_image is not None and st.session_state.classifier.model is not None:
            with st.expander("Detail Prediksi Semua Kelas", expanded=False):
                st.markdown("**Probabilitas untuk setiap kelas:**")
                
                for i, class_name in enumerate(st.session_state.classifier.class_names):
                    class_display = class_name.replace('_', ' ').title()
                    probability = result['all_predictions'][i]
                    
                    # Warna berdasarkan probabilitas
                    if probability > 0.7:
                        color = "#4CAF50"  # Hijau
                        icon = "🟢"
                    elif probability > 0.3:
                        color = "#FF9800"  # Orange
                        icon = "🟡"
                    else:
                        color = "#9E9E9E"  # Abu-abu
                        icon = "⚪"
                    
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"{icon} **{class_display}**")
                        st.progress(float(probability))
                    with col_b:
                        st.markdown(f"<span style='color: {color}; font-weight: bold; font-size: 1.1rem;'>{probability:.1%}</span>", 
                                  unsafe_allow_html=True)

    # Section 5: Info Aplikasi
    st.divider()
    with st.expander("Tentang Aplikasi", expanded=False):
        st.markdown("""
        ## 🌿 Aplikasi Klasifikasi Tanaman Herbal
        
        Aplikasi ini menggunakan teknologi **Deep Learning** dengan arsitektur **Convolutional Neural Network (CNN)** 
        untuk mengidentifikasi 4 jenis tanaman herbal Indonesia yang umum digunakan.
        
        Tanaman yang Dapat Diidentifikasi:
        
        | Tanaman | Nama Ilmiah | Kegunaan Utama |
        |---------|-------------|----------------|
        |**Daun Jambu Biji** | *Psidium guajava* | Mengontrol diabetes, gangguan pencernaan |
        |**Daun Sirih** | *Piper betle* | Kesehatan mulut, antiseptik alami |
        |**Daun Sirsak** | *Annona muricata* | Antioksidan, anti-inflamasi |
        |**Lidah Buaya** | *Aloe vera* | Perawatan kulit, gangguan pencernaan |
        
        Petunjuk Penggunaan:
        1. **Download Model**: Unduh file model dari [GitHub Repository](https://github.com/fathanbimo/Tanaman-Herbal)
        2. **Upload Model**: Upload file model (.h5 atau .keras) ke aplikasi
        3. **Upload Gambar**: Pilih gambar tanaman herbal yang jelas dan berkualitas baik
        4. **Lihat Hasil**: Analisis otomatis akan menampilkan hasil atau menolak jika tidak sesuai
        
        Tips untuk Hasil Terbaik:
        - Gunakan gambar dengan pencahayaan yang baik
        - Pastikan daun terlihat jelas dan tidak buram
        - Hindari gambar dengan background yang terlalu ramai
        - Resolusi gambar minimal 224x224 pixel
        - Upload gambar yang benar-benar merupakan salah satu dari 4 tanaman target
        
        ### ⚠️ Disclaimer:
        > **PENTING**: Aplikasi ini dibuat untuk tujuan **edukasi dan penelitian**. 
        > Hasil prediksi tidak dapat menggantikan konsultasi dengan ahli herbal atau tenaga medis profesional.
        > Selalu konsultasikan penggunaan tanaman herbal dengan ahli yang kompeten.
        """)

if __name__ == "__main__":
    main()
