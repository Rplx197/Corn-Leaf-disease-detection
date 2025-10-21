import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Deploy Model Klasifikasi Gambar",
    # page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk memperlebar sidebar
st.markdown("""
<style>
    /* Perlebar sidebar */
    .css-1d391kg {
        width: 400px;
        min-width: 400px;
        max-width: 400px;
    }
    
    /* Atur lebar main content */
    .css-1v0mbdj {
        margin-left: 400px;
    }
    
    /* Responsif untuk mobile */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 300px;
            min-width: 300px;
            max-width: 300px;
        }
        .css-1v0mbdj {
            margin-left: 300px;
        }
    }
    
    /* Style tambahan untuk sidebar */
    .sidebar .sidebar-content {
        width: 400px;
    }
    
    /* Perbaikan tampilan elemen dalam sidebar */
    .stSidebar {
        min-width: 400px;
    }
    
    /* Style untuk file uploader yang lebih lebar */
    .stFileUploader {
        min-width: 350px;
    }
    
    /* Style untuk selectbox dan text input yang lebih lebar */
    .stSelectbox, .stTextInput {
        min-width: 350px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model
@st.cache_resource
def load_custom_model(model_path):
    """Memuat model dengan caching untuk performa lebih baik"""
    try:
        model = load_model(model_path)
        st.sidebar.success(f"✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Pastikan file model ada di folder 'models/'")
        return None

# Fungsi untuk preprocess gambar TANPA OpenCV
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess gambar menggunakan PIL dan numpy saja"""
    try:
        # Convert ke RGB jika perlu
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize gambar
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert ke numpy array
        img_array = np.array(img_resized)
        
        # Normalisasi
        img_array = img_array.astype('float32') / 255.0
        
        # Expand dimensions untuk batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Fungsi untuk melakukan prediksi
def predict_image(model, processed_img, class_names):
    """Melakukan prediksi pada gambar yang sudah dipreprocess"""
    try:
        # Lakukan prediksi
        predictions = model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # FIX: Handle class names dengan aman
        if class_names and len(class_names) > predicted_class:
            predicted_label = class_names[predicted_class]
        else:
            predicted_label = f"Class {predicted_class}"
        
        return predicted_label, confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# Main apps
def main():
    # Header aplikasi
    st.title("Klasifikasi Penyakit Daun Jagung")
    st.markdown("---")
    
    # Sidebar untuk konfigurasi
    st.sidebar.title("Konfigurasi Model")
    
    # Pilihan model
    st.sidebar.subheader("Pilih Model")
    model_option = st.sidebar.selectbox(
        "Model Klasifikasi:",
        ["InceptionV3", "ResNet50V2"],
        help="Pilih model yang akan digunakan untuk prediksi"
    )
    
    # Mapping model name to file path
    model_paths = {
        "InceptionV3": "Corn_InceptionV3_Final.h5",
        "ResNet50V2": "Corn_ResNet50V2_Final.h5"
    }
    
    # Load model berdasarkan pilihan
    model_path = model_paths[model_option]
    model = load_custom_model(model_path)
    
    # Informasi model
    st.sidebar.markdown(f"**Model Terpilih:** `{model_option}`")
    
    # Upload gambar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Gambar")
    
    uploaded_file = st.sidebar.file_uploader(
        "Pilih gambar untuk diklasifikasi:",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload gambar daun jagung untuk diklasifikasi",
        label_visibility="collapsed"
    )
    
    # Konfigurasi class names
    st.sidebar.markdown("---")
    st.sidebar.subheader("Kelas Klasifikasi")
    
    class_names_input = st.sidebar.text_input(
        "Nama Kelas:",
        value="Blight, Common Rust, Gray Spot, Healthy",
        disabled=True
    )
    
    # FIX: Handle empty class names
    class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]
    
    # Informasi kelas
    st.sidebar.markdown("""
    **Deskripsi Kelas:**
    - **Blight**: Hawar daun
    - **Common Rust**: Karat daun
    - **Gray Spot**: Bercak abu-abu
    - **Healthy**: Sehat
    """)
    
    # Jika tidak ada class names, buat default
    if not class_names:
        class_names = [f"Class {i}" for i in range(4)]
    
    
    # Kolom utama
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Gambar")
        
        if uploaded_file is not None:
            # Tampilkan gambar yang diupload
            try:
                image_display = Image.open(uploaded_file)
                
                # FIX: Gunakan use_container_width bukan use_column_width
                st.image(image_display, caption="Gambar yang diupload", use_container_width=True)
                
                # Informasi gambar
                st.info(f"""
                **Informasi Gambar:**
                - Nama File: {uploaded_file.name}
                - Format: {uploaded_file.type}
                - Ukuran: {image_display.size}
                - Mode: {image_display.mode}
                """)
                
                # Tombol prediksi
                if st.button("🚀 Lakukan Prediksi", type="primary", use_container_width=True):
                    if model is not None:
                        with st.spinner('Melakukan prediksi...'):
                            # Preprocess gambar
                            processed_img = preprocess_image(image_display)
                            
                            if processed_img is not None:
                                # FIX: Pastikan class_names selalu ada dan valid
                                predicted_label, confidence, all_predictions = predict_image(
                                    model, processed_img, class_names
                                )
                                
                                if predicted_label is not None and all_predictions is not None:
                                    # Tampilkan hasil prediksi di kolom 2
                                    with col2:
                                        st.subheader("Hasil Prediksi")
                                        st.success("Prediksi Berhasil!")
                                        
                                        # Tampilkan hasil utama
                                        col_metric1, col_metric2 = st.columns([2, 1])
                                        with col_metric1:
                                            st.metric(
                                                label="Hasil Prediksi",
                                                value=predicted_label,
                                                delta=f"{confidence:.2%}"
                                            )
                                        
                                        with col_metric2:
                                            st.metric(
                                                label="Confidence",
                                                value=f"{confidence:.2%}"
                                            )
                                        
                                        # Progress bar untuk confidence
                                        st.progress(float(confidence))
                                        st.caption(f"Tingkat Kepercayaan: {confidence:.2%}")
                                        
                                        # Visualisasi tambahan
                                        st.subheader("Distribusi Probabilitas")
                                        
                                        # FIX: Buat chart data dengan aman
                                        chart_data = {}
                                        for i, prob in enumerate(all_predictions):
                                            # Pastikan index tidak melebihi jumlah class names
                                            if i < len(class_names):
                                                class_name = class_names[i]
                                            else:
                                                class_name = f"Class {i}"
                                            chart_data[class_name] = float(prob)
                                        
                                        st.bar_chart(chart_data)
                                        
                                        # Tampilkan tabel probabilitas
                                        st.subheader("Detail Probabilitas")
                                        prob_data = []
                                        for i, prob in enumerate(all_predictions):
                                            # FIX: Handle index out of range
                                            if i < len(class_names):
                                                class_name = class_names[i]
                                            else:
                                                class_name = f"Class {i}"
                                            
                                            prob_data.append({
                                                "Kelas": class_name,
                                                "Probabilitas": f"{prob:.4f}",
                                                "Persentase": f"{prob*100:.2f}%"
                                            })
                                        
                                        st.table(prob_data)
                                        
                                        # Tampilkan top 3 predictions
                                        st.subheader("Top 3 Predictions")
                                        top_indices = np.argsort(all_predictions)[-3:][::-1]
                                        
                                        for rank, idx in enumerate(top_indices):
                                            prob = all_predictions[idx]
                                            if idx < len(class_names):
                                                class_name = class_names[idx]
                                            else:
                                                class_name = f"Class {idx}"
                                            
                                            st.write(f"{rank + 1}. **{class_name}**: {prob*100:.2f}%")
                                            
                                else:
                                    st.error("❌ Gagal melakukan prediksi")
                            else:
                                st.error("❌ Gagal memproses gambar")
                    else:
                        st.error("❌ Model belum dimuat dengan benar")
            except Exception as e:
                st.error(f"Error memproses gambar: {e}")
        else:
            st.info("👆 Silakan upload gambar melalui sidebar di sebelah kiri")
            
            # Contoh placeholder
            st.markdown("""
            ### Panduan Penggunaan:
            1. **Pilih model** yang ingin digunakan di sidebar
            2. **Upload gambar** melalui file uploader di sidebar
            3. **Klik tombol 'Lakukan Prediksi'** untuk melihat hasil
            
            ### Contoh Gambar:
            - Gambar daun jagung yang jelas
            - Background netral
            - Pencahayaan cukup
            - Fokus pada area yang sakit (jika ada)
            """)
    
    with col2:
        if not uploaded_file:
            st.subheader("Hasil Prediksi")
            st.info("ℹ️ Hasil prediksi akan ditampilkan di sini setelah upload gambar dan klik tombol prediksi")
            
            # Placeholder untuk contoh output
            st.markdown("""
            ### Contoh Output:
            - **Prediksi Berhasil!**
            - **Hasil:** Blight (95.2%)
            - **Distribusi probabilitas** semua kelas
            - **Tabel detail** probabilitas per kelas
            - **Top 3 Predictions**
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Dibuat dengan menggunakan Streamlit dan TensorFlow | Aplikasi Klasifikasi Penyakit Daun Jagung</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Menjalankan aplikasi
if __name__ == "__main__":
    main()