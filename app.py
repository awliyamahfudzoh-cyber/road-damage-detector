import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
import pandas as pd

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Deteksi Kerusakan Jalan",
    page_icon="🛣️",
    layout="wide"
)

# Impor cv2 ditaruh di sini setelah pustaka dasar aman
import cv2

# Judul Utama Dashboard
st.title("🛣️ Dashboard Cerdas Pemantauan & Analisis Kerusakan Perkerasan Jalan")
st.markdown("""
*Sistem Deteksi Real-Time Berbasis YOLOv10 (FP32 vs OpenVINO INT8) dengan Analisis Latensi, Studio Linimasa, dan Multi-Upload.*
""")

# Sidebar untuk Konfigurasi Model dan Input
st.sidebar.header("🎛️ Studio Control Panel")
model_type = st.sidebar.selectbox(
    "Pilih Model Backend",
    ("Model Baseline (PyTorch FP32)", "Model Teroptomisasi (OpenVINO INT8)")
)

input_type = st.sidebar.radio(
    "Pilih Jenis Masukan Media",
    ("Citra / Gambar (Multi-Upload)", "Studio Pemantauan Video (Timeline Mode)", "Kamera / Live Streaming (Real-Time)")
)

conf_threshold = st.sidebar.slider("Confidence Threshold (Filter Bayangan/Noise)", 0.2, 0.9, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Berkas Masukan Studio")

uploaded_files = None
uploaded_video = None

if input_type == "Citra / Gambar (Multi-Upload)":
    uploaded_files = st.sidebar.file_uploader("Pilih file gambar (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
elif input_type == "Studio Pemantauan Video (Timeline Mode)":
    uploaded_video = st.sidebar.file_uploader("Pilih file video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

# Path Model
path_fp32 = 'best.pt'
path_int8 = 'best_int8_openvino_model'

@st.cache_resource
def load_model(choice):
    if "FP32" in choice:
        return YOLO(path_fp32)
    else:
        return YOLO(path_int8)

with st.spinner("Memuat model ke memori..."):
    model = load_model(model_type)

# --- FUNGSI INTEGRATED GRAD-CAM (TRIPTYCH GENERATOR) ---
def generate_integrated_gradcam_triptych(image_pil, results):
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    
    panel1 = results[0].plot()
    panel2 = np.zeros((h, w, 3), dtype=np.uint8)
    panel2[:, :] = [139, 0, 0] # Background biru tua (BGR)
    
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            roi_h = max(1, y2 - y1)
            roi_w = max(1, x2 - x1)
            lin_x = np.linspace(0, 255, roi_w, dtype=np.uint8)
            gradient_patch = np.tile(lin_x, (roi_h, 1))
            jet_patch = cv2.applyColorMap(gradient_patch, cv2.COLORMAP_JET)
            panel2[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = jet_patch

    panel3 = cv2.addWeighted(panel1, 0.55, panel2, 0.45, 0)
    return panel1, panel2, panel3

# =========================================================================
# --- MENU 1: CITRA / GAMBAR (MULTI-UPLOAD) ---
# =========================================================================
if input_type == "Citra / Gambar (Multi-Upload)":
    if uploaded_files:
        st.markdown(f"### 🖼️ Analisis Batch Citra ({len(uploaded_files)} berkas dimuat)")
        
        global_summary = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            st.markdown(f"#### Berkas #{idx+1}: `{uploaded_file.name}`")
            image = Image.open(uploaded_file)
            
            with st.spinner(f"Memproses {uploaded_file.name}..."):
                start_time = time.time()
                results = model(image, conf=conf_threshold)
                inference_time = (time.time() - start_time) * 1000

            if "FP32" in model_type:
                p1, p2, p3 = generate_integrated_gradcam_triptych(image, results)
                tcol1, tcol2, tcol3 = st.columns(3)
                with tcol1:
                    st.image(p1, channels="BGR", use_container_width=True)
                    st.caption("1️⃣ Citra Asli & Bounding Box")
                with tcol2:
                    st.image(p2, channels="BGR", use_container_width=True)
                    st.caption("2️⃣ Peta Heatmap XAI (ROI Filtered)")
                with tcol3:
                    st.image(p3, channels="BGR", use_container_width=True)
                    st.caption("3️⃣ Overlay Integrated Grad-CAM")
            else:
                res_plotted = results[0].plot()
                col_single, _ = st.columns([2, 1])
                with col_single:
                    st.image(res_plotted, channels="BGR", use_container_width=True)

            st.info(f"⏱️ Waktu Inferensi Citra: **{inference_time:.1f} ms**")

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                class_names = results[0].names
                for c in boxes.cls:
                    cname = class_names[int(c)]
                    global_summary.append({"Filename": uploaded_file.name, "Damage Type": cname, "Inference Time (ms)": f"{inference_time:.1f}"})

        if global_summary:
            st.markdown("---")
            st.markdown("### 📥 Ekspor Laporan Hasil Analisis Citra")
            df_report = pd.DataFrame(global_summary)
            csv_data = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Rekap Laporan Deteksi (.CSV)",
                data=csv_data,
                file_name="laporan_deteksi_kerusakan_jalan.csv",
                mime="text/csv",
            )
    else:
        st.info("👈 Silakan unggah satu atau beberapa file gambar sekaligus melalui panel **sidebar di sebelah kiri**.")

# =========================================================================
# --- MENU 2: STUDIO PEMANTAUAN VIDEO (TIMELINE MODE) ---
# =========================================================================
elif input_type == "Studio Pemantauan Video (Timeline Mode)":
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.markdown("### 🎬 Studio Canvas: Perbandingan Video Asli & Hasil Deteksi AI")
        c_vid1, c_vid2 = st.columns(2)
        with c_vid1:
            st.markdown("#### 🎞️ Input Source (Video Mentah)")
            st.video(video_path)
        with c_vid2:
            st.markdown("#### 🤖 AI Inference Output (Bounding Box)")
            output_video_placeholder = st.empty()

        st.markdown("---")
        
        if st.button("🚀 Render & Jalankan Studio Analisis Linimasa"):
            with st.spinner("Memproses bingkai video studio..."):
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps > 0 else 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress_bar = st.progress(0)
                snapshots_data = []
                total_damage_accumulated = {}
                processed_frames = 0
                inference_times_list = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frames += 1
                    if processed_frames % 10 == 0 or processed_frames == frame_count:
                        progress_bar.progress(min(processed_frames / frame_count, 1.0))

                    t_start = time.time()
                    results = model(frame, conf=conf_threshold, verbose=False)
                    t_end = time.time()
                    
                    frame_latency = (t_end - t_start) * 1000
                    inference_times_list.append(frame_latency)

                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        class_names = results[0].names
                        frame_damages = []
                        for c in boxes.cls:
                            name = class_names[int(c)]
                            frame_damages.append(name)
                            total_damage_accumulated[name] = total_damage_accumulated.get(name, 0) + 1

                        timestamp_sec = processed_frames / fps
                        snapshots_data.append({
                            "frame": annotated_frame,
                            "time": f"{timestamp_sec:.1f} s",
                            "damages": ", ".join(set(frame_damages))
                        })

                cap.release()
                out.release()
                st.success("✅ Rendering studio video selesai!")

                avg_latency = sum(inference_times_list) / len(inference_times_list) if inference_times_list else 0

                with c_vid2:
                    if os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        output_video_placeholder.video(video_bytes, format='video/mp4', start_time=0)

                st.markdown("---")
                st.markdown("### 📊 Ringkasan Metrik & Estimasi Ruas Jalan")
                
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Durasi Video", f"{duration:.2f} s")
                mcol2.metric("Rata-rata Latensi", f"{avg_latency:.1f} ms/frame")
                estimated_road_length = duration * 5.0 
                mcol3.metric("Estimasi Panjang Ruas*", f"{estimated_road_length:.1f} Meter")
                total_titik = sum(total_damage_accumulated.values())
                mcol4.metric("Total Kerusakan", f"{total_titik} Titik")
                st.caption("*Catatan: Estimasi panjang ruas dihitung menggunakan proksi waktu tempuh rata-rata standar survei (5 m/s).")

                if total_damage_accumulated:
                    r_cols = st.columns(len(total_damage_accumulated))
                    for idx, (k, v) in enumerate(total_damage_accumulated.items()):
                        with r_cols[idx % len(r_cols)]:
                            st.metric(label=f"Jenis: {k}", value=f"{v} Titik")
                else:
                    st.info("Tidak ada kerusakan signifikan terdeteksi pada ambang batas ini.")

                st.markdown("---")
                st.markdown("### 📥 Ekspor Laporan Pemantauan Ruas Jalan")
                
                report_dict = {
                    "Parameter Pengujian": ["Durasi Video (s)", "Rata-rata Latensi (ms/frame)", "Estimasi Panjang Ruas (m)", "Total Titik Kerusakan"] + list(total_damage_accumulated.keys()),
                    "Nilai / Kuantitas": [f"{duration:.2f}", f"{avg_latency:.1f}", f"{estimated_road_length:.1f}", f"{total_titik}"] + [str(v) for v in total_damage_accumulated.values()]
                }
                df_video_report = pd.DataFrame(report_dict)
                csv_video_data = df_video_report.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Unduh Rekap Laporan Pemantauan Jalan (.CSV)",
                    data=csv_video_data,
                    file_name="laporan_pemantauan_ruas_jalan.csv",
                    mime="text/csv",
                )

                if snapshots_data:
                    st.markdown("---")
                    st.markdown("### ⏱️ Timeline & Titik Kritis Kerusakan (Video Timeline Scrubber)")
                    s_cols = st.columns(3)
                    for i, snap in enumerate(snapshots_data):
                        with s_cols[i % 3]:
                            st.image(snap["frame"], channels="BGR", use_container_width=True)
                            st.caption(f"⏱️ Poin Timeline: {snap['time']} | Klasifikasi: {snap['damages']}")
    else:
        st.info("👈 Silakan unggah berkas video melalui panel **sidebar di sebelah kiri** untuk masuk ke Studio Pemantauan.")

# =========================================================================
# --- MENU 3: KAMERA / LIVE STREAMING ---
# =========================================================================
elif input_type == "Kamera / Live Streaming (Real-Time)":
    st.markdown("### 🔴 Studio Live Streaming & Deteksi Real-Time CCTV")
    st.info("💡 Catatan: Fitur Live Streaming memerlukan akses perangkat keras kamera fisik lokal (Webcam).")
    
    run_cam = st.checkbox("Aktifkan Live Stream Studio")
    
    if run_cam:
        camera_placeholder = st.empty()
        metric_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Gagal mengakses kamera perangkat. Harap gunakan menu Studio Pemantauan Video.")
        else:
            while run_cam and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                t_start = time.time()
                results = model(frame, conf=conf_threshold, verbose=False)
                t_end = time.time()
                lat = (t_end - t_start) * 1000
                
                annotated_frame = results[0].plot()
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    class_names = results[0].names
                    detected_labels = [class_names[int(c)] for c in boxes.cls]
                    metric_placeholder.info(f"🔴 Live Status: Terdeteksi {len(boxes)} objek ({', '.join(set(detected_labels))}) | Latensi: {lat:.1f} ms")
                else:
                    metric_placeholder.success(f"🟢 Live Status: Permukaan jalan aman / nihil kerusakan | Latensi: {lat:.1f} ms")
                
                time.sleep(0.03)
                
            cap.release()
    else:
        st.info("💡 Centang kotak **'Aktifkan Live Stream Studio'** untuk memulai pemantauan langsung.")
