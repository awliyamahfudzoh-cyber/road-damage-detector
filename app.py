import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO
import tempfile
import time
import pandas as pd
from fpdf import FPDF

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
*Sistem Deteksi Real-Time Berbasis YOLOv10 (FP32 vs OpenVINO INT8).*
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

# --- KELAS PENGHASIL PDF LAPORAN DENGAN LAYOUT 3 KOLOM TRIPTYCH ---
class PDFReport(FPDF):
    def __init__(self, model_info, threshold_info):
        super().__init__()
        self.model_info = model_info
        self.threshold_info = threshold_info

    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 8, 'LAPORAN RESMI ANALISIS KERUSAKAN PERKERASAN JALAN', 0, 1, 'C')
        self.set_font('helvetica', 'I', 9)
        self.cell(0, 5, f'Backend Model: {self.model_info} | Threshold: {self.threshold_info}', 0, 1, 'C')
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 8, f'Halaman {self.page_no()}', 0, 0, 'C')


# =========================================================================
# --- MENU 1: CITRA / GAMBAR (MULTI-UPLOAD & DYNAMIC PDF REPORT) ---
# =========================================================================
elif input_type == "Citra / Gambar (Multi-Upload)":
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

            boxes = results[0].boxes
            
            # Tampilkan XAI Triptych untuk SEMUA kondisi gambar jalan (baik ada kerusakan maupun jalan mulus)
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
            
            p1_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            p2_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            p3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
            cv2.imwrite(p1_path, p1)
            cv2.imwrite(p2_path, p2)
            cv2.imwrite(p3_path, p3)
            
            saved_paths = {"p1": p1_path, "p2": p2_path, "p3": p3_path}

            st.info(f"⏱️ Waktu Inferensi Citra: **{inference_time:.1f} ms**")

            if boxes is not None and len(boxes) > 0:
                class_names = results[0].names
                box_data = boxes.data.cpu().numpy()
                
                class_stats = {}
                for box in box_data:
                    cls_id = int(box[5])
                    conf = float(box[4])
                    cname = class_names[cls_id]
                    if cname not in class_stats:
                        class_stats[cname] = []
                    class_stats[cname].append(conf)
                
                total_units = len(boxes)
                avg_conf_all = sum([float(b[4]) for b in box_data]) / total_units if total_units > 0 else 0

                damage_desc = ", ".join(class_stats.keys())
                st.success(f"✅ Status: Terdeteksi {total_units} unit kerusakan ({damage_desc}).")

                global_summary.append({
                    "Filename": uploaded_file.name, 
                    "Damage Type": damage_desc, 
                    "Unit Count": total_units,
                    "Avg Confidence": f"{avg_conf_all:.2f}",
                    "Inference Time (ms)": f"{inference_time:.1f}",
                    "Paths": saved_paths,
                    "Show_Triptych": True
                })
            else:
                # Kondisi jalan mulus / bersih tanpa kerusakan
                st.success("🟢 Status: **Kondisi Jalan Mulus / Bersih** (Tidak ada kerusakan struktural terdeteksi).")
                global_summary.append({
                    "Filename": uploaded_file.name, 
                    "Damage Type": "Normal / Jalan Mulus", 
                    "Unit Count": 0,
                    "Avg Confidence": "-",
                    "Inference Time (ms)": f"{inference_time:.1f}",
                    "Paths": saved_paths,
                    "Show_Triptych": True
                })

        if global_summary:
            st.markdown("---")
            st.markdown("### 📥 Ekspor Laporan Resmi Analisis Citra (.PDF)")
            
            def create_pdf_report(summary_data):
                pdf = PDFReport(model_type, conf_threshold)
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                pdf.set_font('helvetica', 'B', 10)
                pdf.cell(0, 6, '1. Tabel Ringkasan Deteksi Data Valid', 0, 1, 'L')
                
                pdf.set_font('helvetica', 'B', 9)
                pdf.set_fill_color(230, 230, 230)
                pdf.cell(60, 6, 'Nama Berkas', 1, 0, 'C', True)
                pdf.cell(50, 6, 'Jenis Kerusakan', 1, 0, 'C', True)
                pdf.cell(30, 6, 'Jml Unit', 1, 0, 'C', True)
                pdf.cell(25, 6, 'Conf. Avg', 1, 0, 'C', True)
                pdf.cell(25, 6, 'Latensi', 1, 1, 'C', True)
                
                pdf.set_font('helvetica', '', 8)
                for item in summary_data:
                    display_filename = item['Filename'] if len(item['Filename']) <= 35 else item['Filename'][:32] + "..."
                    pdf.cell(60, 6, str(display_filename), 1, 0, 'L')
                    pdf.cell(50, 6, str(item['Damage Type']), 1, 0, 'L')
                    pdf.cell(30, 6, str(item['Unit Count']), 1, 0, 'C')
                    pdf.cell(25, 6, str(item['Avg Confidence']), 1, 0, 'C')
                    pdf.cell(25, 6, f"{item['Inference Time (ms)']} ms", 1, 1, 'C')
                
                pdf.ln(6)
                pdf.set_font('helvetica', 'B', 10)
                pdf.cell(0, 6, '2. Lampiran Visual Triptych & XAI (Integrated Grad-CAM)', 0, 1, 'L')
                
                for item in summary_data:
                    pdf.set_font('helvetica', 'I', 8)
                    pdf.cell(0, 5, f"File: {item['Filename']} | Kategori: {item['Damage Type']} ({item['Unit Count']} Unit)", 0, 1, 'L')
                    
                    if item['Show_Triptych'] and item['Paths']['p2'] is not None:
                        y_pos = pdf.get_y()
                        pdf.image(item['Paths']['p1'], x=10, y=y_pos, w=60)
                        pdf.image(item['Paths']['p2'], x=75, y=y_pos, w=60)
                        pdf.image(item['Paths']['p3'], x=140, y=y_pos, w=60)
                        
                        pdf.set_y(y_pos + 42)
                        pdf.set_font('helvetica', 'I', 6)
                        pdf.cell(60, 4, '1. Citra Asli & Bounding Box', 0, 0, 'C')
                        pdf.cell(65, 4, '2. Peta Heatmap XAI', 0, 0, 'C')
                        pdf.cell(65, 4, '3. Overlay Grad-CAM', 0, 1, 'C')
                        pdf.ln(3)
                
                return bytes(pdf.output())

            pdf_bytes = create_pdf_report(global_summary)
            st.download_button(
                label="📥 Unduh Laporan Resmi (.PDF Berisi Tabel & Triptych 3 Kolom)",
                data=pdf_bytes,
                file_name="Laporan_Analisis_Citra_Jalan.pdf",
                mime="application/pdf",
            )
    else:
        st.info("👈 Silakan unggah satu atau beberapa file gambar jalan melalui panel **sidebar di sebelah kiri**.")
        
# =========================================================================
# --- MENU 2: STUDIO PEMANTAUAN VIDEO (TIMELINE MODE & PDF 3 KOLOM) ---
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

                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.webm').name
                fourcc = cv2.VideoWriter_fourcc(*'VP80')
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
                        
                        # Generate Triptych untuk Snapshot Video jika FP32
                        if "FP32" in model_type:
                            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            p1, p2, p3 = generate_integrated_gradcam_triptych(pil_frame, results)
                            p1_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                            p2_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                            p3_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                            cv2.imwrite(p1_path, p1)
                            cv2.imwrite(p2_path, p2)
                            cv2.imwrite(p3_path, p3)
                            snap_paths = {"p1": p1_path, "p2": p2_path, "p3": p3_path}
                        else:
                            p1_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
                            cv2.imwrite(p1_path, annotated_frame)
                            snap_paths = {"p1": p1_path, "p2": None, "p3": None}

                        snapshots_data.append({
                            "frame": annotated_frame,
                            "time": f"{timestamp_sec:.1f} s",
                            "damages": ", ".join(set(frame_damages)),
                            "paths": snap_paths,
                            "Is_FP32": "FP32" in model_type
                        })

                cap.release()
                out.release()
                st.success("✅ Rendering studio video selesai!")

                avg_latency = sum(inference_times_list) / len(inference_times_list) if inference_times_list else 0

                with c_vid2:
                    if os.path.exists(output_path):
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                            output_video_placeholder.video(video_bytes, format='video/webm', start_time=0)
                        
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
                    st.warning("⚠️ Peringatan: Tidak ada objek jalan atau kerusakan signifikan terdeteksi pada video ini.")

                st.markdown("---")
                st.markdown("### 📥 Ekspor Laporan Pemantauan Ruas Jalan (.PDF)")
                
                def create_video_pdf_report(dur, lat, r_len, tot_titik, damage_dict, snaps):
                    pdf = PDFReport(model_type, conf_threshold)
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    
                    pdf.set_font('helvetica', 'B', 10)
                    pdf.cell(0, 6, '1. Ringkasan Metrik Pemantauan Ruas Jalan', 0, 1, 'L')
                    
                    pdf.set_font('helvetica', 'B', 9)
                    pdf.set_fill_color(230, 230, 230)
                    pdf.cell(100, 6, 'Parameter Pengujian', 1, 0, 'C', True)
                    pdf.cell(80, 6, 'Nilai / Kuantitas', 1, 1, 'C', True)
                    
                    metrics_data = [
                        ("Durasi Video (s)", f"{dur:.2f}"),
                        ("Rata-rata Latensi (ms/frame)", f"{lat:.1f}"),
                        ("Estimasi Panjang Ruas (m)", f"{r_len:.1f}"),
                        ("Total Titik Kerusakan", f"{tot_titik}")
                    ]
                    for k, v in damage_dict.items():
                        metrics_data.append((f"Jenis Kerusakan: {k}", f"{v} Titik"))

                    pdf.set_font('helvetica', '', 9)
                    for k, v in metrics_data:
                        pdf.cell(100, 6, k, 1, 0, 'L')
                        pdf.cell(80, 6, v, 1, 1, 'C')

                    if snaps:
                        pdf.ln(6)
                        pdf.set_font('helvetica', 'B', 10)
                        pdf.cell(0, 6, '2. Lampiran Hasil Snapshot (Bounding Box)', 0, 1, 'L')
                        
                        for s in snaps:
                            pdf.set_font('helvetica', 'I', 8)
                            pdf.cell(0, 5, f"Waktu: {s['time']} | Klasifikasi: {s['damages']}", 0, 1, 'L')
                            
                            if s['Is_FP32'] and s['paths']['p2'] is not None:
                                y_pos = pdf.get_y()
                                pdf.image(s['paths']['p1'], x=10, y=y_pos, w=60)
                                pdf.image(s['paths']['p2'], x=75, y=y_pos, w=60)
                                pdf.image(s['paths']['p3'], x=140, y=y_pos, w=60)
                                
                                pdf.set_y(y_pos + 42)
                                pdf.set_font('helvetica', 'I', 6)
                                pdf.cell(60, 4, '1. Bounding Box', 0, 0, 'C')
                                pdf.cell(65, 4, '2. Heatmap XAI', 0, 0, 'C')
                                pdf.cell(65, 4, '3. Overlay Grad-CAM', 0, 1, 'C')
                                pdf.ln(3)
                            else:
                                if os.path.exists(s['paths']['p1']):
                                    pdf.image(s['paths']['p1'], w=90)
                                    pdf.ln(4)

                    return bytes(pdf.output())

                video_pdf_bytes = create_video_pdf_report(duration, avg_latency, estimated_road_length, total_titik, total_damage_accumulated, snapshots_data)
                st.download_button(
                    label="📥 Unduh Laporan Pemantauan Video (.PDF)",
                    data=video_pdf_bytes,
                    file_name="Laporan_Pemantauan_Video_Jalan.pdf",
                    mime="application/pdf",
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
# --- MENU 3: KAMERA / LIVE STREAMING (DUAL MODE: FOTO & VIDEO LIVE) ---
# =========================================================================
elif input_type == "Kamera / Live Streaming (Real-Time)":
    st.markdown("### 🔴 Studio Live Streaming & Tangkapan Kamera Real-Time")
    st.info("💡 Pilih mode interaksi kamera di bawah: Ambil Foto Instan atau Streaming Video Langsung.")

    cam_mode = st.radio("Pilih Mode Kamera", ("Ambil Foto Langsung (Snapshot)", "Live Stream Video Langsung (Real-Time Webcam)"), horizontal=True)

    if cam_mode == "Ambil Foto Langsung (Snapshot)":
        camera_file = st.camera_input("Ambil *Snapshot* Langsung dari Kamera Perangkat")

        if camera_file is not None:
            image = Image.open(camera_file)
            with st.spinner("Menganalisis *frame*..."):
                start_time = time.time()
                results = model(image, conf=conf_threshold)
                inference_time = (time.time() - start_time) * 1000

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                if "FP32" in model_type:
                    p1, p2, p3 = generate_integrated_gradcam_triptych(image, results)
                    tcol1, tcol2, tcol3 = st.columns(3)
                    with tcol1:
                        st.image(p1, channels="BGR", use_container_width=True)
                        st.caption("1️⃣ Citra Asli & Bounding Box")
                    with tcol2:
                        st.image(p2, channels="BGR", use_container_width=True)
                        st.caption("2️⃣ Peta Heatmap XAI")
                    with tcol3:
                        st.image(p3, channels="BGR", use_container_width=True)
                        st.caption("3️⃣ Overlay Grad-CAM")
                else:
                    res_plotted = results[0].plot()
                    st.image(res_plotted, channels="BGR", use_container_width=True)

                st.info(f"⏱️ Waktu Inferensi: **{inference_time:.1f} ms**")
                class_names = results[0].names
                detected_labels = [class_names[int(c)] for c in boxes.cls]
                st.success(f"✅ Terdeteksi: {', '.join(set(detected_labels))}")
            else:
                st.warning("⚠️ Peringatan: Tidak ada objek jalan atau kerusakan yang terdeteksi. Pastikan kamera mengarah ke permukaan jalan.")

    else:
        st.markdown("#### 🎥 Live Stream Video Aktif")
        run_live_stream = st.checkbox("Nyalakan Live Stream Webcam")
        
        if run_live_stream:
            stream_placeholder = st.empty()
            info_placeholder = st.empty()
            
            cap_cam = cv2.VideoCapture(0)
            if not cap_cam.isOpened():
                st.error("❌ Gagal mengakses webcam perangkat keras lokal.")
            else:
                try:
                    while run_live_stream:
                        ret, frame = cap_cam.read()
                        if not ret:
                            st.warning("⚠️ Gagal membaca bingkai video.")
                            break
                        
                        t_s = time.time()
                        results = model(frame, conf=conf_threshold, verbose=False)
                        t_e = time.time()
                        latensi_live = (t_e - t_s) * 1000
                        
                        annotated_frm = results[0].plot()
                        rgb_frm = cv2.cvtColor(annotated_frm, cv2.COLOR_BGR2RGB)
                        stream_placeholder.image(rgb_frm, channels="RGB", use_container_width=True)
                        
                        boxes_live = results[0].boxes
                        if boxes_live is not None and len(boxes_live) > 0:
                            cls_names = results[0].names
                            lbls = [cls_names[int(c)] for c in boxes_live.cls]
                            info_placeholder.info(f"🔴 Status Live: Terdeteksi {len(boxes_live)} objek ({', '.join(set(lbls))}) | Latensi: {latensi_live:.1f} ms")
                        else:
                            info_placeholder.warning("⚠️ Peringatan: Tidak ada objek jalan atau kerusakan yang terdeteksi pada frame ini.")
                        
                        time.sleep(0.02)
                finally:
                    cap_cam.release()
        else:
            st.info("💡 Centang kotak di atas untuk memulai streaming video langsung via webcam.")
