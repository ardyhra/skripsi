# app.py
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import shutil
import subprocess
import re
import streamlit.components.v1 as components

from alpr_core import ALPRSystem
from utils import load_video_file, convert_uploaded_image, get_frame_from_cap
from config import DETECTOR_WEIGHTS, RECOGNIZER_WEIGHTS

st.set_page_config(page_title="ALPR - Deteksi & Rekognisi Plat Nomor", layout="wide")
st.title("🚗 Automatic License Plate Recognition (ALPR)")
st.markdown("Sistem deteksi plat nomor dengan **RF-DETR Nano** dan recognizer **Transformer Custom (PAD+EOS)**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    st.info(f"Detector: RF-DETR Nano\nRecognizer: Transformer PAD+EOS (fine-tuned)")
    debug_mode = st.checkbox("Mode Debug (cetak output model)", value=False)
    st.divider()
    st.header("📂 Sumber Input")
    input_type = st.radio(
        "Pilih jenis input:",
        ["Upload Gambar", "Upload Video", "Webcam (Live)", "CCTV Stream"]
    )
    use_tracker = st.checkbox("Aktifkan stabilisasi teks (tracker)", value=True)
    st.divider()
    st.caption("Dibuat untuk keperluan skripsi")

@st.cache_resource
def load_alpr(debug=False):
    try:
        alpr = ALPRSystem(DETECTOR_WEIGHTS, RECOGNIZER_WEIGHTS, debug=debug)
        return alpr
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

alpr = load_alpr(debug=debug_mode)

CCTV_SOURCES = [
    {
        "name": "OpenCCTV - Karet Kuningan 001 (iframe)",
        "kind": "iframe",
        "url": "https://opencctv.org/cameras/indonesia/dki-jakarta/jakarta/karet-kuningan-001-267645",
    },
    {
        "name": "CCTV Semarang - halaman web",
        "kind": "iframe",
        "url": "https://kecbanyumanik.semarangkota.go.id/cctv-semarang",
    },
    {
        "name": "TrafficVision - katalog Semarang",
        "kind": "iframe",
        "url": "https://trafficvision.live/blog/semarang-traffic-cameras",
    },
    {
        "name": "GlobeCam - katalog kamera publik",
        "kind": "iframe",
        "url": "https://globecam.ai/",
    },
]

def _uploaded_video_suffix(uploaded_file):
    _, ext = os.path.splitext(uploaded_file.name or "")
    return ext.lower() if ext.lower() in [".mp4", ".avi", ".mov"] else ".mp4"

def _new_temp_path(suffix):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = temp_file.name
    temp_file.close()
    return path

def _extract_iframe_src(value):
    match = re.search(r"""src=["']([^"']+)["']""", value or "", flags=re.IGNORECASE)
    return match.group(1) if match else (value or "").strip()

def _looks_like_direct_stream(url):
    lowered = (url or "").lower()
    return (
        lowered.startswith(("rtsp://", "rtmp://"))
        or ".m3u8" in lowered
        or ".mp4" in lowered
        or "mjpeg" in lowered
        or "video" in lowered
    )

def _render_iframe(url, height=520):
    components.iframe(url, height=height, scrolling=True)

def _open_video_writer(output_path, fps, frame_size):
    """Create an OpenCV writer without relying on H.264/OpenH264."""
    candidates = [
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    ]
    for codec, suffix in candidates:
        if not output_path.endswith(suffix):
            candidate_path = _new_temp_path(suffix)
        else:
            candidate_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(candidate_path, fourcc, fps, frame_size)
        if writer.isOpened():
            return writer, candidate_path, codec
        writer.release()
        if os.path.exists(candidate_path):
            os.unlink(candidate_path)
    return None, None, None

def _transcode_to_browser_mp4(input_path):
    """Return (output_path, ffmpeg_path, error_message) for browser-safe H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, None, "ffmpeg tidak ditemukan oleh proses Streamlit."
    output_path = _new_temp_path(".mp4")
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
        "-profile:v", "baseline", "-level", "3.0", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", output_path,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode == 0 and os.path.getsize(output_path) > 0:
        return output_path, ffmpeg, None
    if os.path.exists(output_path):
        os.unlink(output_path)
    error_message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg gagal tanpa pesan error."
    return None, ffmpeg, error_message

# ------------------- Upload Gambar -------------------
if input_type == "Upload Gambar":
    uploaded_file = st.file_uploader("Pilih gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bgr = convert_uploaded_image(uploaded_file)
        with st.spinner("Memproses..."):
            annotated, results = alpr.process_frame(image_bgr, use_tracker=use_tracker)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
        if results:
            st.subheader("Hasil Rekognisi")
            for i, res in enumerate(results):
                st.write(f"**Plat {i+1}:** `{res['text']}` (confidence: {res['confidence']:.2f})")
        else:
            st.warning("Tidak ada plat terdeteksi.")

# ------------------- Upload Video -------------------
elif input_type == "Upload Video":
    uploaded_file = st.file_uploader("Pilih video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        video_bytes = uploaded_file.getvalue()
        # Tampilkan video asli
        st.video(video_bytes)
        
        if st.button("Proses Video"):
            # Simpan video upload ke file sementara
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=_uploaded_video_suffix(uploaded_file))
            tfile.write(video_bytes)
            tfile.close()
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Tidak dapat membuka file video. Coba video lain atau konversi ke MP4.")
                os.unlink(tfile.name)
                st.stop()
            
            ret, first_frame = cap.read()
            if not ret:
                st.error("Video tidak berisi frame yang bisa dibaca.")
                cap.release()
                os.unlink(tfile.name)
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 25.0
            height, width = first_frame.shape[:2]
            out, output_path, codec_used = _open_video_writer(
                _new_temp_path(".mp4"),
                fps,
                (width, height),
            )
            if out is None:
                st.error("Tidak dapat membuat video writer OpenCV di sistem ini.")
                cap.release()
                os.unlink(tfile.name)
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            preview = st.empty()
            
            try:
                frame = first_frame
                while True:
                    annotated, _ = alpr.process_frame(frame, use_tracker=use_tracker)
                    out.write(annotated)
                    if frame_count % 30 == 0:
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        progress = frame_count / total_frames if total_frames > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Memproses frame {frame_count}/{total_frames}")
                    ret, frame = cap.read()
                    if not ret:
                        break
            finally:
                cap.release()
                out.release()
            
            progress_bar.progress(1.0)
            status_text.text("Selesai! Menyiapkan video hasil...")
            
            transcoded_path, ffmpeg_path, ffmpeg_error = _transcode_to_browser_mp4(output_path)
            browser_output_path = transcoded_path or output_path
            with open(browser_output_path, "rb") as f:
                result_video_bytes = f.read()

            if transcoded_path:
                st.caption(f"Video hasil dikonversi ke H.264 dengan FFmpeg: {ffmpeg_path}")
            elif ffmpeg_error:
                st.warning(f"Video belum dikonversi ke H.264: {ffmpeg_error}")

            if browser_output_path == output_path and codec_used != "mp4v":
                st.info("Preview browser mungkin terbatas untuk codec ini. File hasil tetap bisa diunduh.")
            elif browser_output_path == output_path:
                st.info("Hasil dibuat dengan codec mp4v. Jika preview masih abu-abu, install ffmpeg lalu proses ulang agar dikonversi ke H.264.")

            video_format = "video/mp4" if browser_output_path.endswith(".mp4") else "video/x-msvideo"
            st.video(result_video_bytes, format=video_format)
            st.download_button(
                "Unduh video hasil",
                data=result_video_bytes,
                file_name="hasil_alpr.mp4" if browser_output_path.endswith(".mp4") else "hasil_alpr.avi",
                mime=video_format,
            )
            st.success("Video hasil telah diproses.")
            
            # Bersihkan file sementara
            os.unlink(tfile.name)
            os.unlink(output_path)
            if browser_output_path != output_path:
                os.unlink(browser_output_path)

# ------------------- Webcam -------------------
elif input_type == "Webcam (Live)":
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    except ImportError:
        st.error("Library streamlit-webrtc belum terinstall. Jalankan: pip install streamlit-webrtc")
        st.stop()

    class ALPRVideoTransformer(VideoTransformerBase):
        def __init__(self, alpr_sys, use_tracker):
            self.alpr = alpr_sys
            self.use_tracker = use_tracker
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = self.alpr.process_frame(img, use_tracker=self.use_tracker)
            return annotated

    webrtc_streamer(
        key="webcam",
        video_transformer_factory=lambda: ALPRVideoTransformer(alpr, use_tracker),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.info("Webcam aktif. Deteksi dan rekognisi berjalan real-time.")

# ------------------- CCTV Stream -------------------
elif input_type == "CCTV Stream":
    st.subheader("CCTV Stream")
    source_names = [source["name"] for source in CCTV_SOURCES] + ["Tambahkan link sendiri"]
    selected_source_name = st.selectbox("Pilih sumber CCTV", source_names)

    direct_url = ""
    iframe_url = ""

    if selected_source_name == "Tambahkan link sendiri":
        custom_mode = st.radio(
            "Jenis link",
            ["Web / iframe", "Direct stream untuk ALPR"],
            horizontal=True,
        )
        custom_value = st.text_area(
            "Tempel URL atau kode iframe",
            placeholder='https://contoh.go.id/cctv atau <iframe src="https://..."></iframe>',
            height=90,
        )
        parsed_url = _extract_iframe_src(custom_value)
        if custom_mode == "Direct stream untuk ALPR" or _looks_like_direct_stream(parsed_url):
            direct_url = parsed_url
        else:
            iframe_url = parsed_url
    else:
        selected_source = next(source for source in CCTV_SOURCES if source["name"] == selected_source_name)
        if selected_source["kind"] == "direct":
            direct_url = selected_source["url"]
        else:
            iframe_url = selected_source["url"]

    if iframe_url:
        st.caption(f"Sumber web/iframe: {iframe_url}")
        _render_iframe(iframe_url)
        st.info(
            "Sumber iframe bisa ditampilkan di aplikasi, tetapi OpenCV tidak bisa langsung memproses isi iframe. "
            "Untuk menjalankan ALPR, gunakan URL stream asli seperti RTSP, HLS .m3u8, MP4, atau MJPEG."
        )
        direct_url = st.text_input(
            "URL direct stream opsional untuk ALPR",
            placeholder="rtsp://... atau https://.../playlist.m3u8",
        )

    if direct_url:
        st.caption(f"Direct stream ALPR: {direct_url}")
        batch_frames = st.number_input("Frame per siklus live", min_value=1, max_value=300, value=30, step=1)
        frame_skip = st.number_input("Lewati frame", min_value=0, max_value=30, value=0, step=1)
        refresh_delay = st.number_input("Jeda antar siklus (detik)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

        stream_state_key = f"cctv_running_{direct_url}"
        if stream_state_key not in st.session_state:
            st.session_state[stream_state_key] = False

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button("Mulai live ALPR"):
                st.session_state[stream_state_key] = True
        with stop_col:
            if st.button("Berhenti live ALPR"):
                st.session_state[stream_state_key] = False

        if st.session_state[stream_state_key]:
            cap = cv2.VideoCapture(direct_url)
            if not cap.isOpened():
                st.error("Tidak dapat membuka stream. Pastikan URL adalah endpoint video langsung, bukan halaman web atau iframe.")
                st.session_state[stream_state_key] = False
            else:
                frame_placeholder = st.empty()
                result_placeholder = st.empty()
                status_placeholder = st.empty()
                processed_frames = 0
                read_frames = 0
                last_results = []

                try:
                    while processed_frames < batch_frames:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream terputus atau frame tidak bisa dibaca.")
                            st.session_state[stream_state_key] = False
                            break

                        read_frames += 1
                        if frame_skip and read_frames % (frame_skip + 1) != 1:
                            continue

                        annotated, results = alpr.process_frame(frame, use_tracker=use_tracker)
                        frame_placeholder.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_container_width=True,
                        )

                        if results:
                            last_results = results
                            result_placeholder.markdown(
                                "\n".join(
                                    [f"**{res['text']}** (conf: {res['confidence']:.2f})" for res in results]
                                )
                            )
                        elif not last_results:
                            result_placeholder.info("Stream terbaca, belum ada plat terdeteksi.")

                        processed_frames += 1
                        status_placeholder.caption(f"Live aktif - batch frame {processed_frames}/{batch_frames}")
                finally:
                    cap.release()

                if st.session_state[stream_state_key]:
                    time.sleep(refresh_delay)
                    st.rerun()
                else:
                    st.success(f"Live ALPR dihentikan setelah batch terakhir ({processed_frames} frame).")

# ------------------- CCTV Stream lama -------------------
elif False and input_type == "CCTV Stream":
    url = st.text_input("Masukkan URL stream CCTV (contoh: rtsp://... atau http://...)")
    if url:
        if st.button("Mulai Stream"):
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                st.error("Tidak dapat membuka stream. Periksa URL.")
            else:
                frame_placeholder = st.empty()
                stop_button = st.button("Berhenti")
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Stream terputus.")
                        break
                    annotated, results = alpr.process_frame(frame, use_tracker=use_tracker)
                    frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    if results:
                        st.sidebar.markdown("**Hasil terakhir:**")
                        for res in results:
                            st.sidebar.write(f"📌 {res['text']} (conf: {res['confidence']:.2f})")
                    time.sleep(0.03)
                cap.release()
