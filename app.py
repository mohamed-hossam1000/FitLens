"""
app.py — FitLens Streamlit UI
Run with: streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
"""

import io
import base64
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from shared.schemas import PipelinePayload
from modules import preprocess_person, preprocess_garments, multi_garment_try_on, GarmentSegmentor
from pipeline import try_on, recolor_garment, generate_video

import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "FitLens",
    page_icon  = "👗",
    layout     = "wide",
)

st.title("👗 FitLens — Virtual Try-On")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "👤 Single Garment",
    "👕👖 Multiple Garments",
    "🧥 Extract from Model",
])


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pil_to_np(pil_image: Image.Image) -> np.ndarray:
    return np.array(pil_image.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)

def image_to_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

def parse_hex_color(hex_color: str):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def show_result(original: np.ndarray, result: np.ndarray):
    """Show side by side comparison + download button."""
    if result is None:
        st.error("Pipeline failed — no result image returned.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original, width=500)
    with col2:
        st.subheader("Try-On Result")
        st.image(result, width=500)
    st.download_button(
        label     = "⬇️ Download Result",
        data      = image_to_bytes(np_to_pil(result)),
        file_name = "fitlens_result.png",
        mime      = "image/png",
    )

def video_option(key_suffix: str = ""):
    """Locked video generation option."""
    st.divider()
    with st.expander("🎬 Video Generation (Coming Soon)"):
        st.info(
            "Video generation via Wan2.1 I2V is available in the full pipeline "
            "but requires additional GPU resources. Coming soon."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox(
                "Motion style",
                ["walking", "turning", "posing", "windy"],
                disabled = True,
                key      = f"motion_{key_suffix}",
            )
        with col2:
            st.button("Generate Video", disabled=True, key=f"vid_btn_{key_suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single Garment
# ══════════════════════════════════════════════════════════════════════════════
# ── initialize session state ───────────────────────────────────────────────────
if "single_result" not in st.session_state:
    st.session_state.single_result   = None
if "single_original" not in st.session_state:
    st.session_state.single_original = None
if "multi_result" not in st.session_state:
    st.session_state.multi_result    = None
if "multi_original" not in st.session_state:
    st.session_state.multi_original  = None
if "extract_result" not in st.session_state:
    st.session_state.extract_result   = None
if "extract_original" not in st.session_state:
    st.session_state.extract_original = None
with tab1:
    st.header("Single Garment Try-On")
    st.caption("Upload a photo of yourself and one garment. Gender and garment type are detected automatically.")

    col1, col2 = st.columns(2)
    with col1:
        person_file  = st.file_uploader("Person photo", type=["jpg", "jpeg", "png"], key="s_person")
    with col2:
        garment_file = st.file_uploader("Garment image", type=["jpg", "jpeg", "png"], key="s_garment")

    with st.expander("🎨 Recolor Garment (optional)"):
        enable_recolor = st.checkbox("Enable recoloring", key="s_recolor_check")
        target_color   = st.color_picker("Pick a color", "#DC3232", key="s_color") if enable_recolor else None

    run_single = st.button("👗 Run Try-On", key="run_single", type="primary")

    if run_single:
        if not person_file or not garment_file:
            st.warning("Please upload both a person photo and a garment image.")
        else:
            person_image  = pil_to_np(Image.open(person_file))
            garment_image = pil_to_np(Image.open(garment_file))

            with st.spinner("Detecting gender and garment type..."):
                person_info  = preprocess_person(person_image)
                garment_info = preprocess_garments([garment_image])[0]

            st.info(f"Detected — Gender: **{person_info['gender']}** | Garment: **{garment_info['type']}**")

            target_rgb = parse_hex_color(target_color) if enable_recolor and target_color else None

            payload = PipelinePayload(
                person_image  = person_image,
                garment_image = garment_info["image"],
                garment_type  = garment_info["type"],
                target_rgb    = target_rgb,
            )

            with st.spinner("Running try-on... this may take a minute ⏳"):
                payload = try_on(payload)

            if enable_recolor and target_rgb:
                with st.spinner("Recoloring garment..."):
                    payload = recolor_garment(payload)

            # ── save to session state ─────────────────────────────────────────
            st.session_state.single_result   = payload.result_image
            st.session_state.single_original = person_image

    # ── render from session state — survives reconnects ───────────────────────
    if st.session_state.single_result is not None:
        show_result(st.session_state.single_original, st.session_state.single_result)
        video_option("single")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Multiple Garments
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Multiple Garments Try-On")
    st.caption("Upload a person photo and multiple garments. They are applied sequentially.")

    person_file_m = st.file_uploader("Person photo", type=["jpg", "jpeg", "png"], key="m_person")

    st.subheader("Garments")
    st.caption("Upload 2 or more garment images.")
    garment_files = st.file_uploader(
        "Garment images",
        type                  = ["jpg", "jpeg", "png"],
        accept_multiple_files = True,
        key                   = "m_garments",
    )

    with st.expander("🎨 Recolor Garment (optional)"):
        enable_recolor_m = st.checkbox("Enable recoloring", key="m_recolor_check")
        target_color_m   = st.color_picker("Pick a color", "#DC3232", key="m_color") if enable_recolor_m else None

    run_multi = st.button("👗 Run Try-On", key="run_multi", type="primary")

    if run_multi:
        if not person_file_m:
            st.warning("Please upload a person photo.")
        elif not garment_files or len(garment_files) < 2:
            st.warning("Please upload at least 2 garment images.")
        else:
            person_image   = pil_to_np(Image.open(person_file_m))
            garment_images = [pil_to_np(Image.open(f)) for f in garment_files]

            with st.spinner("Detecting gender and garment types..."):
                person_info = preprocess_person(person_image)
                garments    = preprocess_garments(garment_images)

            types_detected = " | ".join([f"Garment {i+1}: **{g['type']}**" for i, g in enumerate(garments)])
            st.info(f"Detected — Gender: **{person_info['gender']}** | {types_detected}")

            target_rgb_m = parse_hex_color(target_color_m) if enable_recolor_m and target_color_m else None

            with st.spinner(f"Applying {len(garments)} garments sequentially... this may take a while ⏳"):
                final_image = multi_garment_try_on(person_image, garments)

            payload = PipelinePayload(
                person_image  = person_image,
                garment_image = garments[-1]["image"],
                garment_type  = garments[-1]["type"],
                target_rgb    = target_rgb_m,
                result_image  = final_image,
            )

            if enable_recolor_m and target_rgb_m:
                with st.spinner("Recoloring garment..."):
                    payload = recolor_garment(payload)

            show_result(person_image, payload.result_image)
            video_option("multi")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Extract from Fashion Model
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Extract Garment from Fashion Model")
    st.caption(
        "Upload a fashion model photo. Click on the garment you want to extract, "
        "then upload your own photo to try it on."
    )

    col1, col2 = st.columns(2)
    with col1:
        model_file    = st.file_uploader("Fashion model photo", type=["jpg", "jpeg", "png"], key="e_model")
    with col2:
        person_file_e = st.file_uploader("Your photo",          type=["jpg", "jpeg", "png"], key="e_person")

    garment_type_override = st.selectbox(
        "Garment type (select manually)",
        ["upper", "lower", "shoes", "overall"],
        key = "e_garment_type",
    )

    # ── Canvas — only shown when model image is uploaded ──────────────────────
    model_image     = None
    positive_points = []

    if model_file:
        model_image   = Image.open(model_file).convert("RGB")
        canvas_width  = 400
        scale         = canvas_width / model_image.width
        canvas_height = int(model_image.height * scale)
        model_resized = model_image.resize((canvas_width, canvas_height))

        st.subheader("Click on the garment to extract")
        st.caption("🟢 Click = include region | add more clicks to refine")

        canvas_result = st_canvas(
            fill_color           = "rgba(0, 200, 120, 0.3)",
            stroke_width         = 3,
            stroke_color         = "#00C878",
            background_image     = model_resized,
            background_color     = "#ffffff",
            update_streamlit     = True,
            width                = canvas_width,
            height               = canvas_height,
            drawing_mode         = "point",
            point_display_radius = 8,
            key                  = "canvas",
        )

        # Scale canvas coords back to original image coords
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            for obj in canvas_result.json_data["objects"]:
                x = int(obj["left"] / scale)
                y = int(obj["top"]  / scale)
                positive_points.append((x, y))
            st.caption(f"✅ {len(positive_points)} point(s) selected: {positive_points}")

        # Live mask preview
        if positive_points:
            with st.spinner("Previewing mask..."):
                seg = GarmentSegmentor()
                seg.set_image(model_image)
                for x, y in positive_points:
                    seg.add_click(x, y, is_positive=True)
                preview = seg.get_preview()
            st.image(preview, caption="Mask preview", width=500)

    with st.expander("🎨 Recolor Garment (optional)"):
        enable_recolor_e = st.checkbox("Enable recoloring", key="e_recolor_check")
        target_color_e   = st.color_picker("Pick a color", "#DC3232", key="e_color") if enable_recolor_e else None

    run_extract = st.button("👗 Run Try-On", key="run_extract", type="primary")

    if run_extract:
        if not model_file or model_image is None:
            st.warning("Please upload a fashion model photo.")
        elif not person_file_e:
            st.warning("Please upload your photo.")
        elif not positive_points:
            st.warning("Please click on the garment to extract it first.")
        else:
            person_image_e = pil_to_np(Image.open(person_file_e))

            with st.spinner("Extracting garment with SAM2..."):
                seg = GarmentSegmentor()
                seg.set_image(model_image)
                for x, y in positive_points:
                    seg.add_click(x, y, is_positive=True)
                cutout        = seg.get_cutout()
                garment_image = pil_to_np(cutout)

            with st.spinner("Detecting gender..."):
                person_info = preprocess_person(person_image_e)

            st.info(f"Detected — Gender: **{person_info['gender']}** | Garment: **{garment_type_override}**")

            target_rgb_e = parse_hex_color(target_color_e) if enable_recolor_e and target_color_e else None

            payload = PipelinePayload(
                person_image  = person_image_e,
                garment_image = garment_image,
                garment_type  = garment_type_override,
                target_rgb    = target_rgb_e,
            )

            with st.spinner("Running try-on... this may take a minute ⏳"):
                payload = try_on(payload)

            if enable_recolor_e and target_rgb_e:
                with st.spinner("Recoloring garment..."):
                    payload = recolor_garment(payload)

            show_result(person_image_e, payload.result_image)
            video_option("extract")