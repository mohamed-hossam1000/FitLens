# ──────────────────────────────────────────────────────────────────
# Person Segmentation + White Background
# ──────────────────────────────────────────────────────────────────

def segment_on_white(person_image: np.ndarray, predictor) -> tuple:
    """
    Segment person using SAM2 and place on white background.

    Returns:
        clean_image : np.ndarray — person on white background
        mask_u8     : np.ndarray — binary mask (255=person, 0=background)
    """
    h, w = person_image.shape[:2]

    predictor.set_image(person_image)

    box = np.array([w * 0.15, h * 0.05, w * 0.85, h * 0.95])
    input_point = np.array([
        [w // 2, h // 3],
        [w // 2, h // 2],
    ])
    input_label = np.array([1, 1])

    masks, scores, _ = predictor.predict(
        point_coords     = input_point,
        point_labels     = input_label,
        box              = box,
        multimask_output = False,
    )

    mask    = masks[0]
    mask_u8 = (mask * 255).astype(np.uint8)

    print(f"✅ SAM2 segmentation score: {scores[0]:.4f}")

    person_masked = cv2.bitwise_and(person_image, person_image, mask=mask_u8)
    white_bg      = Image.new("RGB", (w, h), (255, 255, 255))
    white_bg.paste(Image.fromarray(person_masked), mask=Image.fromarray(mask_u8))
    white_bg      = ImageOps.expand(white_bg, border=40, fill=(255, 255, 255))

    print("✅ Person placed on white background")

    # Return both the clean image AND the mask
    # mask_u8 is saved so we can restore the original background later
    return np.array(white_bg), mask_u8



def restore_background(
    original_person: np.ndarray,
    tryon_result:    np.ndarray,
    mask_u8:         np.ndarray,
) -> np.ndarray:
    """
    Place the try-on result person back onto the original background.

    How it works:
        - mask_u8 = 255 where the person is, 0 where background is
        - We take the person region from tryon_result
        - We take the background region from original_person
        - We blend them together

    Args:
        original_person : np.ndarray (RGB) — original person photo
        tryon_result    : np.ndarray (RGB) — try-on output image
        mask_u8         : np.ndarray       — SAM2 mask from preprocessing

    Returns:
        np.ndarray (RGB) — try-on person on original background
    """

    # Resize all to the same size just in case
    h, w = original_person.shape[:2]
    tryon_result = cv2.resize(tryon_result, (w, h))
    mask_u8      = cv2.resize(mask_u8,      (w, h))

    # Normalize mask to 0-1 for blending
    mask_float = mask_u8.astype(np.float32) / 255.0
    mask_3ch   = np.stack([mask_float] * 3, axis=-1)  # expand to 3 channels

    # Blend:
    # where mask=1 (person region) → take from try-on result
    # where mask=0 (background)    → take from original photo
    original_f = original_person.astype(np.float32)
    tryon_f    = tryon_result.astype(np.float32)

    blended = tryon_f * mask_3ch + original_f * (1 - mask_3ch)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    print("✅ Try-on person restored onto original background")
    return blended