from PIL import Image, ImageColor
from rembg import remove

def cutout_on_gray(img: Image.Image, gray="#7f7f7f"):
    """
    High-quality cutout with alpha-matting + edge decontam,
    then composite on neutral mid-gray to avoid color bleeding.
    """
    fg = remove(
        img,
        alpha_matting=True,
        alpha_matting_erode_size=15,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        post_process=True,   # small morphology clean-up
    ).convert("RGBA")

    bg = Image.new("RGBA", fg.size, ImageColor.getrgb(gray)+(255,))
    out = Image.alpha_composite(bg, fg).convert("RGB")  # feed RGB to the model
    return out
