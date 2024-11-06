from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Load an image
image = Image.open("flower.jpg")

# Apply different filters from Pillow
original_image = image
blurred_image = image.filter(ImageFilter.BLUR)
contour_image = image.filter(ImageFilter.CONTOUR)
detail_image = image.filter(ImageFilter.DETAIL)
edge_enhance_image = image.filter(ImageFilter.EDGE_ENHANCE)
edge_enhance_more_image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
emboss_image = image.filter(ImageFilter.EMBOSS)
find_edges_image = image.filter(ImageFilter.FIND_EDGES)
sharpen_image = image.filter(ImageFilter.SHARPEN)
smooth_image = image.filter(ImageFilter.SMOOTH)
smooth_more_image = image.filter(ImageFilter.SMOOTH_MORE)

# Create a list of images and their titles for display
images = [
    (original_image, "Original"),
    (blurred_image, "Blur"),
    (contour_image, "Contour"),
    (detail_image, "Detail"),
    (edge_enhance_image, "Edge Enhance"),
    (edge_enhance_more_image, "Edge Enhance More"),
    (emboss_image, "Emboss"),
    (find_edges_image, "Find Edges"),
    (sharpen_image, "Sharpen"),
    (smooth_image, "Smooth"),
    (smooth_more_image, "Smooth More")
]

# Display images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
