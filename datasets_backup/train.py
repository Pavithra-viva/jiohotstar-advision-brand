import cv2
import os
import re

# ---- FOLDER PATHS ----
image_folder = "datasets"
os.makedirs("labels", exist_ok=True)

# ---- GET ALL IMAGE FILES ----
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# ---- EXTRACT CLASS NAMES FROM FILENAMES ----
def extract_class_name(filename):
    name = os.path.splitext(filename)[0]         # remove extension
    name = re.split(r'[_\d]', name)[0]           # remove numbers & underscores
    return name.lower()

# build unique class list
classes = sorted(list({extract_class_name(f) for f in image_files}))
print("✅ Classes detected:", classes)

drawing = False
ix, iy = -1, -1
current_image_index = 0

def load_image():
    global img, img_copy, image_name, class_id, label_path
    image_name = image_files[current_image_index]

    img = cv2.imread(os.path.join(image_folder, image_name))
    img_copy = img.copy()

    class_name = extract_class_name(image_name)
    class_id = classes.index(class_name)

    label_path = f"labels/{image_name.split('.')[0]}.txt"
    print(f"\n📌 Now labeling: {image_name}  → class: {class_name} (ID={class_id})")


def mouse_event(event, x, y, flags, param):
    global drawing, ix, iy, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img = img_copy.copy()
        cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)

        h, w = img.shape[:2]
        x_center = (ix + x) / 2 / w
        y_center = (iy + y) / 2 / h
        width = abs(x - ix) / w
        height = abs(y - iy) / h

        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ Saved box → {label_path}")


# ---- START ----
load_image()
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", mouse_event)

while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        current_image_index += 1
        if current_image_index >= len(image_files):
            print("\n🎉 All images labeled successfully!")
            break
        load_image()

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# ---- PRINT FINAL CLASS LIST ----
print("\n✅ FINAL CLASS LIST FOR YOLO:")
for i, c in enumerate(classes):
    print(f"{i}: {c}")
