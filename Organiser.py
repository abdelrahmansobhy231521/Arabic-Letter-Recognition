import os
import shutil
import re

# Define mapping from label numbers to Arabic letters
label_to_letter = {
    1: "Alef",
    2: "Baa",
    3: "Taa",
    4: "Thaa",
    5: "Jeem",
    6: "Haa",
    7: "Khaa",
    8: "Daal",
    9: "Thaal",
    10: "Raa",
    11: "Zay",
    12: "Seen",
    13: "Sheen",
    14: "Saad",
    15: "Daad",
    16: "Ttaa",
    17: "Zhaa",
    18: "Ain",
    19: "Ghain",
    20: "Faa",
    21: "Qaaf",
    22: "Kaaf",
    23: "Laam",
    24: "Meem",
    25: "Noon",
    26: "Haa2",
    27: "Waw",
    28: "Yaa"
}

# Set your source directory (where all the images are)
source_dir = r"D:\CodAlpha\Task3\archive (8)\Test Images 3360x32x32\test"  # Update this to your actual folder path

# Loop through each file in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".png"):
        # Try to extract the label number using regex
        match = re.search(r'label_(\d+)', filename)
        if match:
            label_num = int(match.group(1))
            letter_name = label_to_letter.get(label_num, f"Unknown_{label_num}")
            target_dir = os.path.join(source_dir, letter_name)

            # Create the directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            # Move the file
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.move(src_path, dst_path)

print("✅ All images organized by Arabic letter.")
