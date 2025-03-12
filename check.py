import h5py

MODEL_PATH = r"D:\OneDrive\Desktop\major-project-main - Copy1\major-project-main - Copy\deficiency_classifier.h5"

print("Checking HDF5 file...")

try:
    with h5py.File(MODEL_PATH, "r") as f:
        print("✅ HDF5 File Loaded Successfully!")
except Exception as e:
    print("❌ Error:", e)

print("Script finished execution.")
