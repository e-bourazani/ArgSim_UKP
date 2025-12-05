import amrlib
import os

# Load the model
stog = amrlib.load_stog_model()

# Find the model folder by reading the meta file
meta_file = os.path.join(os.path.dirname(stog._meta_path), "amrlib_meta.json")
print("Model folder:", os.path.dirname(meta_file))
