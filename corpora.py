import subprocess
import sys

subprocess.run([f"{sys.executable}", "-m", "textblob.download_corpora"])
print("Working")

