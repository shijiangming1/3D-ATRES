
import os

dir_path = '/root/sjm/3DLLM'
target_filename = 'llmapi.py'

print(f"--- Checking contents of: {dir_path} ---")

found = False
for filename in os.listdir(dir_path):
    # Print the filename as a string and as raw bytes to reveal hidden characters
    print(f"Found file: '{filename}' | As bytes: {filename.encode('utf-8')}")
    if filename == target_filename:
        found = True

print("\n--- Analysis ---")
if found:
    print("✅ SUCCESS: Python found a file with the exact name 'llmapi.py'.")
else:
    print("❌ FAILURE: Python did NOT find a file with the exact name 'llmapi.py'.")
    print("Look at the byte output above to see the actual filenames."








·X
X

XX
