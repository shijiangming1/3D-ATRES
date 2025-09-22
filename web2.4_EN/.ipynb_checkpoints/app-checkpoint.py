import os
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import shutil
import uuid
import time
import threading
import requests
import json

# ==============================================================================
# == Large Model API Settings
# ==============================================================================
API_KEY = "sk-v5sa4gm4bawsddqxr2jfkw3jyqr466ij7rnvwrcxe2nexs33"  # Please replace with your API Key
MODEL = "/maas/deepseek-ai/DeepSeek-V3.1"


def chat_completion(api_key, model, messages):
    url = "https://maas-api.lanyun.net/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": model, "messages": messages}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# ==============================================================================
# == Environment and Path Settings
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming llmapi.py is in the project root directory
project_root = current_dir
sys.path.insert(0, project_root)

# ==============================================================================
# == Simulate Model Loading (for local testing)
# ==============================================================================
# Since your real model is not accessible, a mock class is created here to ensure the program runs
# class MockModelWrapper:
# def __init__(self, *args, **kwargs):
#     print("[Note] Using a simulated 3D large model (MockModelWrapper).")
#
# def preprocess_pointcloud(self, filepath):
#     print(f"[Simulating] Preprocessing point cloud: {filepath}")
#     # Return some simulated data
#     return (
#     "coords_data", "colors_data", "pc_data", "voxel_coords", "p2v_map", "v2p_map", "spatial_shape", "sp_mask")
#
# def inference(self, *args, **kwargs):
#     print("[Simulating] Performing model inference...")
#     # Simulate successful localization and return a random mask
#     mask = np.zeros(1000, dtype=bool)
#     mask[np.random.choice(1000, 100, replace=False)] = True
#     return ("Simulated inference result", mask)

# ==============================================================================
# == Simulate Model Loading (for GPU testing)
# ==============================================================================
# Since your real model is not accessible, a mock class is created here to ensure the program runs
# Note: Make sure your ModelWrapper and related dependencies are available
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/root/sjm/3DLLM"
sys.path.insert(0, project_root)
from llmapi import ModelWrapper
from llmapi import ModelWrapper

MODEL_PATH = "/root/lanyun-fs/public_model/finetune-3d-llava-lora-0801-W08GMM0CL05COT0-bs-2-gpu8-V7-best"
MODEL_BASE_PATH = "/root/lanyun-fs/public_model/llava-v1.5-7b"

print("Loading 3D large model, please wait...")
model_wrapper = ModelWrapper(
    model_path=MODEL_PATH,
    model_base=MODEL_BASE_PATH,
    conv_mode="llava_v1"
)
print("[✓] 3D large model loaded successfully!")

preprocessed_cache = {
    "filepath": None,
    "data": None
}

# --- Add this line to initialize global variables ---
pregenerated_video_cache = {
    "status": "none",  # Can be 'none', 'generating', 'ready', 'error'
    "filepath": None
}


# ==============================================================================
# == Video Generation and Model Call
# ==============================================================================
def execute_video_generation_logic(pointcloud_full_path, mask):
    try:
        print(f"[*] Starting navigation video generation...")
        print(f"    - Input point cloud: {pointcloud_full_path}")
        print(f"    - Input Mask shape: {mask.shape}, valid points: {np.sum(mask)}")

        # Import navigation API
        from navigation_api import NavigationAPI

        # Generate the corresponding scene path from the point cloud path (.ply -> .glb)
        scene_path = pointcloud_full_path.replace('.ply', '.glb')
        print(f"    - Scene path: {scene_path}")

        # Create an instance of the navigation API
        api = NavigationAPI(
            yaml_path="/root/sjm/habitat-lab-old/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_scannet.yaml",
            scene_path=scene_path
        )

        # Execute navigation task
        output_path = GENERATED_VIDEOS_FOLDER
        video_path, stats = api.navigate_to_target_with_mask(
            ply_path=pointcloud_full_path,
            pred_mask=mask,
            output_path=output_path,
            scene_path=scene_path
        )

        print(f"[✓] Navigation video generated successfully!")
        print(f"    - Video path: {video_path}")
        print(f"    - Navigation stats: {stats}")

        # Extract video filename
        unique_filename = os.path.basename(video_path)

        return unique_filename, None

    except Exception as e:
        print(f"[!] An error occurred during video generation: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def video_generation_worker(pointcloud_full_path, mask):
    global pregenerated_video_cache
    pregenerated_video_cache['status'] = 'generating'
    pregenerated_video_cache['filepath'] = None
    print("[Thread] Background video generation thread started.")

    filename, error = execute_video_generation_logic(pointcloud_full_path, mask)

    if error:
        pregenerated_video_cache['status'] = 'error'
        print(f"[Thread] Background video generation failed: {error}")
    else:
        pregenerated_video_cache['filepath'] = filename
        pregenerated_video_cache['status'] = 'ready'
        print(f"[Thread] Background video is ready, file: {filename}")


def call_large_language_model(mode, query, filepath):
    try:
        query_ori = query
        MESSAGES = [{"role": "system",
                     "content": "The customer's input must be related to indoor furniture. If the input is not a common item in a room, reply with 0, for example 'hello' or 'where is the monkey' (Note: descriptions of rooms or objects are allowed). If it is related, no matter what language the customer inputs, strictly translate it into English, without any other extensions."},
                    {"role": "user", "content": query}]
        result = chat_completion(API_KEY, MODEL, MESSAGES)
        query = result["choices"][0]["message"]["content"]
        print("Large language model translation", query)
        print(f"Simulated translation: {query_ori} -> {query} (English)")  # Simulated translation
    except Exception as e:
        print(f"Translation API call failed: {e}")

    global preprocessed_cache, pregenerated_video_cache

    if mode != 'locate' or not filepath:
        pregenerated_video_cache['status'] = 'none'

    if not filepath:
        return {'type': 'chat', 'response': 'I am a 3D visual localization large model, please upload a point cloud and tell me about the room, for example, "what color is the chair?".'}

    full_path = os.path.join(UPLOADS_FOLDER, filepath)
    if not os.path.exists(full_path):
        return {'type': 'chat', 'response': f'Error: File {filepath} not found on the server.'}

    if preprocessed_cache.get("filepath") != filepath:
        processed_data = model_wrapper.preprocess_pointcloud(full_path)
        preprocessed_cache["filepath"] = filepath
        preprocessed_cache["data"] = processed_data

    if not preprocessed_cache.get("data"):
        return {'type': 'chat', 'response': 'Error: Failed to process point cloud data.'}

    coords, colors, pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, sp_mask = preprocessed_cache["data"]

    if mode == 'chat':
        if "0" in query:
            try:
                MESSAGES = [{"role": "system",
                             "content": "I am a 3D visual localization large model developed by Shanghai Innovation Institute, please ask questions related to point clouds."},
                            {"role": "user", "content": query_ori}]
                result = chat_completion(API_KEY, MODEL, MESSAGES)
                text = result["choices"][0]["message"]["content"]
                print("Large language model translation", query)
                print(f"Simulated translation: {query_ori} -> {query} (English)")  # Simulated translation
            except Exception as e:
                print(f"Translation API call failed: {e}")

            # text = "The 3D localization visual large model cannot provide service for this instruction!"
            return {'type': 'chat', 'response': text}
        else:
            # input_text = "<image> \n {} Answer the question simply.".format(query)
            input_text = "<image> \n {}? Answer the question in detail.".format(query)

            pred = model_wrapper.inference(pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, input_text, sp_mask)
            text = pred[0]
            mask = pred[1]
            # try:
            #      MESSAGES = [
            #          {"role": "system",
            #           "content": "No matter what language the customer inputs, translate it into Chinese. Strictly translate, do not generate any other content unrelated to the translation"},
            #          {"role": "user", "content": text}
            #      ]
            #
            #      result = chat_completion(API_KEY, MODEL, MESSAGES)
            #      # Check for a successful response and extract the content
            #
            #      text = result["choices"][0]["message"]["content"]
            #      print("Large language model translation", query)
            #      return {'type': 'chat', 'response': text}
            # except:
            return {'type': 'chat', 'response': text}

    elif mode == 'locate':
        if "0" in query:
            pregenerated_video_cache['status'] = 'none'
            return {'type': 'locate', 'indices': [], 'response': "Target not found", 'video_pregeneration_started': False}

        input_text = f"<image> \n Please output the segmentation mask according to the following description. \n{query}"
        text_resp, mask = model_wrapper.inference(pc_data, voxel_coords, p2v_map, v2p_map, spatial_shape, input_text,
                                                    sp_mask)

        if mask.any():
            text = "Target found, highlighted in red. Navigation feature is unlocked."
            one_indices = (np.where(mask == 1)[0]).tolist()

            if pregenerated_video_cache['status'] != 'generating':
                thread = threading.Thread(target=video_generation_worker, args=(full_path, mask))
                thread.start()

            return {'type': 'locate', 'indices': one_indices, 'response': text, 'video_pregeneration_started': True}
        else:
            text = "Target not found"
            pregenerated_video_cache['status'] = 'none'
            return {'type': 'locate', 'indices': [], 'response': text, 'video_pregeneration_started': False}

    return {'type': 'chat', 'response': 'Sorry, I cannot recognize this mode.'}


# ==============================================================================
# == Flask Application Settings
# ==============================================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOADS_FOLDER = os.path.join(project_root, 'uploads')
if not os.path.exists(UPLOADS_FOLDER): os.makedirs(UPLOADS_FOLDER)
GENERATED_VIDEOS_FOLDER = os.path.join(project_root, 'generated_videos')
if not os.path.exists(GENERATED_VIDEOS_FOLDER): os.makedirs(GENERATED_VIDEOS_FOLDER)


# ==============================================================================
# == Flask Route Definitions
# ==============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat_api', methods=['POST'])
def chat_api():
    try:
        data = request.get_json()
        if not all(k in data for k in ['mode', 'query']):
            return jsonify({'error': 'Invalid request format'}), 400
        result = call_large_language_model(mode=data.get('mode'), query=data.get('query'),
                                             filepath=data.get('filepath'))
        return jsonify(result)
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': f'Internal Server Error: {e}'}), 500


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS_FOLDER, filename)


@app.route('/upload_ply', methods=['POST'])
def upload_ply_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    if file and file.filename.endswith('.ply'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOADS_FOLDER, filename)
        file.save(filepath)
        return jsonify({'success': True, 'message': f'File {filename} successfully saved.', 'filename': filename})
    return jsonify({'success': False, 'error': 'Invalid file type'}), 400


@app.route('/get_navigation_video', methods=['POST'])
def get_navigation_video():
    global pregenerated_video_cache
    status = pregenerated_video_cache.get('status')
    if status == 'ready':
        filename = pregenerated_video_cache.get('filepath')
        video_url = f"/generated_videos/{filename}"
        return jsonify({'success': True, 'status': 'ready', 'video_url': video_url})
    elif status == 'generating':
        return jsonify({'success': True, 'status': 'generating', 'message': 'Navigation video is still being generated, please try again later.'})
    else:
        return jsonify({'success': False, 'status': status, 'error': 'Navigation video is not ready yet or generation failed.'})


@app.route('/generated_videos/<path:filename>')
def serve_generated_video(filename):
    return send_from_directory(GENERATED_VIDEOS_FOLDER, filename)


# ==============================================================================
# == Start Server
# ==============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False)