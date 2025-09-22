document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element Caching ---
    const dom = {
        viewerElement: document.getElementById('viewer'),
        selectFileBtn: document.getElementById('select-file-btn'),
        fileInput: document.getElementById('file-input'),
        clearSceneBtn: document.getElementById('clear-scene-btn'),
        demoSelect: document.getElementById('demo-select'),
        chatBox: document.getElementById('chat-box'),
        chatInput: document.getElementById('chat-input'),
        sendBtn: document.getElementById('send-btn'),
        modeSelect: document.getElementById('mode-select'),
        loaderOverlay: document.getElementById('loader-overlay'),
        loaderText: document.getElementById('loader-text'),
        navigateBtn: document.getElementById('navigate-btn'),
        videoModal: document.getElementById('video-modal'),
        videoModalClose: document.getElementById('video-modal-close'),
        navigationVideo: document.getElementById('navigation-video')
    };

    // --- 3D Scene State ---
    let mainPointCloud = null;
    let currentPointCloudFilepath = null;
    let currentMaskIndices = null;
    let lastUserActionTime = Date.now(); // Add a variable to track the last user action time

    // --- Manage All 3D Rendering Instances ---
    let renderInstances = []; // Main window and all chat inline windows

    // --- Initialization ---
    function init() {
        initMainViewer();
        initEventListeners();
        updateChatUIState();
        addMessage('Hello! Please select a demo file or upload your own point cloud file to start interacting.', 'system');
        animate();
    }

    // --- Initialize Main 3D Viewer ---
    function initMainViewer() {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        const { clientWidth, clientHeight } = dom.viewerElement;

        const camera = new THREE.PerspectiveCamera(60, clientWidth / clientHeight, 0.1, 1000);
        camera.position.set(5, 5, 5);
        camera.up.set(0, 0, 1);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(clientWidth, clientHeight);
        dom.viewerElement.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        renderInstances.push({ scene, camera, renderer, controls, element: dom.viewerElement });

        window.addEventListener('resize', onWindowResize, false);
        dom.viewerElement.addEventListener('mousedown', () => lastUserActionTime = Date.now());
        dom.viewerElement.addEventListener('touchstart', () => lastUserActionTime = Date.now());
        dom.viewerElement.addEventListener('mouseup', () => lastUserActionTime = Date.now());
        dom.viewerElement.addEventListener('touchend', () => lastUserActionTime = Date.now());
    }

    function initEventListeners() {
        dom.selectFileBtn.addEventListener('click', () => dom.fileInput.click());
        dom.fileInput.addEventListener('change', handleFileSelect);
        dom.clearSceneBtn.addEventListener('click', () => {
            if (mainPointCloud) resetCamera(renderInstances[0], mainPointCloud);
        });
        dom.demoSelect.addEventListener('change', (e) => loadDemoFile(e.target.value));
        dom.sendBtn.addEventListener('click', sendMessage);
        dom.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
        dom.navigateBtn.addEventListener('click', getNavigationVideo);
        dom.videoModalClose.addEventListener('click', () => {
            dom.navigationVideo.pause();
            dom.navigationVideo.src = '';
            dom.videoModal.style.display = 'none';
        });
    }

    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file || !file.name.toLowerCase().endsWith('.ply')) return;
        showLoader('Uploading file...');
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('/upload_ply', { method: 'POST', body: formData });
            const result = await response.json();
            if (result.success) {
                loadPointCloud(`/uploads/${result.filename}`, result.filename);
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            hideLoader();
            console.error(error);
        } finally {
            dom.fileInput.value = '';
        }
    }

    function loadDemoFile(filename) {
        if (!filename) return;
        loadPointCloud(`/uploads/${filename}`, filename);
    }

    function loadPointCloud(url, filename) {
        showLoader('Loading point cloud model...');
        const loader = new THREE.PLYLoader();
        loader.load(url,
            (geometry) => {
                cleanupAllScenes(); // Clean up old model and all inline windows
                geometry.computeBoundingBox();

                if (!geometry.attributes.color) {
                    const count = geometry.attributes.position.count;
                    const cols = new Float32Array(count * 3);
                    for (let i = 0; i < count; i++) cols.set([1, 1, 1], i * 3);
                    geometry.setAttribute('color', new THREE.BufferAttribute(cols, 3));
                }

                const material = new THREE.PointsMaterial({ size: 0.07, vertexColors: true });
                mainPointCloud = new THREE.Points(geometry, material);

                mainPointCloud.userData.originalColors = new Float32Array(geometry.attributes.color.array.slice());

                renderInstances[0].scene.add(mainPointCloud);
                currentPointCloudFilepath = filename;
                resetCamera(renderInstances[0], mainPointCloud);
                updateChatUIState();
                addMessage(`Point cloud file loaded successfully: ${filename}`, 'system');
                hideLoader();
            },
            (xhr) => showLoader(`Loading model... ${Math.round(xhr.loaded / xhr.total * 100)}%`),
            (error) => {
                hideLoader();
                console.error('Error loading point cloud file:', error);
            }
        );
    }

    async function sendMessage() {
        const query = dom.chatInput.value.trim();
        if (!query || !currentPointCloudFilepath) return;

        addMessage(query, 'user');
        dom.chatInput.value = '';
        const loadingMsg = addMessage('Thinking...', 'system');

        try {
            const response = await fetch('/chat_api', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: dom.modeSelect.value, query: query, filepath: currentPointCloudFilepath })
            });
            loadingMsg.remove();
            if (!response.ok) throw new Error(`Server response error: ${response.statusText}`);
            const result = await response.json();
            handleApiResponse(result);
        } catch (error) {
            loadingMsg.remove();
            addMessage(`Request failed: ${error.message}`, 'system');
        }
    }

    async function handleApiResponse(data) {
        let responseText = data.response;
        dom.navigateBtn.disabled = true; // Ensure the button is disabled after each new request

        if (data.type === 'locate') {
            currentMaskIndices = data.indices;
            if (data.indices && data.indices.length > 0) {
                highlightPointsInMainViewer(data.indices);
                const viewerHtml = createInlineViewerHtml(data.indices);
                responseText += `<p>Target point cloud:</p>${viewerHtml}`;

                if (data.video_pregeneration_started) {
                    addMessage(responseText, 'system');
                    await pollForNavigationVideoReady();
                    return;
                }
            }
        }
        addMessage(responseText, 'system');
    }

    async function pollForNavigationVideoReady() {
        // Add a status message to the chat box, replacing the full-screen loader
        const statusMessage = addMessage('Navigation video is being generated in the background, please wait...', 'system');

        let status = 'generating';
        let videoUrl = null;

        while (status === 'generating') {
            try {
                const response = await fetch('/get_navigation_video', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const result = await response.json();
                status = result.status;

                if (status === 'ready') {
                    videoUrl = result.video_url;
                    break;
                } else if (status === 'error') {
                    throw new Error(result.error || 'Navigation video generation failed.');
                }
            } catch (error) {
                statusMessage.remove();
                addMessage(`Failed to get navigation video: ${error.message}`, 'system');
                return;
            }

            await new Promise(resolve => setTimeout(resolve, 3000)); // Query every 3 seconds
        }

        if (videoUrl) {
            statusMessage.remove(); // Remove the progress message
            addMessage('Navigation video is ready, please click the "Navigate" button to view.', 'system');
            dom.navigateBtn.disabled = false;
        }
    }

    function createInlineViewerHtml(indices) {
        const viewerId = `inline-viewer-${Date.now()}`;
        // Hardcode the container size to a square, e.g., 300x300
        const viewerHtml = `<div class="inline-viewer-container" id="${viewerId}" style="width:300px; height:300px;"></div>`;
        // Use setTimeout to ensure the DOM element has been created
        setTimeout(() => createInlineViewerInstance(viewerId, indices), 0);
        return viewerHtml;
    }

    function createInlineViewerInstance(viewerId, indices) {
        const viewerDiv = document.getElementById(viewerId);
        if (!viewerDiv) return;

        const { clientWidth, clientHeight } = viewerDiv;
        // Use a fixed aspect ratio of 1, because the container is already square
        const aspectRatio = 1;

        const positions = mainPointCloud.geometry.attributes.position.array;
        const colors = mainPointCloud.userData.originalColors;

        const subPositions = new Float32Array(indices.length * 3);
        const subColors = new Float32Array(indices.length * 3);

        indices.forEach((index, i) => {
            subPositions.set(positions.slice(index * 3, index * 3 + 3), i * 3);
            subColors.set(colors.slice(index * 3, index * 3 + 3), i * 3);
        });

        const subGeometry = new THREE.BufferGeometry();
        subGeometry.setAttribute('position', new THREE.BufferAttribute(subPositions, 3));
        subGeometry.setAttribute('color', new THREE.BufferAttribute(subColors, 3));

        const subMaterial = new THREE.PointsMaterial({ size: 0.05, vertexColors: true });
        const subPointCloud = new THREE.Points(subGeometry, subMaterial);

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xFFFFFF);
        scene.add(subPointCloud);

        const camera = new THREE.PerspectiveCamera(50, aspectRatio, 0.01, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(clientWidth, clientHeight);
        viewerDiv.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const newInstance = { scene, camera, renderer, controls, element: viewerDiv, isInline: true };
        renderInstances.push(newInstance);
        resetCamera(newInstance, subPointCloud);
    }

    async function getNavigationVideo() {
        if (!currentPointCloudFilepath || !currentMaskIndices || currentMaskIndices.length === 0) {
            addMessage('Please perform a successful localization first to unlock the navigation feature.', 'system');
            return;
        }

        try {
            const response = await fetch('/get_navigation_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const result = await response.json();
            if (result.status === 'ready') {
                dom.navigationVideo.src = result.video_url;
                dom.navigationVideo.load();
                dom.navigationVideo.play();
                dom.videoModal.style.display = 'flex';
            } else if (result.status === 'error') {
                addMessage(`Failed to get navigation video: ${result.error}`, 'system');
            } else {
                addMessage('Navigation video has not been generated yet, please try again later.', 'system');
            }
        } catch (error) {
            addMessage(`Failed to get navigation video: ${error.message}`, 'system');
        }
    }


    function highlightPointsInMainViewer(indices) {
        const geom = mainPointCloud.geometry;
        const colors = geom.attributes.color.array;
        const red = new THREE.Color(1, 0, 0);

        // Reset colors first, in case there were previous highlights
        const original = mainPointCloud.userData.originalColors;
        for (let i = 0; i < colors.length; i++) {
            colors[i] = original[i];
        }

        indices.forEach(i => colors.set([red.r, red.g, red.b], i * 3));
        geom.attributes.color.needsUpdate = true;
    }

    function resetCamera(instance, object) {
        const { camera, controls } = instance;
        object.geometry.computeBoundingBox();
        const box = object.geometry.boundingBox;
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();

        camera.position.copy(center);
        camera.position.add(new THREE.Vector3(size * 0.7, size * 0.7, size * 1.2));
        camera.lookAt(center);
        controls.target.copy(center);
        controls.update();
    }

    function cleanupAllScenes() {
        for (let i = renderInstances.length - 1; i > 0; i--) {
            const inst = renderInstances[i];
            inst.scene.traverse(obj => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) Array.isArray(obj.material) ? obj.material.forEach(m => m.dispose()) : obj.material.dispose();
            });
            inst.renderer.dispose();
            if (inst.element && inst.element.parentNode) inst.element.parentNode.removeChild(inst.element);
        }
        renderInstances = [renderInstances[0]];

        if (mainPointCloud) {
            renderInstances[0].scene.remove(mainPointCloud);
            mainPointCloud.geometry.dispose();
            mainPointCloud.material.dispose();
            mainPointCloud = null;
        }
        dom.chatBox.innerHTML = '';
        dom.navigateBtn.disabled = true;
    }

    function animate() {
        requestAnimationFrame(animate);
        renderInstances.forEach(inst => {
            if (inst.controls) inst.controls.update();
            if (inst.renderer && inst.scene && inst.camera) inst.renderer.render(inst.scene, inst.camera);
        });

        // Add auto-rotation for the main viewer
        const mainInstance = renderInstances[0];
        const elapsedTime = Date.now() - lastUserActionTime;
        const autoRotateDelay = 5000; // 3 second delay

        if (mainInstance) {
            if (elapsedTime > autoRotateDelay) {
                mainInstance.controls.autoRotate = true;
                mainInstance.controls.autoRotateSpeed = 0.2;
            } else {
                mainInstance.controls.autoRotate = false;
            }
        }
    }

    function onWindowResize() {
        renderInstances.forEach(inst => {
            const { element, camera, renderer } = inst;
            const { clientWidth, clientHeight } = element;
            if (clientWidth === 0 || clientHeight === 0) return;
            camera.aspect = clientWidth / clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(clientWidth, clientHeight);
        });
    }

    // Updated addMessage function
    function addMessage(text, type) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message-wrapper ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = type === 'user' ? 'U' : 'AI';

        const contentBox = document.createElement('div');
        contentBox.className = 'message-content';

        // Check if the text contains HTML structure, especially the inline 3D viewer container
        if (text.includes('<div class="inline-viewer-container"')) {
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = text;
            Array.from(tempDiv.children).forEach(child => contentBox.appendChild(child));
        } else {
            contentBox.textContent = text;
        }

        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(contentBox);
        dom.chatBox.appendChild(messageWrapper);
        dom.chatBox.scrollTop = dom.chatBox.scrollHeight;
        return messageWrapper;
    }

    function updateChatUIState() {
        const enabled = !!currentPointCloudFilepath;
        dom.chatInput.disabled = !enabled;
        dom.sendBtn.disabled = !enabled;
        dom.modeSelect.disabled = !enabled;
        dom.chatInput.placeholder = enabled ? 'Enter message...' : 'Please load a point cloud file first';
    }

    function showLoader(text) { dom.loaderText.textContent = text; dom.loaderOverlay.style.display = 'flex'; }
    function hideLoader() { dom.loaderOverlay.style.display = 'none'; }

    init();
});