// ========================================
// IndexedDB video storage (avoids sessionStorage size limit)
// ========================================
const _IDB_NAME    = 'oaklandDumping';
const _IDB_STORE   = 'pendingVideos';

function _openDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(_IDB_NAME, 1);
        req.onupgradeneeded = e => e.target.result.createObjectStore(_IDB_STORE, { keyPath: 'id' });
        req.onsuccess = e => resolve(e.target.result);
        req.onerror   = e => reject(e.target.error);
    });
}

async function saveVideosForProcessing(videoFiles) {
    const db = await _openDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_IDB_STORE);
        store.clear();
        videoFiles.forEach((v, i) => store.put({ id: i, name: v.name, file: v.file }));
        tx.oncomplete = () => resolve();
        tx.onerror    = e => reject(e.target.error);
    });
}

// ========================================

const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const fileList = document.getElementById('fileList');
const selectedFiles = document.getElementById('selectedFiles');
const submitBtn = document.getElementById('submitBtn');
const processBtn = document.getElementById('processBtn');
const imageGrid = document.getElementById('imageGrid');
const galleryEmpty = document.getElementById('galleryEmpty');
const metadataOverlay = document.getElementById('metadataOverlay');
const metadataCloseBtn = document.getElementById('metadataCloseBtn');
const metadataCancelBtn = document.getElementById('metadataCancelBtn');
const metadataSaveBtn = document.getElementById('metadataSaveBtn');
const dialogCameraSelect = document.getElementById('dialogCameraSelect');
const dialogCoordinatesBox = document.getElementById('dialogCoordinatesBox');
const dialogCameraArea = document.getElementById('dialogCameraArea');
const dialogDateSelect = document.getElementById('dialogDateSelect');
const dialogTimeSelect = document.getElementById('dialogTimeSelect');

let stagedFiles = [];
let galleryItems = []; // { name, src, type, file, metadata }
let cameras = [];
let dateOptions = [];
let timeOptions = [];
let activeMetadataIndex = -1;

initializeMetadata();

// File input change
fileInput.addEventListener('change', e => addFiles([...e.target.files]));

// Drag & drop
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = [...e.dataTransfer.files].filter(isSupportedMedia);
    addFiles(files);
});

function addFiles(files) {
    const supportedFiles = files.filter(isSupportedMedia);
    const rejectedCount = files.length - supportedFiles.length;

    if (rejectedCount > 0) {
        showToast(`${rejectedCount} file${rejectedCount > 1 ? 's' : ''} ignored (images/videos only)`, 'error');
    }

    if (supportedFiles.length === 0) return;

    const batchCategory = getBatchMediaCategory(supportedFiles);
    if (!batchCategory) {
        showToast('Mixed image/video batches are not allowed', 'error');
        return;
    }

    const lockedCategory = getLockedMediaCategory();
    if (lockedCategory && lockedCategory !== batchCategory) {
        showToast(`Only ${lockedCategory} files can be added right now`, 'error');
        return;
    }

    stagedFiles = [...stagedFiles, ...supportedFiles].slice(0, 10);
    renderFileList();
}

function renderFileList() {
    fileList.innerHTML = '';
    if (stagedFiles.length === 0) {
        selectedFiles.style.display = 'none';
        submitBtn.disabled = true;
        return;
    }
    selectedFiles.style.display = 'block';
    submitBtn.disabled = false;
    stagedFiles.forEach((f, i) => {
        const fileIcon = f.type.startsWith('video/') ? '🎬' : '🖼️';
        const chip = document.createElement('div');
        chip.className = 'file-chip';
        chip.innerHTML = `<span class="file-chip-name">${fileIcon} ${f.name}</span><span class="file-chip-remove" onclick="removeFile(${i})">✕</span>`;
        fileList.appendChild(chip);
    });
}

function removeFile(index) {
    stagedFiles.splice(index, 1);
    renderFileList();
}

function handleSubmit() {
    if (stagedFiles.length === 0) return;

    stagedFiles.forEach(file => {
        const src = URL.createObjectURL(file);
        addToGallery(file.name, src, file.type, file, createEmptyMetadata());
    });

    showToast(`✅ ${stagedFiles.length} media file${stagedFiles.length > 1 ? 's' : ''} submitted`, 'success');
    stagedFiles = [];
    fileInput.value = '';
    renderFileList();
}

function addToGallery(name, src, type, file, metadata) {
    galleryItems.push({ name, src, type, file, metadata });
    renderGallery();
}

function removeGalleryItem(index) {
    if (index < 0 || index >= galleryItems.length) return;
    URL.revokeObjectURL(galleryItems[index].src);
    galleryItems.splice(index, 1);

    if (activeMetadataIndex === index) {
        closeMetadataDialog();
    } else if (activeMetadataIndex > index) {
        activeMetadataIndex -= 1;
    }

    renderGallery();
}

function renderGallery() {
    imageGrid.innerHTML = '';
    if (galleryItems.length === 0) {
        imageGrid.style.display = 'none';
        galleryEmpty.style.display = 'flex';
        updateProcessButtonState();
        return;
    }
    imageGrid.style.display = 'grid';
    galleryEmpty.style.display = 'none';
    galleryItems.forEach(({ name, src, type }, i) => {
        const thumb = document.createElement('div');
        thumb.className = 'image-thumb';
        const isVideo = type.startsWith('video/');
        const metadataComplete = isMetadataComplete(galleryItems[i].metadata);
        const metadataIcon = metadataComplete ? '✓' : '!';
        const metadataClass = metadataComplete ? 'complete' : 'incomplete';
        const mediaElement = isVideo
            ? `<video src="${src}" muted playsinline preload="metadata"></video>`
            : `<img src="${src}" alt="${name}">`;
        thumb.innerHTML = `
            ${mediaElement}
            <div class="thumb-label">${name}</div>
            <div class="thumb-remove" title="Remove" onclick="event.stopPropagation(); removeGalleryItem(${i})">✕</div>
            <div class="metadata-indicator ${metadataClass}" title="Edit metadata" onclick="event.stopPropagation(); openMetadataDialog(${i})">${metadataIcon}</div>
        `;
        thumb.onclick = () => window.open(src, '_blank');
        imageGrid.appendChild(thumb);
    });

    updateProcessButtonState();
}

function isSupportedMedia(file) {
    return file.type.startsWith('image/') || file.type.startsWith('video/');
}

function getMediaCategory(mimeType) {
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('video/')) return 'video';
    return '';
}

function getBatchMediaCategory(files) {
    const categories = [...new Set(files.map(f => getMediaCategory(f.type)))].filter(Boolean);
    return categories.length === 1 ? categories[0] : '';
}

function getLockedMediaCategory() {
    if (stagedFiles.length > 0) return getMediaCategory(stagedFiles[0].type);
    if (galleryItems.length > 0) return getMediaCategory(galleryItems[0].type);
    return '';
}

function showToast(msg, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    setTimeout(() => toast.className = 'toast', 3000);
}

async function initializeMetadata() {
    dateOptions = buildDateOptions();
    timeOptions = buildTimeOptions();
    await loadCameras();
}

async function loadCameras() {
    try {
        const response = await fetch('../assets/cameras.json');
        if (!response.ok) throw new Error('Failed to load camera data');
        cameras = await response.json();
    } catch (error) {
        showToast('Unable to load camera metadata', 'error');
    }
}

function buildDateOptions() {
    const options = [];
    const today = new Date();
    for (let offset = 0; offset < 30; offset += 1) {
        const date = new Date(today);
        date.setDate(today.getDate() - offset);
        options.push({ value: formatDate(date), label: formatDateLabel(date) });
    }
    return options;
}

function buildTimeOptions() {
    const options = [];
    for (let hour = 0; hour < 24; hour += 1) {
        for (let minute = 0; minute < 60; minute += 30) {
            const value = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
            options.push({ value, label: value });
        }
    }
    return options;
}

function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function formatDateLabel(date) {
    return date.toLocaleDateString('en-US', {
        weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'
    });
}

function openMetadataDialog(index) {
    const item = galleryItems[index];
    if (!item) return;

    activeMetadataIndex = index;
    populateDialogCameraOptions();
    populateSelectOptions(dialogDateSelect, dateOptions, 'Select date');
    populateSelectOptions(dialogTimeSelect, timeOptions, 'Select time');

    dialogCameraSelect.value = item.metadata.cameraIndex;
    dialogDateSelect.value = item.metadata.date;
    dialogTimeSelect.value = item.metadata.time;
    updateDialogCoordinateDisplay(item.metadata);

    metadataOverlay.classList.add('open');
    metadataOverlay.setAttribute('aria-hidden', 'false');
}

function closeMetadataDialog() {
    activeMetadataIndex = -1;
    metadataOverlay.classList.remove('open');
    metadataOverlay.setAttribute('aria-hidden', 'true');
}

function populateDialogCameraOptions() {
    populateSelectOptions(
        dialogCameraSelect,
        cameras.map((camera, index) => ({ value: String(index), label: camera.name })),
        cameras.length > 0 ? 'Select camera' : 'Camera data unavailable'
    );
}

function populateSelectOptions(selectElement, options, placeholder) {
    selectElement.innerHTML = '';
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = placeholder;
    selectElement.appendChild(defaultOption);
    options.forEach(({ value, label }) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        selectElement.appendChild(option);
    });
}

function updateDialogCoordinateDisplay(metadata) {
    if (metadata.cameraIndex === '') {
        dialogCoordinatesBox.textContent = 'Select a camera to view coordinates';
        dialogCameraArea.textContent = '';
        return;
    }
    const camera = cameras[Number(metadata.cameraIndex)];
    if (!camera) {
        dialogCoordinatesBox.textContent = 'Camera not found';
        dialogCameraArea.textContent = '';
        return;
    }
    dialogCoordinatesBox.textContent = `${camera.latitude}, ${camera.longitude}`;
    dialogCameraArea.textContent = `Area: ${camera.area}`;
}

function createEmptyMetadata() {
    return { cameraIndex: '', latitude: '', longitude: '', area: '', date: '', time: '' };
}

function isMetadataComplete(metadata) {
    return metadata.cameraIndex !== '' && metadata.date !== '' && metadata.time !== '';
}

function updateProcessButtonState() {
    const hasMedia = galleryItems.length > 0;
    const allComplete = hasMedia && galleryItems.every(item => isMetadataComplete(item.metadata));
    processBtn.disabled = !allComplete;
}

dialogCameraSelect.addEventListener('change', event => {
    if (activeMetadataIndex < 0) return;
    const selectedIndex = event.target.value;
    const metadata = galleryItems[activeMetadataIndex].metadata;
    metadata.cameraIndex = selectedIndex;
    if (selectedIndex === '') {
        metadata.latitude = '';
        metadata.longitude = '';
        metadata.area = '';
    } else {
        const camera = cameras[Number(selectedIndex)];
        metadata.latitude = String(camera.latitude);
        metadata.longitude = String(camera.longitude);
        metadata.area = camera.area;
    }
    updateDialogCoordinateDisplay(metadata);
});

dialogDateSelect.addEventListener('change', event => {
    if (activeMetadataIndex < 0) return;
    galleryItems[activeMetadataIndex].metadata.date = event.target.value;
});

dialogTimeSelect.addEventListener('change', event => {
    if (activeMetadataIndex < 0) return;
    galleryItems[activeMetadataIndex].metadata.time = event.target.value;
});

metadataSaveBtn.addEventListener('click', () => {
    if (activeMetadataIndex < 0) return;
    renderGallery();
    closeMetadataDialog();
});

metadataCancelBtn.addEventListener('click', closeMetadataDialog);
metadataCloseBtn.addEventListener('click', closeMetadataDialog);

metadataOverlay.addEventListener('click', event => {
    if (event.target === metadataOverlay) closeMetadataDialog();
});

processBtn.addEventListener('click', async () => {
    const mediaCategory = getLockedMediaCategory();

    if (!mediaCategory) {
        showToast('No uploaded media to process', 'error');
        return;
    }

    if (galleryItems.some(item => !isMetadataComplete(item.metadata))) {
        showToast('Complete metadata for all media before processing', 'error');
        return;
    }

    const mediaType = mediaCategory === 'video' ? 'dynamic' : 'static';
    const destinationPage = mediaType === 'dynamic' ? 'processed-dynamic.html' : 'processed-static.html';
    const now = new Date();
    const sessionId = `session-${formatSessionTimestamp(now)}`;

    // 1. Read all files into base64 dataUrls
    let mediaEntries = [];
    try {
        mediaEntries = await Promise.all(
            galleryItems.map(item => new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve({ name: item.name, dataUrl: reader.result });
                reader.onerror = reject;
                reader.readAsDataURL(item.file);
            }))
        );
    } catch (err) {
        showToast('Failed to prepare media for processing', 'error');
        console.error(err);
        return;
    }

    // 2. Save to the correct storage depending on media type
    if (mediaType === 'static') {
        // Images are small enough for sessionStorage
        sessionStorage.setItem('lastSessionImages', JSON.stringify(mediaEntries));
    } else {
        // Videos are too large for sessionStorage — use IndexedDB instead
        try {
            await saveVideosForProcessing(
                galleryItems.map(item => ({ name: item.name, file: item.file }))
            );
        } catch (err) {
            showToast('Failed to store video for processing', 'error');
            console.error(err);
            return;
        }
    }

    // 3. Save to local folder if directory picker is available
    if (typeof window.showDirectoryPicker === 'function') {
        try {
            const submissionFolder = `${mediaType}-submission`;
            const rootHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
            const submissionHandle = await getOrCreateDirectory(rootHandle, submissionFolder);
            const sessionHandle = await getOrCreateDirectory(submissionHandle, sessionId);

            const usedFilenames = new Set();
            const items = [];

            for (let i = 0; i < galleryItems.length; i += 1) {
                const item = galleryItems[i];
                const originalName = item.file?.name || item.name;
                const baseName = sanitizeFilename(originalName);
                const outputName = ensureUniqueFilename(baseName, usedFilenames);
                usedFilenames.add(outputName);

                await writeFileToDirectory(sessionHandle, outputName, item.file);

                // Only run detection for static images at this stage
                let trashCount = 0;
                let avgConfidence = 0.0;
                if (mediaType === 'static') {
                    try {
                        const formData = new FormData();
                        formData.append('image', item.file, originalName);
                        formData.append('min_confidence', '0.5');
                        const res = await fetch('http://127.0.0.1:8000/api/detect', { method: 'POST', body: formData });
                        if (res.ok) {
                            const data = await res.json();
                            trashCount = data.count;
                            avgConfidence = data.average_confidence;
                        } else {
                            console.warn(`Detection returned ${res.status} for ${originalName}`);
                        }
                    } catch (detErr) {
                        console.warn(`Detection failed for ${originalName}:`, detErr);
                    }
                }

                const unprocessedMediaPath = `${submissionFolder}/${sessionId}/${outputName}`;
                const mediaDateEpoch = toEpochSeconds(item.metadata.date, item.metadata.time);
                const mediaObj = new Media(
                    unprocessedMediaPath,
                    mediaDateEpoch,
                    Number(item.metadata.latitude),
                    Number(item.metadata.longitude),
                    avgConfidence,
                    trashCount
                );

                items.push({ mediaId: `m${i + 1}`, ...mediaObj.toJSON() });
            }

            const manifest = {
                sessionId,
                mediaType,
                createdAtEpoch: Math.floor(now.getTime() / 1000),
                items
            };

            const manifestName = `${sessionId}.json`;
            const manifestRelativePath = `${submissionFolder}/${sessionId}/${manifestName}`;
            await writeFileToDirectory(
                sessionHandle,
                manifestName,
                new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' })
            );

            sessionStorage.setItem('lastSessionManifest', JSON.stringify(manifest));
            sessionStorage.setItem('lastSessionPath', manifestRelativePath);
            showToast(`Session saved: ${manifestRelativePath}`, 'success');

        } catch (error) {
            if (error?.name === 'AbortError') {
                showToast('Folder save skipped', 'error');
            } else {
                showToast('Failed to save session folder', 'error');
                console.error(error);
            }
        }
    }

    // 4. Always redirect
    setTimeout(() => {
        window.location.href = destinationPage;
    }, 800);
});

function formatSessionTimestamp(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${year}${month}${day}-${hours}${minutes}${seconds}`;
}

function sanitizeFilename(filename) {
    const cleaned = filename.replace(/[<>:"/\\|?*\x00-\x1F]/g, '_').trim();
    return cleaned || 'media';
}

function ensureUniqueFilename(filename, usedNames) {
    if (!usedNames.has(filename)) return filename;
    const dotIndex = filename.lastIndexOf('.');
    const hasExtension = dotIndex > 0;
    const stem = hasExtension ? filename.slice(0, dotIndex) : filename;
    const ext = hasExtension ? filename.slice(dotIndex) : '';
    let counter = 1;
    let candidate = `${stem}-${counter}${ext}`;
    while (usedNames.has(candidate)) {
        counter += 1;
        candidate = `${stem}-${counter}${ext}`;
    }
    return candidate;
}

function toEpochSeconds(dateValue, timeValue) {
    const timestamp = new Date(`${dateValue}T${timeValue}:00`);
    if (Number.isNaN(timestamp.getTime())) return Math.floor(Date.now() / 1000);
    return Math.floor(timestamp.getTime() / 1000);
}

async function getOrCreateDirectory(parentHandle, dirName) {
    return parentHandle.getDirectoryHandle(dirName, { create: true });
}

async function writeFileToDirectory(directoryHandle, filename, content) {
    const fileHandle = await directoryHandle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(content);
    await writable.close();
}