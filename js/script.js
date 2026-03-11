const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const fileList = document.getElementById('fileList');
const selectedFiles = document.getElementById('selectedFiles');
const submitBtn = document.getElementById('submitBtn');
const imageGrid = document.getElementById('imageGrid');
const galleryEmpty = document.getElementById('galleryEmpty');

let stagedFiles = [];
let galleryItems = []; // { name, src }

// File input change
fileInput.addEventListener('change', e => addFiles([...e.target.files]));

// Drag & drop
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
    addFiles(files);
});

function addFiles(files) {
    const combined = [...stagedFiles, ...files].slice(0, 10);
    stagedFiles = combined;
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
        const chip = document.createElement('div');
        chip.className = 'file-chip';
        chip.innerHTML = `<span class="file-chip-name">📄 ${f.name}</span><span class="file-chip-remove" onclick="removeFile(${i})">✕</span>`;
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
        const reader = new FileReader();
        reader.onload = e => addToGallery(file.name, e.target.result);
        reader.readAsDataURL(file);
    });

    showToast(`✅ ${stagedFiles.length} image${stagedFiles.length > 1 ? 's' : ''} submitted`, 'success');
    stagedFiles = [];
    fileInput.value = '';
    renderFileList();
}

function addToGallery(name, src) {
    galleryItems.push({ name, src });
    renderGallery();
}

function removeGalleryItem(index) {
    galleryItems.splice(index, 1);
    renderGallery();
}

function renderGallery() {
    imageGrid.innerHTML = '';
    if (galleryItems.length === 0) {
        galleryEmpty.style.display = 'flex';
        return;
    }
    galleryEmpty.style.display = 'none';
    galleryItems.forEach(({ name, src }, i) => {
        const thumb = document.createElement('div');
        thumb.className = 'image-thumb';
        thumb.innerHTML = `
            <img src="${src}" alt="${name}">
            <div class="thumb-label">${name}</div>
            <div class="thumb-remove" title="Remove" onclick="event.stopPropagation(); removeGalleryItem(${i})">✕</div>
        `;
        thumb.querySelector('img').onclick = () => window.open(src, '_blank');
        imageGrid.appendChild(thumb);
    });
}

function showToast(msg, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast show ${type}`;
    setTimeout(() => toast.className = 'toast', 3000);
}