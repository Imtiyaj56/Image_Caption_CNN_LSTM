const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const generateBtn = document.getElementById('generate-btn');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const loader = document.getElementById('loader');
const resultSection = document.getElementById('result-section');
const captionText = document.getElementById('caption-text');
const copyBtn = document.getElementById('copy-btn');
const placeholderInfo = document.getElementById('placeholder-info');

let currentFile = null;

// Click to upload
dropZone.addEventListener('click', (e) => {
    if (e.target !== removeBtn && !currentFile) {
        fileInput.click();
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

// Drag and drop
['dragover', 'dragenter'].forEach(type => {
    dropZone.addEventListener(type, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
});

['dragleave', 'dragend', 'drop'].forEach(type => {
    dropZone.addEventListener(type, () => {
        dropZone.classList.remove('drag-over');
    });
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewContainer.classList.remove('hidden');
        generateBtn.disabled = false;
        resultSection.classList.add('hidden');
        placeholderInfo.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    previewContainer.classList.add('hidden');
    generateBtn.disabled = true;
    resultSection.classList.add('hidden');
    placeholderInfo.classList.remove('hidden');
});

// Generate Caption
generateBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Loading state
    generateBtn.disabled = true;
    loader.classList.remove('hidden');
    generateBtn.querySelector('span').textContent = 'Analyzing Neural Data...';
    resultSection.classList.add('hidden');
    placeholderInfo.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();

        // Show result
        captionText.textContent = data.caption || 'Neural engine could not resolve a caption.';
        resultSection.classList.remove('hidden');

        // Scroll to result (for mobile mainly)
        if (window.innerWidth < 1024) {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

    } catch (error) {
        console.error(error);
        alert(`Precision Analysis Error: ${error.message}`);
        placeholderInfo.classList.remove('hidden');
    } finally {
        generateBtn.disabled = false;
        loader.classList.add('hidden');
        generateBtn.querySelector('span').textContent = 'Analyze Image';
    }
});

// Copy to clipboard
copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(captionText.textContent);
    const originalSvg = copyBtn.innerHTML;
    copyBtn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    copyBtn.style.color = 'var(--success)';

    setTimeout(() => {
        copyBtn.innerHTML = originalSvg;
        copyBtn.style.color = '';
    }, 2000);
});
