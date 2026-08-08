// ---------------------------------------------------------------------
// Upload bay: click / drag-drop preview
// ---------------------------------------------------------------------
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('image');
const bayEmpty = document.getElementById('bay-empty');
const bayPreview = document.getElementById('bay-preview');
const previewImg = document.getElementById('preview-img');
const previewFilename = document.getElementById('preview-filename');

function showPreview(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewFilename.textContent = file.name;
    bayEmpty.style.display = 'none';
    bayPreview.style.display = 'flex';
    bayPreview.style.flexDirection = 'column';
    bayPreview.style.alignItems = 'center';
  };
  reader.readAsDataURL(file);
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
  });
}

if (dropZone) {
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });
  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      showPreview(file);
    }
  });
}

// ---------------------------------------------------------------------
// Scan overlay on submit
// ---------------------------------------------------------------------
const uploadForm = document.getElementById('upload-form');
const scanOverlay = document.getElementById('scan-overlay');

if (uploadForm) {
  uploadForm.addEventListener('submit', (e) => {
    if (!fileInput.files || !fileInput.files[0]) return; // let native validation handle it
    scanOverlay.classList.add('active');
  });
}

// ---------------------------------------------------------------------
// Filmstrip viewer
// ---------------------------------------------------------------------
const viewerImg = document.getElementById('viewer-img');
const viewerTag = document.getElementById('viewer-tag');
const thumbs = document.querySelectorAll('.film-thumb');

thumbs.forEach(thumb => {
  thumb.addEventListener('click', () => {
    thumbs.forEach(t => t.classList.remove('active'));
    thumb.classList.add('active');
    viewerImg.src = thumb.dataset.src;
    viewerTag.textContent = thumb.dataset.tag;
  });
});
