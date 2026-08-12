// ---------------------------------------------------------------------
// Upload bay: click / drag-drop preview
// ---------------------------------------------------------------------
const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('image');
const bayEmpty     = document.getElementById('bay-empty');
const bayPreview   = document.getElementById('bay-preview');
const previewImg   = document.getElementById('preview-img');
const previewFilename = document.getElementById('preview-filename');

function showPreview(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src        = e.target.result;
    previewFilename.textContent = file.name;
    bayEmpty.style.display   = 'none';
    bayPreview.style.display = 'flex';
    bayPreview.style.flexDirection = 'column';
    bayPreview.style.alignItems    = 'center';
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
// Scan overlay + progress message cycling
// FIX: form now uses fetch() instead of native submit so we can:
//   1. Keep the overlay alive for the full 60-120s inference time
//   2. Show rotating status messages so user knows it's working
//   3. Handle errors gracefully instead of a blank/stuck screen
//   4. Inject the result HTML when the server responds
// ---------------------------------------------------------------------
const uploadForm = document.getElementById('upload-form');
const scanOverlay = document.getElementById('scan-overlay');

// Grab the status text element inside your overlay
// (uses id="status-text" — add this id to the element showing
//  "RUNNING PIPELINE..." in your HTML if it doesn't have it already)
const statusText = document.getElementById('status-text');

// Messages that cycle every 8 seconds so user sees progress feedback
const PIPELINE_MESSAGES = [
  'RUNNING PIPELINE ...',
  'enhance → segment → classify → grad-cam',
  'Enhancing image quality ...',
  'Running U-Net segmentation ...',
  'Extracting damage region (ROI) ...',
  'Classifying damage type ...',
  'Generating Grad-CAM heatmap ...',
  'Computing severity score ...',
  'Almost there — building your report ...',
];

let _msgIndex    = 0;
let _msgInterval = null;

function startMessageCycle() {
  _msgIndex = 0;
  if (statusText) statusText.textContent = PIPELINE_MESSAGES[0];
  _msgInterval = setInterval(() => {
    _msgIndex = (_msgIndex + 1) % PIPELINE_MESSAGES.length;
    if (statusText) statusText.textContent = PIPELINE_MESSAGES[_msgIndex];
  }, 8000); // rotate every 8 seconds
}

function stopMessageCycle() {
  if (_msgInterval) { clearInterval(_msgInterval); _msgInterval = null; }
}

function showError(message) {
  stopMessageCycle();
  if (statusText) {
    statusText.textContent = message;
    statusText.style.color = '#ff4444';
  }
  // Show a retry button if it exists in your HTML
  const retryBtn = document.getElementById('retry-btn');
  if (retryBtn) retryBtn.style.display = 'inline-block';

  // After 4 seconds hide overlay so user can try again
  setTimeout(() => {
    if (scanOverlay) {
      scanOverlay.classList.remove('active');
      scanOverlay.setAttribute('aria-hidden', 'true');
    }
    if (statusText) statusText.style.color = ''; // reset colour
  }, 4000);
}

if (uploadForm) {
  uploadForm.addEventListener('submit', (e) => {
    // Stop the browser's native form submit
    e.preventDefault();

    // Validate: must have a file selected
    if (!fileInput || !fileInput.files || !fileInput.files[0]) return;

    // Show the scan overlay immediately
    if (scanOverlay) {
      scanOverlay.classList.add('active');
      scanOverlay.setAttribute('aria-hidden', 'false');
    }

    // Start cycling status messages
    startMessageCycle();

    // Build FormData from the form
    const formData = new FormData(uploadForm);

    // Abort controller — 3 minute timeout (180 000 ms)
    // Render free tier inference takes 60-120s — 3 min is safe headroom
    const controller = new AbortController();
    const timeoutId  = setTimeout(() => {
      controller.abort();
    }, 180000);

    fetch('/predict', {
      method: 'POST',
      body:   formData,
      signal: controller.signal,
    })
    .then(response => {
      clearTimeout(timeoutId);
      stopMessageCycle();

      if (!response.ok) {
        // Server returned 4xx / 5xx — read the body for the error message
        return response.text().then(text => {
          // Try to extract a meaningful message from Flask's HTML error page
          const match = text.match(/<p>(.*?)<\/p>/s);
          const msg   = match ? match[1].replace(/<[^>]+>/g, '').trim() : `Server error ${response.status}`;
          throw new Error(msg);
        });
      }

      return response.text();
    })
    .then(html => {
      // Success — replace the whole page with the results HTML
      // (your Flask /predict route returns a full rendered template)
      document.open();
      document.write(html);
      document.close();
    })
    .catch(err => {
      clearTimeout(timeoutId);
      stopMessageCycle();

      if (err.name === 'AbortError') {
        showError('Inference timed out (>3 min). Please try a smaller image or try again.');
      } else {
        showError('Error: ' + err.message);
      }
    });
  });
}

// ---------------------------------------------------------------------
// Filmstrip viewer
// ---------------------------------------------------------------------
const viewerImg = document.getElementById('viewer-img');
const viewerTag = document.getElementById('viewer-tag');
const thumbs    = document.querySelectorAll('.film-thumb');

thumbs.forEach(thumb => {
  thumb.addEventListener('click', () => {
    thumbs.forEach(t => t.classList.remove('active'));
    thumb.classList.add('active');
    viewerImg.src         = thumb.dataset.src;
    viewerTag.textContent = thumb.dataset.tag;
  });
});