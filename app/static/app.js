const API_BASE = 'http://localhost:8000/api/v1';
const API_KEY = 'smart-retail-secret-api-key-2026';

let productStream = null;
let faceStream = null;
let lastCapturedFaceBlob = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initFileUploads();
    refreshDashboard();
});

// Navigation Tabs (Prevents any default form submission or page hash navigation reloads)
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPages = document.querySelectorAll('.tab-page');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            if (e) {
                e.preventDefault();
                e.stopPropagation();
            }
            const targetTab = item.getAttribute('data-tab');

            navItems.forEach(nav => nav.classList.remove('active'));
            tabPages.forEach(page => page.classList.remove('active'));

            item.classList.add('active');
            const targetPage = document.getElementById(`tab-${targetTab}`);
            if (targetPage) {
                targetPage.classList.add('active');
            }

            updateTabTitles(targetTab);
        });
    });
}

function updateTabTitles(tab) {
    const titleEl = document.getElementById('tabTitle');
    const subTitleEl = document.getElementById('tabSubtitle');

    const titles = {
        dashboard: { t: 'Retail Analytics Dashboard', s: 'Real-time store metrics, AI predictions, and customer intelligence' },
        vision: { t: 'Live Product Scanner', s: 'Scan product via live camera feed or upload image to classify' },
        face: { t: 'Face Loyalty & Customer Registration', s: 'Recognize registered store customers or register new members' },
        sentiment: { t: 'Sentiment Analyzer', s: 'Analyze customer review feedback using TF-IDF + Classifier' },
        chatbot: { t: 'AI Retail Chatbot', s: 'Hybrid FAQ rule-matching & ML fallback assistant' }
    };

    if (titles[tab]) {
        titleEl.textContent = titles[tab].t;
        subTitleEl.textContent = titles[tab].s;
    }
}

// Helper to safely parse API responses
async function parseApiResponse(res) {
    const text = await res.text();
    if (!res.ok) {
        let errorDetail = `HTTP Error ${res.status}`;
        try {
            const errObj = JSON.parse(text);
            if (errObj.detail) errorDetail = errObj.detail;
        } catch (e) {}
        throw new Error(errorDetail);
    }
    return JSON.parse(text);
}

// Refresh Dashboard Stats (Only updates stats text values, NEVER touches active tab state)
async function refreshDashboard() {
    try {
        const res = await fetch(`${API_BASE}/dashboard/stats`, {
            headers: { 'X-API-Key': API_KEY }
        });
        const data = await parseApiResponse(res);

        const elVisitors = document.getElementById('statTodayVisitors');
        if (elVisitors) elVisitors.textContent = data.today_visitors;
        
        const elReturning = document.getElementById('statReturningCustomers');
        if (elReturning) elReturning.textContent = data.returning_customers;

        const elRate = document.getElementById('statReturningRate');
        if (elRate) elRate.textContent = `${data.returning_rate}%`;
        
        const elTotalReviews = document.getElementById('statTotalReviews');
        if (elTotalReviews) elTotalReviews.textContent = data.total_reviews;

        const elPositive = document.getElementById('statPositiveReviews');
        if (elPositive) elPositive.textContent = data.positive_reviews;

        const elSentRate = document.getElementById('statPositiveSentimentRate');
        if (elSentRate) elSentRate.textContent = `${data.positive_sentiment_rate}%`;

        const elIntent = document.getElementById('statMostAskedIntent');
        if (elIntent) elIntent.textContent = data.most_asked_intent;

        const elCategory = document.getElementById('statMostPredictedCategory');
        if (elCategory) elCategory.textContent = data.most_predicted_product_category;
    } catch (err) {
        console.error('Failed to fetch dashboard stats:', err);
    }
}

// Open Drill-Down Details Modal
async function openDetailModal(category, title) {
    const modal = document.getElementById('detailModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');

    modalTitle.textContent = title;
    modalBody.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i> Fetching log records...</div>';
    modal.style.display = 'flex';

    try {
        const res = await fetch(`${API_BASE}/dashboard/details?category=${category}`, {
            headers: { 'X-API-Key': API_KEY }
        });
        const data = await parseApiResponse(res);

        if (!data.data || data.data.length === 0) {
            modalBody.innerHTML = '<div class="empty-state">No log records found for this metric.</div>';
            return;
        }

        // Render HTML Table of records
        const keys = Object.keys(data.data[0]);
        let headersHtml = keys.map(k => `<th>${k.toUpperCase()}</th>`).join('');
        let rowsHtml = data.data.map(row => {
            let cols = keys.map(k => `<td>${row[k]}</td>`).join('');
            return `<tr>${cols}</tr>`;
        }).join('');

        modalBody.innerHTML = `
            <div style="margin-bottom:0.75rem; font-size:0.85rem; color:#94a3b8;">Total Records Logged: <strong>${data.total_records}</strong></div>
            <table class="log-table">
                <thead><tr>${headersHtml}</tr></thead>
                <tbody>${rowsHtml}</tbody>
            </table>
        `;
    } catch (err) {
        modalBody.innerHTML = `<div class="empty-state" style="color:#ef4444;">Failed to load details: ${err.message}</div>`;
    }
}

function closeDetailModal() {
    const modal = document.getElementById('detailModal');
    if (modal) modal.style.display = 'none';
}

// File Upload Fallback Setup
function initFileUploads() {
    setupZone('productDropZone', 'productImageInput');
    setupZone('faceDropZone', 'faceImageInput');
}

function setupZone(zoneId, inputId) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    zone.addEventListener('click', (e) => {
        if (e) e.preventDefault();
        input.click();
    });
}

// --- Live Web Camera Controls ---
async function startProductCamera(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    try {
        const video = document.getElementById('productWebcam');
        const dropZone = document.getElementById('productDropZone');
        productStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = productStream;
        video.style.display = 'block';
        dropZone.style.display = 'none';
    } catch (err) {
        alert('Could not access camera: ' + err.message);
    }
}

async function startFaceCamera(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    try {
        const video = document.getElementById('faceWebcam');
        const dropZone = document.getElementById('faceDropZone');
        faceStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = faceStream;
        video.style.display = 'block';
        dropZone.style.display = 'none';
    } catch (err) {
        alert('Could not access camera: ' + err.message);
    }
}

function captureFrameBlob(videoEl, canvasEl) {
    const context = canvasEl.getContext('2d');
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    context.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    return new Promise(resolve => canvasEl.toBlob(resolve, 'image/jpeg', 0.9));
}

// --- 1. Product Classifier (Camera / Upload) ---
async function captureAndClassifyProduct(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    const video = document.getElementById('productWebcam');
    const canvas = document.getElementById('productCanvas');
    const fileInput = document.getElementById('productImageInput');
    const resultsArea = document.getElementById('productResultsArea');

    let blob = null;
    if (video.style.display !== 'none' && video.srcObject) {
        blob = await captureFrameBlob(video, canvas);
    } else if (fileInput.files && fileInput.files.length > 0) {
        blob = fileInput.files[0];
    } else {
        alert('Please start camera or select an image file first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', blob, 'scanned_product.jpg');

    resultsArea.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i> Scanning product...</div>';

    try {
        const res = await fetch(`${API_BASE}/vision/classify-product`, {
            method: 'POST',
            headers: { 'X-API-Key': API_KEY },
            body: formData
        });
        const data = await parseApiResponse(res);

        let top3Html = data.top_3_predictions.map(p => `
            <div style="display:flex; justify-content:space-between; margin-top:0.4rem; padding:0.4rem; background:rgba(15,23,42,0.5); border-radius:6px;">
                <span>${p.category}</span>
                <strong>${p.confidence}%</strong>
            </div>
        `).join('');

        resultsArea.innerHTML = `
            <div style="text-align:center; padding:1rem;">
                <span style="font-size:0.85rem; color:#94a3b8;">Scanned Product Category</span>
                <h2 style="color:#10b981; margin:0.3rem 0; font-size:1.6rem;">${data.predicted_category}</h2>
                <div style="font-size:1.1rem; font-weight:700; color:#3b82f6;">Confidence: ${data.confidence_score}%</div>
            </div>
            <div style="margin-top:1rem;">
                <h4 style="font-size:0.9rem; color:#94a3b8;">Top-3 Predictions Ranking:</h4>
                ${top3Html}
            </div>
        `;
    } catch (err) {
        resultsArea.innerHTML = `<div class="empty-state" style="color:#ef4444;">${err.message}</div>`;
    }
}

// --- 2. Face Scanner & Registration ---
async function captureAndRecognizeFace(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    const video = document.getElementById('faceWebcam');
    const canvas = document.getElementById('faceCanvas');
    const fileInput = document.getElementById('faceImageInput');
    const resultsArea = document.getElementById('faceResultsArea');

    let blob = null;
    if (video.style.display !== 'none' && video.srcObject) {
        blob = await captureFrameBlob(video, canvas);
    } else if (fileInput.files && fileInput.files.length > 0) {
        blob = fileInput.files[0];
    } else {
        alert('Please start camera or select an image file first.');
        return;
    }

    lastCapturedFaceBlob = blob;

    const formData = new FormData();
    formData.append('file', blob, 'scanned_face.jpg');

    resultsArea.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i> Scanning face encodings...</div>';

    try {
        const res = await fetch(`${API_BASE}/vision/recognize-face`, {
            method: 'POST',
            headers: { 'X-API-Key': API_KEY },
            body: formData
        });
        const data = await parseApiResponse(res);

        const isReturning = data.customer_status === 'Returning Customer';
        
        if (isReturning) {
            resultsArea.innerHTML = `
                <div style="text-align:center; padding:1.5rem;">
                    <i class="fa-solid fa-circle-check" style="font-size:3.5rem; color:#10b981; margin-bottom:0.75rem;"></i>
                    <span style="font-size:0.85rem; color:#94a3b8;">Customer Recognized</span>
                    <h2 style="color:#10b981; margin:0.4rem 0; font-size:1.8rem;">${data.customer_id}</h2>
                    <div style="margin-top:0.5rem; color:#94a3b8;">Database Similarity Match: ${data.confidence_score}%</div>
                </div>
            `;
        } else {
            resultsArea.innerHTML = `
                <div style="text-align:center; padding:1rem;">
                    <i class="fa-solid fa-user-xmark" style="font-size:2.5rem; color:#f59e0b; margin-bottom:0.5rem;"></i>
                    <h3 style="color:#f59e0b; margin-bottom:0.25rem;">Customer Not Found</h3>
                    <p style="color:#94a3b8; font-size:0.85rem;">This face embedding does not match any registered member in the database.</p>
                    
                    <div style="margin-top:1.25rem; padding:1rem; background:rgba(15,23,42,0.6); border-radius:8px; border:1px solid #334155; text-align:left;">
                        <h4 style="margin-bottom:0.5rem; font-size:0.95rem; color:#3b82f6;"><i class="fa-solid fa-user-plus"></i> Save New Customer to Database</h4>
                        <p style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.75rem;">Enter customer name to save their face encoding into database:</p>
                        <input type="text" id="regCustomerName" placeholder="Enter Customer Name (e.g. Sarah Jenkins)" style="width:100%; padding:0.65rem; background:#0f172a; border:1px solid #334155; color:#fff; border-radius:6px; margin-bottom:0.75rem;">
                        <button type="button" class="btn btn-block btn-success" onclick="registerNewCustomer(event)"><i class="fa-solid fa-floppy-disk"></i> Save Customer Face</button>
                    </div>
                </div>
            `;
        }
    } catch (err) {
        resultsArea.innerHTML = `<div class="empty-state" style="color:#ef4444;">${err.message}</div>`;
    }
}

async function registerNewCustomer(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    const nameInput = document.getElementById('regCustomerName');
    const name = nameInput ? nameInput.value.trim() : '';

    if (!name) {
        alert('Please enter the customer name to save.');
        return;
    }
    if (!lastCapturedFaceBlob) {
        alert('No face image captured yet. Please click Scan Face first.');
        return;
    }

    const formData = new FormData();
    formData.append('customer_name', name);
    formData.append('file', lastCapturedFaceBlob, 'register_face.jpg');

    try {
        const res = await fetch(`${API_BASE}/vision/register-face`, {
            method: 'POST',
            headers: { 'X-API-Key': API_KEY },
            body: formData
        });
        const data = await parseApiResponse(res);

        alert(`Success: ${data.message}`);
        captureAndRecognizeFace(e);
    } catch (err) {
        alert('Registration failed: ' + err.message);
    }
}

// --- 3. Sentiment Analysis ---
async function analyzeSentiment(e) {
    if (e) { e.preventDefault(); e.stopPropagation(); }
    const textInput = document.getElementById('sentimentTextInput').value;
    const resultsArea = document.getElementById('sentimentResultsArea');

    if (!textInput || textInput.trim().length === 0) {
        alert('Please enter review text to analyze.');
        return;
    }

    resultsArea.innerHTML = '<div class="empty-state"><i class="fa-solid fa-spinner fa-spin"></i> Analyzing review sentiment...</div>';

    try {
        const res = await fetch(`${API_BASE}/analyze-sentiment`, {
            method: 'POST',
            headers: {
                'X-API-Key': API_KEY,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: textInput })
        });
        const data = await parseApiResponse(res);

        let sColor = '#10b981';
        if (data.sentiment === 'negative') sColor = '#ef4444';
        if (data.sentiment === 'neutral') sColor = '#f59e0b';

        resultsArea.innerHTML = `
            <div style="text-align:center; padding:1rem;">
                <span style="font-size:0.85rem; color:#94a3b8;">Predicted Sentiment</span>
                <h2 style="color:${sColor}; text-transform:uppercase; margin:0.3rem 0; font-size:1.6rem;">${data.sentiment}</h2>
                <div style="font-size:1.1rem; font-weight:700; color:#3b82f6;">Confidence: ${data.confidence_score}%</div>
            </div>
            <div style="margin-top:1rem; padding:0.75rem; background:rgba(15,23,42,0.5); border-radius:6px; font-size:0.85rem;">
                <span style="color:#94a3b8;">Preprocessed Text:</span>
                <p style="margin-top:0.25rem;"><em>"${data.cleaned_text}"</em></p>
            </div>
        `;
    } catch (err) {
        resultsArea.innerHTML = `<div class="empty-state" style="color:#ef4444;">${err.message}</div>`;
    }
}

// --- 4. Chatbot Interaction ---
function handleChatKeyPress(e) {
    if (e.key === 'Enter') {
        if (e.preventDefault) e.preventDefault();
        if (e.stopPropagation) e.stopPropagation();
        sendChatMessage(e);
        return false;
    }
}

async function sendChatMessage(e) {
    if (e) {
        if (e.preventDefault) e.preventDefault();
        if (e.stopPropagation) e.stopPropagation();
    }

    const inputEl = document.getElementById('chatInput');
    const msgContainer = document.getElementById('chatMessages');
    if (!inputEl || !msgContainer) return false;

    const question = inputEl.value.trim();
    if (!question) return false;

    // 1. Immediately render user question
    appendBubble(msgContainer, question, 'user');
    inputEl.value = '';

    // 2. Render temporary loading bot bubble
    const loadingBubble = document.createElement('div');
    loadingBubble.className = 'chat-bubble bot';
    loadingBubble.innerHTML = '<div class="message-content"><i class="fa-solid fa-spinner fa-spin"></i> Typing response...</div>';
    msgContainer.appendChild(loadingBubble);
    msgContainer.scrollTop = msgContainer.scrollHeight;

    try {
        const res = await fetch(`${API_BASE}/chatbot`, {
            method: 'POST',
            headers: {
                'X-API-Key': API_KEY,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });
        const data = await parseApiResponse(res);

        // Remove loading indicator and append actual AI response
        if (loadingBubble.parentNode) {
            loadingBubble.parentNode.removeChild(loadingBubble);
        }
        appendBubble(msgContainer, data.response, 'bot', data.match_type, data.intent);
    } catch (err) {
        if (loadingBubble.parentNode) {
            loadingBubble.parentNode.removeChild(loadingBubble);
        }
        appendBubble(msgContainer, `Sorry, ${err.message}`, 'bot');
    }

    return false;
}

function appendBubble(container, text, sender, matchType = '', intent = '') {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${sender}`;

    let metaInfo = matchType ? `<div style="font-size:0.7rem; opacity:0.7; margin-bottom:0.25rem;">[${matchType} | ${intent}]</div>` : '';

    bubble.innerHTML = `
        ${metaInfo}
        <div class="message-content">${text}</div>
    `;

    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}
