const themeBtn = document.getElementById('themeBtn');
const htmlEl = document.documentElement;

themeBtn.addEventListener('click', () => {
    const currentTheme = htmlEl.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    htmlEl.setAttribute('data-theme', newTheme);
    themeBtn.textContent = newTheme === 'dark' ? '☀️ Light Mode' : '🌙 Dark Mode';
});

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');

let selectedFile = null;

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        selectedFile = files[0];
        
        if (!selectedFile.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.readAsDataURL(selectedFile);
        reader.onload = () => {
            imagePreview.src = reader.result;
            imagePreview.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
    }
}

// Analysis Logic
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // UI State
    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    loader.classList.remove('hidden');
    
    // Reset animations by cloning and replacing nodes
    document.querySelectorAll('.fade-in').forEach(el => {
        el.style.animation = 'none';
        el.offsetHeight; // trigger reflow
        el.style.animation = null; 
    });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://localhost:8001/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.error) throw new Error(data.error);

        populateDashboard(data);
        
        loader.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Scroll to results smoothly
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (error) {
        console.error(error);
        alert('Failed to analyze image. Check console for details. Ensure backend is running.');
        loader.classList.add('hidden');
    } finally {
        analyzeBtn.disabled = false;
    }
});

function populateDashboard(data) {
    // Colors and Icons
    const colorMap = {
        'LEGITIMATE': 'var(--success)',
        'SUSPICIOUS': 'var(--warning)',
        'ADVERSARIAL': 'var(--danger)',
        'LOW': 'var(--success)',
        'MEDIUM': 'var(--warning)',
        'HIGH': 'var(--danger)',
        'high': 'var(--success)',
        'medium': 'var(--warning)',
        'low': 'var(--danger)'
    };

    const iconMap = {
        'LEGITIMATE': '✅',
        'SUSPICIOUS': '⚠️',
        'ADVERSARIAL': '🚨'
    };

    const dDecision = (data.final_decision || "UNKNOWN").toUpperCase();
    const dRisk = (data.risk_level || "UNKNOWN").toUpperCase();
    const dConf = (data.confidence || "UNKNOWN").toLowerCase();

    // Summary Card
    const verdictText = document.getElementById('finalVerdictText');
    verdictText.textContent = dDecision;
    verdictText.style.color = colorMap[dDecision] || 'var(--primary)';
    document.getElementById('finalVerdictIcon').textContent = iconMap[dDecision] || '❓';

    const riskBadge = document.getElementById('riskLevelBadge');
    riskBadge.textContent = `Risk: ${dRisk}`;
    riskBadge.style.color = colorMap[dRisk] || 'var(--primary)';
    riskBadge.style.borderColor = colorMap[dRisk] || 'var(--primary)';

    const confBadge = document.getElementById('confidenceBadge');
    confBadge.textContent = `Conf: ${dConf.toUpperCase()}`;
    confBadge.style.color = colorMap[dConf] || 'var(--primary)';
    confBadge.style.borderColor = colorMap[dConf] || 'var(--primary)';

    document.getElementById('cnnPredText').textContent = data.cnn_prediction || "N/A";
    document.getElementById('overrideText').textContent = data.override ? 'YES' : 'NO';

    // Dashboard Image
    const heatmapImg = document.getElementById('heatmapImage');
    if (data.dashboard_url) {
        heatmapImg.src = data.dashboard_url + '?t=' + new Date().getTime(); // cache bust
        heatmapImg.classList.remove('hidden');
    } else if (data.heatmap_url) {
        // Fallback
        heatmapImg.src = data.heatmap_url + '?t=' + new Date().getTime();
        heatmapImg.classList.remove('hidden');
    } else {
        heatmapImg.classList.add('hidden');
    }

    // Reasoning
    document.getElementById('judgeReasoningText').textContent = data.reasoning || "No reasoning provided.";

    // Thresholds
    const tDetails = data.threshold_details || {};
    const tValue = tDetails.computed_threshold !== undefined ? tDetails.computed_threshold : (tDetails.threshold || 0);
    
    document.getElementById('tValue').textContent = tValue.toFixed(4);
    
    const percentage = Math.min(Math.max(tValue * 100, 0), 100);
    const tBar = document.getElementById('tBar');
    tBar.style.width = '0%'; // Reset for animation
    setTimeout(() => {
        tBar.style.width = `${percentage}%`;
    }, 500);

    // Show raw scores (not inverted factors) so values are human-readable
    // e.g. CNN Confidence: 0.9983 means 99.83% confident, NOT 0% confident
    document.getElementById('cnnConf').textContent = (tDetails.confidence_score || 0).toFixed(4);
    document.getElementById('anomalyScore').textContent = (tDetails.anomaly_score || 0).toFixed(4);
    document.getElementById('driftScore').textContent = (tDetails.embedding_drift || 0).toFixed(4);
    document.getElementById('qualityScore').textContent = (tDetails.image_quality_score || 0).toFixed(4);

    // Advocates
    const advPro = data.advocate_pro || {};
    const advOpp = data.advocate_opp || {};

    document.getElementById('proStance').textContent = advPro.stance || "N/A";
    document.getElementById('proArg').textContent = advPro.argument || "No argument.";
    
    document.getElementById('oppStance').textContent = advOpp.stance || "N/A";
    document.getElementById('oppArg').textContent = advOpp.argument || "No argument.";
}
