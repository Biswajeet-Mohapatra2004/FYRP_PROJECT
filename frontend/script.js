/* ============================================================
   FYRP — Dashboard Script
   ============================================================ */

// ── Particle background ──────────────────────────────────────
(function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    const ctx = canvas.getContext('2d');
    let W, H, particles = [];

    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }

    function Particle() {
        this.x  = Math.random() * W;
        this.y  = Math.random() * H;
        this.vx = (Math.random() - 0.5) * 0.4;
        this.vy = (Math.random() - 0.5) * 0.4;
        this.r  = Math.random() * 1.5 + 0.3;
        this.a  = Math.random() * 0.5 + 0.1;
        const hues = [185, 270, 300, 220];
        this.hue = hues[Math.floor(Math.random() * hues.length)];
    }

    function init() {
        resize();
        particles = Array.from({length: 120}, () => new Particle());
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = W;
            if (p.x > W) p.x = 0;
            if (p.y < 0) p.y = H;
            if (p.y > H) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${p.a})`;
            ctx.fill();
        });

        // Draw faint connection lines between close particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const d  = Math.sqrt(dx*dx + dy*dy);
                if (d < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0,245,255,${0.06 * (1 - d/100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    init();
    draw();
})();


// ── Loader step cycling ──────────────────────────────────────
let stepTimer = null;
const STEPS = ['step1','step2','step3','step4'];
const STEP_DELAY = 2600;

function startLoaderSteps() {
    let i = 0;
    STEPS.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });
    document.getElementById(STEPS[0]).classList.add('active');
    stepTimer = setInterval(() => {
        if (i < STEPS.length - 1) {
            document.getElementById(STEPS[i]).classList.replace('active', 'done');
            i++;
            document.getElementById(STEPS[i]).classList.add('active');
        }
    }, STEP_DELAY);
}

function stopLoaderSteps() {
    clearInterval(stepTimer);
    STEPS.forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    });
}


// ── Drop zone ────────────────────────────────────────────────
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const imagePreview= document.getElementById('imagePreview');
const analyzeBtn  = document.getElementById('analyzeBtn');
let selectedFile  = null;

['dragenter','dragover','dragleave','drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
);

['dragenter','dragover'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.add('dragover'))
);
['dragleave','drop'].forEach(ev =>
    dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'))
);

dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', function() { handleFiles(this.files); });

function handleFiles(files) {
    if (!files.length) return;
    const f = files[0];
    if (!f.type.startsWith('image/')) { alert('Please select an image file.'); return; }
    selectedFile = f;
    const reader = new FileReader();
    reader.onload = e => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(f);
}


// ── Analyze ──────────────────────────────────────────────────
const loader         = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    resultsSection.classList.add('hidden');
    loader.classList.remove('hidden');
    startLoaderSteps();

    // Reset fade-up animations for re-runs
    document.querySelectorAll('.fade-up').forEach(el => {
        el.style.animation = 'none';
        el.offsetHeight;
        el.style.animation = '';
    });

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://localhost:8001/analyze', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        stopLoaderSteps();
        await new Promise(r => setTimeout(r, 400));

        populateDashboard(data);
        loader.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        console.error(err);
        stopLoaderSteps();
        loader.classList.add('hidden');
        alert(`Analysis failed: ${err.message}\n\nMake sure the backend is running on port 8001.`);
    } finally {
        analyzeBtn.disabled = false;
    }
});


// ── Populate dashboard ───────────────────────────────────────
function populateDashboard(data) {
    const decision = (data.final_decision || 'UNKNOWN').toUpperCase();
    const risk     = (data.risk_level     || 'UNKNOWN').toUpperCase();
    const conf     = (data.confidence     || 'unknown').toLowerCase();

    const decisionColor = { LEGITIMATE: 'var(--green)', SUSPICIOUS: 'var(--yellow)', ADVERSARIAL: 'var(--red)' };
    const riskColor     = { LOW: 'var(--green)',  MEDIUM: 'var(--yellow)', HIGH: 'var(--red)' };
    const confColor     = { high: 'var(--green)', medium: 'var(--yellow)', low: 'var(--red)' };
    const iconMap       = { LEGITIMATE: '✅', SUSPICIOUS: '⚠️', ADVERSARIAL: '🚨' };

    // ── Verdict banner ──
    const banner = document.getElementById('verdictBanner');
    banner.className = 'verdict-banner fade-up';
    banner.classList.add(`banner-${decision.toLowerCase()}`);

    document.getElementById('vbIcon').textContent = iconMap[decision] || '❓';

    const vbDec = document.getElementById('vbDecision');
    vbDec.textContent = decision;
    vbDec.style.color = decisionColor[decision] || 'var(--cyan)';

    const vbRisk = document.getElementById('vbRisk');
    vbRisk.textContent = `RISK: ${risk}`;
    vbRisk.style.color = riskColor[risk] || 'var(--cyan)';

    const vbConf = document.getElementById('vbConf');
    vbConf.textContent = `CONF: ${conf.toUpperCase()}`;
    vbConf.style.color = confColor[conf] || 'var(--cyan)';

    document.getElementById('vbCnn').textContent = data.cnn_prediction || 'N/A';
    document.getElementById('vbOverride').textContent = data.override ? 'YES' : 'NO';

    // Timestamp meta
    document.getElementById('resultsMeta').textContent =
        `ANALYZED · ${new Date().toLocaleTimeString('en-US', {hour12: false})}`;

    // ── XAI image ──
    const img = document.getElementById('heatmapImage');
    if (data.dashboard_url || data.heatmap_url) {
        img.src = (data.dashboard_url || data.heatmap_url) + '?t=' + Date.now();
        img.classList.remove('hidden');
    } else {
        img.classList.add('hidden');
    }

    // ── Reasoning ──
    document.getElementById('judgeReasoningText').textContent =
        data.reasoning || 'No reasoning provided.';

    // ── Threshold ──
    const td = data.threshold_details || {};
    const T  = td.computed_threshold !== undefined ? td.computed_threshold : 0;

    const tValEl = document.getElementById('tValue');
    tValEl.textContent = T.toFixed(4);
    tValEl.style.color      = T < 0.2 ? 'var(--green)' : T > 0.65 ? 'var(--red)' : 'var(--yellow)';
    tValEl.style.textShadow = T < 0.2 ? '0 0 20px var(--green)' : T > 0.65 ? '0 0 20px var(--red)' : '0 0 20px var(--yellow)';

    // Needle position: track maps 0→100% to left 0→100%
    // Zone boundaries: safe 0–0.2, amb 0.2–0.65, threat 0.65–1.0
    setTimeout(() => {
        document.getElementById('tNeedle').style.left = `${Math.min(T * 100, 99)}%`;
    }, 300);

    // Factor bars
    const conf_s   = td.confidence_score    || 0;
    const anom_s   = td.anomaly_score       || 0;
    const drift_s  = td.embedding_drift     || 0;
    const qual_s   = td.image_quality_score || 0;

    setFactor('cnnConf',     'bar-conf',  conf_s,  '%');
    setFactor('anomalyScore','bar-anom',  anom_s,  '');
    setFactor('driftScore',  'bar-drift', drift_s, '');
    setFactor('qualityScore','bar-qual',  qual_s,  '');

    // Color the anomaly box red if high, green if low
    colorFactor('fb-anom', anom_s, 0.4, 0.7);
    colorFactor('fb-drift', drift_s, 0.3, 0.6);

    // ── Advocates ──
    const pro = data.advocate_pro || {};
    const opp = data.advocate_opp || {};

    document.getElementById('proStance').textContent = pro.stance   || '—';
    document.getElementById('proArg').textContent    = pro.argument || 'Debate skipped (fast-path).';
    document.getElementById('oppStance').textContent = opp.stance   || '—';
    document.getElementById('oppArg').textContent    = opp.argument || 'Debate skipped (fast-path).';
}

function setFactor(valId, barId, value, suffix) {
    document.getElementById(valId).textContent = value.toFixed(4) + suffix;
    setTimeout(() => {
        document.getElementById(barId).style.width = `${Math.min(value * 100, 100)}%`;
    }, 500);
}

function colorFactor(boxId, value, warnThreshold, dangerThreshold) {
    const box = document.getElementById(boxId);
    if (!box) return;
    if (value >= dangerThreshold) {
        box.style.borderColor = 'rgba(255,51,102,0.4)';
        box.querySelector('.factor-bar').style.background = 'linear-gradient(90deg, var(--yellow), var(--red))';
    } else if (value >= warnThreshold) {
        box.style.borderColor = 'rgba(255,215,0,0.3)';
        box.querySelector('.factor-bar').style.background = 'linear-gradient(90deg, var(--cyan), var(--yellow))';
    }
}
