// TruthVision JavaScript Suite - Black & White Edition

// 1. Interactive Grad-CAM Opacity Slider
function initGradCamSlider() {
    const slider = document.getElementById('gradCamSlider');
    const overlayImg = document.getElementById('gradCamOverlay');
    const opacityLabel = document.getElementById('opacityVal');

    if (slider && overlayImg) {
        slider.addEventListener('input', function() {
            const val = parseFloat(this.value);
            overlayImg.style.opacity = val;
            if (opacityLabel) opacityLabel.innerText = Math.round(val * 100) + '%';
        });
    }
}

// 2. Video Timeline Chart with Chart.js (Monochrome)
function renderVideoTimeline(timestamps, probabilities) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    const dataPoints = timestamps.map((t, idx) => ({
        x: t,
        y: probabilities[idx]
    }));

    new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Deepfake Probability (%)',
                data: dataPoints,
                borderColor: '#ffffff',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.25,
                pointRadius: 4,
                pointHoverRadius: 7,
                pointBackgroundColor: (context) => {
                    const val = context.raw?.y;
                    return val >= 50 ? '#ffffff' : '#737373';
                },
                pointBorderColor: '#000000',
                pointBorderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Time (seconds)', color: '#a3a3a3' },
                    grid: { color: '#262626' },
                    ticks: { color: '#a3a3a3' }
                },
                y: {
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Risk Index (%)', color: '#a3a3a3' },
                    grid: { color: '#262626' },
                    ticks: { color: '#a3a3a3' }
                }
            },
            plugins: {
                legend: { labels: { color: '#ffffff', font: { weight: '600' } } },
                tooltip: {
                    backgroundColor: '#171717',
                    titleColor: '#ffffff',
                    bodyColor: '#e5e5e5',
                    borderColor: '#404040',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `Risk: ${context.parsed.y}% at ${context.parsed.x}s`;
                        }
                    }
                }
            }
        }
    });
}

// 3. Live Webcam Face Authenticity Scanner (Monochrome)
let webcamStream = null;
let isScanning = false;

function initWebcamScanner() {
    const video = document.getElementById('webcamFeed');
    const canvas = document.getElementById('hudCanvas');
    const startBtn = document.getElementById('startWebcamBtn');
    const stopBtn = document.getElementById('stopWebcamBtn');
    const statusBadge = document.getElementById('liveStatusBadge');
    const probBadge = document.getElementById('liveProbBadge');

    if (!video || !startBtn) return;

    startBtn.addEventListener('click', async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            video.srcObject = webcamStream;
            isScanning = true;
            startBtn.classList.add('d-none');
            stopBtn.classList.remove('d-none');
            scanLoop();
        } catch (err) {
            alert('Camera access error: ' + err.message);
        }
    });

    stopBtn.addEventListener('click', () => {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
        }
        isScanning = false;
        startBtn.classList.remove('d-none');
        stopBtn.classList.add('d-none');
    });

    async function scanLoop() {
        if (!isScanning) return;

        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/api/detect-frame/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: dataUrl })
            });
            const data = await response.json();

            if (data.status === 'success') {
                if (data.is_fake) {
                    statusBadge.className = 'badge-fake';
                    statusBadge.innerText = `DEEPFAKE (${data.confidence}%)`;
                } else {
                    statusBadge.className = 'badge-real';
                    statusBadge.innerText = `AUTHENTIC (${data.confidence}%)`;
                }

                if (probBadge) {
                    probBadge.innerText = `Index: ${data.deepfake_probability}%`;
                }

                // Draw high-contrast bounding box on canvas
                if (data.bbox) {
                    const [x, y, w, h] = data.bbox;
                    ctx.strokeStyle = data.is_fake ? '#ffffff' : '#a3a3a3';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(x, Math.max(0, y - 24), 160, 22);
                    ctx.font = 'bold 13px Inter, sans-serif';
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(`${data.classification} ${data.confidence}%`, x + 4, Math.max(16, y - 8));
                }
            }
        } catch (e) {
            console.error('Frame error:', e);
        }

        if (isScanning) {
            setTimeout(scanLoop, 500);
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initGradCamSlider();
    initWebcamScanner();
});
