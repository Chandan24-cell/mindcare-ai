/**
 * MindCare AI application shell.
 *
 * Responsibilities in this file:
 * - API request orchestration for prediction and report endpoints
 * - Camera capture and upload workflows
 * - Form normalization for manual and sensor inputs
 * - Result rendering, chart updates, and toast notifications
 *
 * Dashboard chrome concerns such as greeting rotation, voice assistant, and
 * settings toggles live in     .
 */

// =============================================================================
// Configuration + State
// =============================================================================

const APP_CONFIG = Object.freeze({
    apiBase: '',
    apiTimeoutMs: 5000,
    maxPredictionHistory: 10,
    themeStorageKey: 'mindcare-theme',
    toastDurationMs: 2000,
    camera: {
        width: 640,
        height: 480,
        captureQuality: 0.9
    }
});

const appState = {
    cameraStream: null,
    stressTrendChart: null,
    isLiveMode: false,
    predictionHistory: []
};

function getById(id) {
    return document.getElementById(id);
}

function bindEvent(target, eventName, handler) {
    if (target) target.addEventListener(eventName, handler);
}

function applyStoredTheme() {
    const storedTheme = localStorage.getItem(APP_CONFIG.themeStorageKey);
    const isLightTheme = storedTheme === 'light';

    document.body.classList.toggle('light-mode', isLightTheme);
    document.body.classList.toggle('dark-mode', !isLightTheme);
}

applyStoredTheme();

// =============================================================================
// DOM References
// =============================================================================

const elements = {
    loading: getById('loading'),
    notification: getById('notification'),
    results: getById('results'),
    mlModeToggle: getById('mlModeToggle'),
    modeLabel: getById('modeLabel'),
    modeDescription: getById('modeDescription'),
    startCameraBtn: getById('startCamera'),
    stopCameraBtn: getById('stopCamera'),
    captureFrameBtn: getById('captureFrame'),
    cameraPreview: getById('cameraPreview'),
    cameraWrap: getById('cameraWrap'),
    cameraControls: getById('cameraControls'),
    cameraPlaceholder: getById('cameraPlaceholder'),
    imageUpload: getById('imageUpload'),
    imagePreview: getById('imagePreview'),
    uploadContent: document.querySelector('.upload-content'),
    previewImg: getById('previewImg'),
    removeImageBtn: getById('removeImage'),
    analyzeImageBtn: getById('analyzeImage'),
    fileName: getById('fileName'),
    manualForm: getById('manualForm'),
    sensorForm: getById('sensorForm'),
    emotionResult: getById('emotionResult'),
    stressResult: getById('stressResult'),
    confidenceResult: getById('confidenceResult'),
    reasonResult: getById('reasonResult'),
    suggestionsList: getById('suggestionsList'),
    stressProgress: getById('stressProgress'),
    chart: getById('chart'),
    exportResultsBtn: getById('exportResults'),
    reportEmail: getById('reportEmail')
};

function getThemeColors() {
    const styles = getComputedStyle(document.body);
    return {
        text: (styles.getPropertyValue('--text-primary') || '#f8fafc').trim(),
        muted: (styles.getPropertyValue('--text-secondary') || '#94a3b8').trim(),
        grid: (styles.getPropertyValue('--glass-border') || 'rgba(148,163,184,0.3)').trim(),
    };
}

// Mock response generator for fallback
function generateMockResponse(inputType, inputData = {}) {
    const emotions = ['happy', 'sad', 'neutral', 'angry', 'fear'];
    const stressLevels = ['low', 'medium', 'high'];
    
    let emotion = emotions[Math.floor(Math.random() * emotions.length)];
    let stressLevel = stressLevels[Math.floor(Math.random() * stressLevels.length)];
    let confidence = Math.round(Math.random() * 35 + 60) / 100;
    
    let reason = '';
    let disclaimer = 'Demo Mode - Using simulated responses';
    
    if (inputType === 'manual' && inputData.mood) {
        emotion = inputData.mood;
        stressLevel = inputData.stress_scale < 4 ? 'low' : inputData.stress_scale > 7 ? 'high' : 'medium';
        reason = `Demo: User reported mood as '${emotion}' with stress level ${inputData.stress_scale}/10`;
    } else if (inputType === 'sensor' && inputData.self_mood) {
        emotion = inputData.self_mood;
        let stressScore = 0;
        if (inputData.heart_rate > 90) stressScore += 1;
        if (inputData.hrv < 40) stressScore += 1;
        if (inputData.sleep_hours < 6) stressScore += 1;
        if (inputData.stress_scale > 7) stressScore += 1;
        stressLevel = stressScore >= 2 ? 'high' : stressScore === 1 ? 'medium' : 'low';
        reason = `Demo: Based on heart rate (${inputData.heart_rate} bpm), HRV (${inputData.hrv}), and self-reported stress (${inputData.stress_scale}/10)`;
    } else if (inputType === 'image') {
        reason = `Demo: Simulated image analysis detected '${emotion}' with ${(confidence * 100).toFixed(1)}% confidence`;
    }
    
    const suggestions = {
        happy: ['Maintain your positive routine', 'Practice gratitude journaling', 'Share your happiness with others'],
        sad: ['Practice deep breathing exercises', 'Consider reaching out to a friend', 'Take a short walk outside'],
        neutral: ['Monitor your mood throughout the day', 'Try light exercise', 'Journal your thoughts'],
        angry: ['Try calming breathing techniques', 'Reduce environmental stimuli', 'Consider a brief meditation'],
        fear: ['Practice grounding exercises', 'Use calming visualization', 'Focus on slow, deep breaths']
    };
    
    const baseSuggestions = suggestions[emotion] || ['Practice mindfulness', 'Take deep breaths'];
    if (stressLevel === 'high') {
        baseSuggestions.push('Consider talking to a professional if stress persists');
    }
    
    return {
        emotion: emotion,
        stress_level: stressLevel,
        confidence: confidence,
        reason: reason || `Demo: Random analysis - ${emotion} with ${stressLevel} stress`,
        suggestion: baseSuggestions,
        disclaimer: disclaimer,
        mode: 'mock'
    };
}

// =============================================================================
// Mode + State Management
// =============================================================================
function toggleMLMode() {
    appState.isLiveMode = elements.mlModeToggle.checked;
    if (appState.isLiveMode) {
        elements.modeLabel.textContent = 'Live AI Mode';
        elements.modeDescription.textContent = 'Using ViT model for real predictions';
        showNotification('Switched to Live AI Mode - Using trained models', 'success');
    } else {
        elements.modeLabel.textContent = 'Demo Mode';
        elements.modeDescription.textContent = 'Using simulated AI responses for demonstration';
        showNotification('Switched to Demo Mode - Using demo responses', 'info');
    }
}

// =============================================================================
// UI Utilities
// =============================================================================
function showLoading() {
    document.body.classList.add('is-loading');
    if (elements.loading) elements.loading.classList.remove('hidden');
}

function hideLoading() {
    document.body.classList.remove('is-loading');
    if (elements.loading) elements.loading.classList.add('hidden');
}

function showBackendRequiredError(action) {
    const messages = {
        'camera': 'Live AI Mode requires the backend server to be running. Please ensure the backend is deployed and accessible.',
        'upload': 'Live AI Mode requires the backend server to be running. Please ensure the backend is deployed and accessible.',
        'manual': 'Live AI Mode requires the backend server to be running. Please ensure the backend is deployed and accessible.',
        'sensor': 'Live AI Mode requires the backend server to be running. Please ensure the backend is deployed and accessible.'
    };
    
    const message = messages[action] || 'Backend server is not reachable. Please try again or switch to Demo Mode.';
    showNotification(message, 'error');
}

function showTimeoutError() {
    showNotification('Request timed out. Please try again.', 'error');
}

function toastIcon(type) {
    const icons = {
        success: '<path d="M20 6 9 17l-5-5"/>',
        error: '<circle cx="12" cy="12" r="9"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
        warning: '<path d="M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
        info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>'
    };
    return icons[type] || icons.info;
}

// =============================================================================
// Notifications
// =============================================================================

function showNotification(message, type = 'info', duration = APP_CONFIG.toastDurationMs) {
    const stack = getById('notification-stack');
    if (!stack) return;
    const displayMessage = stringifyErrorValue(message) || 'Something went wrong';
    const toastType = ['success', 'error', 'warning', 'info'].includes(type) ? type : 'info';

    const toast = document.createElement('div');
    toast.className = `notification-toast notification-${toastType}`;
    toast.setAttribute('role', toastType === 'error' ? 'alert' : 'status');
    toast.style.setProperty('--toast-duration', `${duration}ms`);

    toast.innerHTML = `
        <div class="notification-accent"></div>
        <div class="notification-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round">
                ${toastIcon(toastType)}
            </svg>
        </div>
        <div class="notification-copy">
            <div class="notification-title">${toastType}</div>
            <div class="notification-message"></div>
        </div>
        <button class="notification-close" aria-label="Close notification">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
        <div class="notification-progress" aria-hidden="true"></div>
    `;
    const messageElement = toast.querySelector('.notification-message');
    if (messageElement) messageElement.textContent = displayMessage;

    const close = () => {
        toast.classList.add('is-leaving');
        setTimeout(() => toast.remove(), 260);
    };
    toast.querySelector('.notification-close').addEventListener('click', close);

    let remaining = duration;
    let startedAt = Date.now();
    let timer = setTimeout(close, remaining);
    toast.addEventListener('mouseenter', () => {
        clearTimeout(timer);
        remaining -= Date.now() - startedAt;
        toast.classList.add('is-paused');
    });
    toast.addEventListener('mouseleave', () => {
        toast.classList.remove('is-paused');
        startedAt = Date.now();
        timer = setTimeout(close, Math.max(remaining, 800));
    });
    toast.addEventListener('click', (event) => {
        if (event.target.closest('.notification-close')) {
            clearTimeout(timer);
        }
    });

    stack.appendChild(toast);
}

function showToast(message, type = 'info') {
    showNotification(message, type);
}

function showError(message, rawError = message) {
    const readableMessage = getReadableErrorMessage(message, 'Something went wrong');
    showToast(readableMessage, 'error');
    console.error(rawError);
}

function showSuccess(message) {
    showNotification(message, 'success');
}

function disableButton(button) {
    if (button) {
        button.disabled = true;
        button.style.opacity = '0.6';
        button.style.cursor = 'not-allowed';
    }
}

function enableButton(button) {
    if (button) {
        button.disabled = false;
        button.style.opacity = '1';
        button.style.cursor = 'pointer';
    }
}

function isNetworkError(error) {
    const errorMessage = error.message ? error.message.toLowerCase() : '';
    return errorMessage.includes('fetch') || 
           errorMessage.includes('network') || 
           errorMessage.includes('failed to fetch') || 
           errorMessage.includes('load failed') || 
           errorMessage.includes('networkerror') || 
           errorMessage.includes('net::') || 
           errorMessage.includes('cors') ||
           errorMessage.includes('err_connection') ||
           errorMessage.includes('econnreset') ||
           errorMessage.includes('etimedout') ||
           errorMessage.includes('timeout') ||
           errorMessage.includes('socket') ||
           errorMessage.includes('typeerror') ||
           errorMessage.includes('abort') ||
           !navigator.onLine;
}

function isTimeoutError(error) {
    const errorMessage = error.message ? error.message.toLowerCase() : '';
    return errorMessage.includes('timeout') || 
           errorMessage.includes('etimedout') ||
           error.name === 'TimeoutError' ||
           error.name === 'AbortError';
}

// =============================================================================
// API Handling
// =============================================================================

function currentPredictionMode() {
    return appState.isLiveMode ? 'real' : 'mock';
}

function predictionEndpoint(kind) {
    return `${APP_CONFIG.apiBase}/predict/${kind}?mode=${currentPredictionMode()}`;
}

// Create a fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = APP_CONFIG.apiTimeoutMs) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        if (error.name === 'AbortError') {
            const timeoutError = new Error('Request timeout - Backend may be waking up');
            timeoutError.name = 'TimeoutError';
            throw timeoutError;
        }
        throw error;
    }
}

function validationFieldName(loc) {
    if (!Array.isArray(loc)) return 'request';
    const parts = loc
        .map(part => String(part))
        .filter(part => !['body', 'query', 'path'].includes(part));
    return parts.length ? parts.join('.') : 'request';
}

function formatValidationIssue(issue) {
    if (!issue || typeof issue !== 'object') {
        return stringifyErrorValue(issue);
    }

    const field = validationFieldName(issue.loc);
    const type = issue.type || '';
    const message = issue.msg || 'Invalid value';
    const ctx = issue.ctx || {};

    if (type === 'missing') return `Missing required field: ${field}`;
    if (type === 'extra_forbidden') return `Unexpected field: ${field}`;
    if (type.includes('enum')) {
        return `Invalid ${field} value${ctx.expected ? `: expected ${ctx.expected}` : ''}`;
    }
    if (type === 'greater_than_equal') return `Invalid ${field} value: must be at least ${ctx.ge}`;
    if (type === 'less_than_equal') return `Invalid ${field} value: must be at most ${ctx.le}`;
    if (type === 'int_parsing' || type === 'int_type') return `Invalid ${field} value: must be an integer`;
    if (type === 'float_parsing' || type === 'float_type') return `Invalid ${field} value: must be a number`;
    if (type === 'string_too_short') return `Invalid ${field} value: must be at least ${ctx.min_length} characters`;
    if (type === 'string_too_long') return `Invalid ${field} value: must be at most ${ctx.max_length} characters`;

    return field ? `Invalid ${field} value: ${message}` : message;
}

function formatValidationDetails(detail) {
    if (!Array.isArray(detail)) return '';
    return detail
        .map(formatValidationIssue)
        .filter(Boolean)
        .join('; ');
}

function getResponseData(error) {
    return error?.response?.data ?? error?.data ?? error?.response;
}

function stringifyErrorValue(value) {
    if (value === null || value === undefined) return '';

    if (typeof value === 'string') {
        const trimmed = value.trim();
        return trimmed && trimmed !== '[object Object]' ? trimmed : '';
    }

    if (Array.isArray(value)) {
        const validationMessage = formatValidationDetails(value);
        if (validationMessage) return validationMessage;

        const joined = value.map(stringifyErrorValue).filter(Boolean).join('; ');
        if (joined) return joined;
    }

    if (typeof value === 'object') {
        const nestedCandidates = [
            value.response?.data?.detail,
            value.response?.data?.error,
            value.response?.data?.message,
            value.response?.data,
            value.response?.detail,
            value.response?.error,
            value.response?.message,
            value.data?.detail,
            value.data?.error,
            value.data?.message,
            value.data,
            value.detail,
            value.error,
            value.message
        ];

        for (const nested of nestedCandidates) {
            if (nested === value) continue;
            const nestedMessage = stringifyErrorValue(nested);
            if (nestedMessage) return nestedMessage;
        }

        try {
            const serialized = JSON.stringify(value);
            if (serialized && serialized !== '{}') return serialized;
        } catch (e) {}

        const stringified = String(value);
        return stringified && stringified !== '[object Object]' ? stringified : '';
    }

    return String(value);
}

function readableBackendMessage(responseBody) {
    return stringifyErrorValue(responseBody?.data?.detail)
        || stringifyErrorValue(responseBody?.data?.error)
        || stringifyErrorValue(responseBody?.data?.message)
        || stringifyErrorValue(responseBody?.data)
        || stringifyErrorValue(responseBody?.error)
        || stringifyErrorValue(responseBody?.detail)
        || stringifyErrorValue(responseBody?.message)
        || stringifyErrorValue(responseBody?.validation_errors)
        || stringifyErrorValue(responseBody);
}

function getReadableErrorMessage(error, fallbackMessage = 'Something went wrong') {
    const responseData = getResponseData(error);
    const candidates = [
        error?.response?.data?.detail,
        error?.response?.data?.error,
        error?.response?.data?.message,
        error?.response?.data,
        error?.data?.detail,
        error?.data?.error,
        error?.data?.message,
        error?.data,
        error?.detail,
        error?.error,
        responseData?.detail,
        responseData?.error,
        responseData?.message,
        responseData,
        error?.message,
        error
    ];

    for (const candidate of candidates) {
        const readable = stringifyErrorValue(candidate);
        if (readable) return readable;
    }

    return fallbackMessage;
}

function showCaughtError(error, fallbackMessage = 'Something went wrong') {
    const detail = getReadableErrorMessage(error, '');
    const message = detail && detail !== fallbackMessage
        ? `${fallbackMessage}: ${detail}`
        : fallbackMessage;
    showError(message, error);
}

function buildApiError(responseBody, response, fallbackMessage = '') {
    const message = readableBackendMessage(responseBody)
        || fallbackMessage
        || `HTTP error! status: ${response?.status || 'unknown'}`;
    const apiError = new Error(message);
    apiError.detail = responseBody?.detail;
    apiError.error = responseBody?.error;
    apiError.data = responseBody;
    apiError.response = {
        data: responseBody,
        detail: responseBody?.detail,
        error: responseBody?.error,
        status: response?.status
    };
    apiError.status = response?.status;
    return apiError;
}

async function readJsonResponse(response) {
    try {
        return await response.json();
    } catch (error) {
        throw buildApiError(
            { detail: 'Invalid JSON response from server' },
            response,
            error?.message
        );
    }
}

async function requestJson(url, options, { fallbackMessage = '', isSuccess = data => data.success !== false } = {}) {
    const response = await fetchWithTimeout(url, options, APP_CONFIG.apiTimeoutMs);
    const data = await readJsonResponse(response);

    if (!response.ok || !isSuccess(data)) {
        throw buildApiError(data, response, fallbackMessage);
    }

    return data;
}

function postJson(url, payload, requestOptions = {}) {
    return requestJson(
        url,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        },
        requestOptions
    );
}

function postFormData(url, formData, requestOptions = {}) {
    return requestJson(
        url,
        {
            method: 'POST',
            body: formData
        },
        requestOptions
    );
}

// =============================================================================
// Camera Logic
// =============================================================================

// Camera Functions - No backend check needed
function isSecureForMedia() {
    return window.isSecureContext;
}

function isLocalhostHost() {
    return location.hostname === 'localhost';
}

async function startCamera() {
    if (!validateTerms()) return;
    
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError('Camera API not available. Use a modern browser with camera support.');
        return;
    }
    
    if (!isSecureForMedia()) {
        const hostHint = isLocalhostHost()
            ? 'Reload the page on http://localhost.'
            : 'Use http://localhost instead of 127.0.0.1, or serve over HTTPS.';
        showError(`Camera requires a secure context. ${hostHint}`);
        return;
    }
    
    try {
        showLoading();
        appState.cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: APP_CONFIG.camera.width,
                height: APP_CONFIG.camera.height
            }
        });

        if (elements.cameraPreview) {
            elements.cameraPreview.srcObject = appState.cameraStream;
            elements.cameraPreview.classList.remove('hidden');
        }
        if (elements.cameraWrap) elements.cameraWrap.classList.add('active');

        if (elements.startCameraBtn) elements.startCameraBtn.classList.add('hidden');
        if (elements.cameraControls) elements.cameraControls.classList.remove('hidden');
        if (elements.cameraPlaceholder) elements.cameraPlaceholder.classList.add('hidden');

        hideLoading();
        showSuccess('Camera started successfully');

    } catch (error) {
        hideLoading();
        showCaughtError(error, 'Failed to start camera');
    }
}

function stopCamera() {
    if (appState.cameraStream) {
        appState.cameraStream.getTracks().forEach(track => track.stop());
        appState.cameraStream = null;
    }

    if (elements.cameraWrap) elements.cameraWrap.classList.remove('active', 'is-capturing');
    if (elements.cameraPreview) {
        elements.cameraPreview.srcObject = null;
        elements.cameraPreview.classList.add('hidden');
    }
    if (elements.startCameraBtn) elements.startCameraBtn.classList.remove('hidden');
    if (elements.cameraControls) elements.cameraControls.classList.add('hidden');
    if (elements.cameraPlaceholder) elements.cameraPlaceholder.classList.remove('hidden');

    showSuccess('Camera stopped');
}

async function captureFrame() {
    if (!appState.cameraStream) {
        showError('Camera is not active');
        return;
    }

    try {
        disableButton(elements.captureFrameBtn);
        showLoading();
        if (elements.cameraWrap) elements.cameraWrap.classList.add('is-capturing');

        const canvas = document.createElement('canvas');
        canvas.width = APP_CONFIG.camera.width;
        canvas.height = APP_CONFIG.camera.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(
            elements.cameraPreview,
            0,
            0,
            APP_CONFIG.camera.width,
            APP_CONFIG.camera.height
        );

        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', APP_CONFIG.camera.captureQuality);
        });

        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        const data = await postFormData(predictionEndpoint('image'), formData);

        hideLoading();
        enableButton(elements.captureFrameBtn);
        if (elements.cameraWrap) elements.cameraWrap.classList.remove('is-capturing');
        displayResults(data);

    } catch (error) {
        if (elements.cameraWrap) elements.cameraWrap.classList.remove('is-capturing');
        handleApiError(error, 'camera', elements.captureFrameBtn);
    }
}

// =============================================================================
// Image Upload Logic
// =============================================================================

async function uploadImage(event) {
    if (!validateTerms()) {
        event.target.value = '';
        return;
    }
    const file = event.target.files[0];
    if (!file) return;

    if (elements.fileName) {
        elements.fileName.textContent = file.name;
    }

    try {
        const analyzeBtn = elements.analyzeImageBtn;
        disableButton(analyzeBtn);
        showLoading();
        
        const formData = new FormData();
        formData.append('file', file);

        const data = await postFormData(predictionEndpoint('image'), formData);

        hideLoading();
        if (analyzeBtn) enableButton(analyzeBtn);
        displayResults(data);
        showSuccess('Image analyzed successfully');

    } catch (error) {
        handleApiError(error, 'upload', elements.analyzeImageBtn);
    }
}

// =============================================================================
// Manual Form Logic
// =============================================================================

function ensureRequiredFields(data, fields) {
    for (const field of fields) {
        if (!data[field]) {
            throw new Error(`Please fill in ${field.replace('_', ' ')}`);
        }
    }
}

function normalizeManualPayload(form) {
    const data = Object.fromEntries(new FormData(form));

    if (!data.mood || !data.stress_scale) {
        throw new Error('Please fill in all required fields');
    }

    data.stress_scale = parseInt(data.stress_scale, 10);

    if (data.stress_scale < 1 || data.stress_scale > 10) {
        throw new Error('Stress scale must be between 1 and 10');
    }

    return data;
}

async function submitManual(event) {
    event.preventDefault();
    if (!validateTerms()) return;

    const submitBtn = event.target.querySelector('button[type="submit"]');

    try {
        disableButton(submitBtn);
        showLoading();

        const data = normalizeManualPayload(event.target);
        const result = await postJson(predictionEndpoint('manual'), data);

        hideLoading();
        enableButton(submitBtn);
        displayResults(result);
        showSuccess('Manual data submitted successfully');

    } catch (error) {
        handleApiError(error, 'manual', submitBtn);
    }
}

// =============================================================================
// Sensor Form Logic
// =============================================================================

function assertFiniteNumber(value, message) {
    if (!Number.isFinite(value)) throw new Error(message);
}

function assertInteger(value, message) {
    if (!Number.isInteger(value)) throw new Error(message);
}

function assertInRange(value, min, max, message) {
    if (value < min || value > max) throw new Error(message);
}

function normalizeSensorPayload(form) {
    const data = Object.fromEntries(new FormData(form));
    const required = ['heart_rate', 'hrv', 'sleep_hours', 'activity_level', 'self_mood', 'stress_scale'];
    ensureRequiredFields(data, required);

    const payload = {
        heart_rate: parseFloat(data.heart_rate),
        hrv: parseFloat(data.hrv),
        sleep_hours: parseFloat(data.sleep_hours),
        activity_level: parseInt(data.activity_level, 10),
        self_mood: data.self_mood,
        stress_scale: parseInt(data.stress_scale, 10)
    };

    assertFiniteNumber(payload.heart_rate, 'Heart rate must be a number');
    assertFiniteNumber(payload.hrv, 'HRV must be a number');
    assertFiniteNumber(payload.sleep_hours, 'Sleep hours must be a number');
    assertInteger(payload.activity_level, 'Activity level must be an integer');
    assertInteger(payload.stress_scale, 'Stress scale must be an integer');
    assertInRange(payload.heart_rate, 30, 220, 'Heart rate must be between 30 and 220');
    assertInRange(payload.hrv, 1, 300, 'HRV must be between 1 and 300');
    assertInRange(payload.sleep_hours, 0, 24, 'Sleep hours must be between 0 and 24');
    assertInRange(payload.activity_level, 1, 10, 'Activity level must be between 1 and 10');
    assertInRange(payload.stress_scale, 1, 10, 'Stress scale must be between 1 and 10');

    return payload;
}

async function submitSensor(event) {
    event.preventDefault();
    if (!validateTerms()) return;

    const submitBtn = event.target.querySelector('button[type="submit"]');
    let payload = null;

    try {
        disableButton(submitBtn);
        showLoading();

        payload = normalizeSensorPayload(event.target);
        console.debug('[MindCare] Sensor payload:', payload);

        const result = await postJson(predictionEndpoint('sensor'), payload);

        hideLoading();
        enableButton(submitBtn);
        displayResults(result);
        showSuccess('Sensor data submitted successfully');

    } catch (error) {
        handleApiError(error, 'sensor', submitBtn, { payload });
    }
}

// Centralized error handler for all API calls
function handleApiError(error, action, button, context = {}) {
    // Always hide loading and enable button first
    hideLoading();
    if (button) enableButton(button);

    // Extra logging for debugging
    try {
        console.debug('[MindCare] API error context:', {
            action,
            context,
            errorType: error?.name,
            errorMessage: error?.message,
            errorRaw: error
        });
    } catch (e) {}

    // Handle timeout specifically
    if (isTimeoutError(error)) {
        showTimeoutError();
        return;
    }

    // Handle network errors
    if (isNetworkError(error)) {
        // Try to surface backend-provided error message first
        // (validation errors often come back as {"detail": ...} or {"error": ...})
        const hasBackendDetails = Boolean(
            error?.detail || error?.error || error?.response ||
            context?.detail || context?.error || context?.response
        );
        const parsedMessage = hasBackendDetails
            ? parseReadableErrorMessage(error, action, context)
            : '';
        if (parsedMessage) {
            showError(parsedMessage, error);
            return;
        }

        // Demo mode: fallback to mock response
        if (!appState.isLiveMode) {
            console.debug('[MindCare] Backend unreachable, using mock response');
            const mockData = generateMockResponse(action === 'upload' ? 'image' : action);
            displayResults(mockData);
            showSuccess('Demo: Analyzed with simulated response (backend waking up)');
        } else {
            // Live AI mode: show backend required error
            showBackendRequiredError(action);
        }
        return;
    }

    // Other errors: extract a readable message (avoid showing [object Object])
    const parsedMessage = parseReadableErrorMessage(error, action, context);
    showError(parsedMessage, error);
}

function parseReadableErrorMessage(error, action, context = {}) {
    const response = context?.response || error?.response;
    const responseData = response?.data || response;
    const candidates = [
        error?.response?.data?.detail,
        error?.response?.data?.error,
        error?.response?.data?.message,
        error?.response?.data,
        error?.data?.detail,
        error?.data?.error,
        error?.data?.message,
        error?.data,
        error?.detail,
        error?.error,
        responseData?.detail,
        responseData?.error,
        responseData?.message,
        responseData,
        error?.message,
        context?.response?.data?.detail,
        context?.response?.data?.error,
        context?.response?.data?.message,
        context?.response?.data,
        context?.detail,
        context?.error
    ];

    for (const candidate of candidates) {
        const readable = stringifyErrorValue(candidate);
        if (readable) {
            return `Failed to process ${action}: ${readable}`;
        }
    }

    const serialized = stringifyErrorValue(error);
    if (serialized) {
        return `Failed to process ${action}: ${serialized}`;
    }

    return `Failed to process ${action}. Please try again.`;
}

// =============================================================================
// Report Handling
// =============================================================================

async function exportReport() {
    const emailInput = elements.reportEmail;
    if (!emailInput || !emailInput.value.trim()) {
        showError('Please enter your email address for the report.');
        emailInput?.focus();
        return;
    }

    const email = emailInput.value.trim();
    const emotion = elements.emotionResult?.textContent?.trim() || 'Unknown';
    const stressLevel = elements.stressResult?.textContent?.trim() || 'Unknown';
    const confidence = elements.confidenceResult?.textContent?.trim() || '0';

    // Parse confidence (remove % if present)
    const confidenceValue = parseFloat(confidence.replace('%', '')) / 100;

    // Get detailed analysis
    const reason = elements.reasonResult?.textContent?.trim() || '';

    // Get suggestions from list
    const suggestions = [];
    if (elements.suggestionsList) {
        elements.suggestionsList.querySelectorAll('li').forEach(li => {
            suggestions.push(li.textContent.trim());
        });
    }

    // Validate we have results to export
    if (emotion === '—' || emotion === '-') {
        showError('No analysis results to export. Please run an analysis first.');
        return;
    }

    try {
        showLoading();

        const result = await postJson(
            `${APP_CONFIG.apiBase}/generate-report`,
            {
                email: email,
                emotion: emotion,
                stress_level: stressLevel,
                confidence: confidenceValue,
                reason: reason,
                suggestions: suggestions
            },
            {
                fallbackMessage: 'Failed to generate report',
                isSuccess: data => data.success === true
            }
        );

        hideLoading();

        // Debug log
        console.log("Generate report response:", result);

        // Backend standardized response
        const reportData = result?.data || {};
        const reportPath = reportData.report_url;

        // CASE 1: Download available
        if (reportData.download_available === true && reportPath) {

            const reportUrl = reportPath.startsWith('http')
                ? reportPath
                : `${APP_CONFIG.apiBase}${reportPath.startsWith('/') ? '' : '/'}${reportPath}`;

            console.log("Opening report URL:", reportUrl);

            showSuccess('Report generated successfully!');
            window.open(reportUrl, '_blank');

        } else {

            // CASE 2: Email-only flow
            console.log("Report emailed successfully");

            showSuccess(
                'Your wellness report has been emailed successfully. Please check your inbox.'
            );
        }

    } catch (error) {
        hideLoading();
        showCaughtError(error, 'Failed to generate report');
    }
}

// =============================================================================
// UI Rendering
// =============================================================================

function displayResults(data) {
    if (elements.results) {
        elements.results.classList.remove('hidden');
        elements.results.classList.remove('results-ready');
        void elements.results.offsetWidth;
        elements.results.classList.add('results-ready');
        elements.results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    appState.predictionHistory.push({
        emotion: data.emotion,
        stress_level: data.stress_level,
        confidence: data.confidence,
        timestamp: new Date().toLocaleTimeString()
    });

    if (appState.predictionHistory.length > APP_CONFIG.maxPredictionHistory) {
        appState.predictionHistory.shift();
    }

    const emotion = data.emotion || '-';
    const stressLevel = data.stress_level || '-';
    const confidence = Number(data.confidence) || 0;
    if (elements.emotionResult) {
        elements.emotionResult.textContent = emotion;
        elements.emotionResult.closest('.result-card')?.setAttribute('data-emotion', emotion);
    }
    if (elements.stressResult) {
        elements.stressResult.textContent = stressLevel;
        elements.stressResult.closest('.result-card')?.setAttribute('data-stress', stressLevel);
    }
    if (elements.confidenceResult) elements.confidenceResult.textContent = data.confidence
        ? `${(data.confidence * 100).toFixed(1)}%`
        : '-';
    const stressProgress = elements.stressProgress;
    if (stressProgress) {
        const stressWidths = { low: 32, medium: 66, high: 100 };
        stressProgress.style.width = `${stressWidths[String(stressLevel).toLowerCase()] || 0}%`;
        stressProgress.dataset.level = String(stressLevel).toLowerCase();
    }
    if (elements.confidenceResult) {
        elements.confidenceResult.closest('.result-card')?.style.setProperty('--confidence-score', `${Math.round(confidence * 100)}%`);
    }
    if (elements.reasonResult) {
        elements.reasonResult.textContent = data.message || data.reason || 'No analysis available';
    }

    if (elements.suggestionsList) {
        elements.suggestionsList.innerHTML = '';
        if (data.suggestion && Array.isArray(data.suggestion)) {
            data.suggestion.forEach(suggestion => {
                const li = document.createElement('li');
                li.textContent = suggestion;
                elements.suggestionsList.appendChild(li);
            });
        }
    }

    updateChart(data);

    // Voice Assistant: Read out the recommendations if enabled
    if (typeof window.speakMessage === 'function' && data.suggestion && Array.isArray(data.suggestion) && data.suggestion.length > 0) {
        const emotion = data.emotion || '';
        const stress = data.stress_level || '';
        const intro = `Analysis complete. Detected emotion: ${emotion}. Stress level: ${stress}. Here are your recommendations:`;
        const recitation = data.suggestion.map((s, i) => `${i+1}. ${s}`).join('. ');
        const fullMessage = intro + ' ' + recitation;
        // Delay slightly to allow UI to settle
        setTimeout(() => {
            try {
                speakMessage(fullMessage);
            } catch (e) {
                console.debug('[MindCare] Voice assistant not available');
            }
        }, 500);
    }
}

// =============================================================================
// Charts
// =============================================================================

function updateChart(latestData) {
    if (!elements.chart) return;
    const ctx = elements.chart.getContext('2d');

    if (appState.stressTrendChart) {
        appState.stressTrendChart.destroy();
    }

    const stressLevelMap = { 'low': 1, 'medium': 2, 'high': 3 };
    const { text: textColor, grid: gridColor } = getThemeColors();
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = gridColor;

    const labels = appState.predictionHistory.map(p => p.timestamp);
    const stressData = appState.predictionHistory.map(p => stressLevelMap[p.stress_level] || 0);
    const confidenceData = appState.predictionHistory.map(p => (p.confidence || 0) * 3);
    const chartAreaGradient = ctx.createLinearGradient(0, 0, 0, 280);
    chartAreaGradient.addColorStop(0, 'rgba(0, 229, 255, 0.22)');
    chartAreaGradient.addColorStop(0.55, 'rgba(16, 217, 168, 0.08)');
    chartAreaGradient.addColorStop(1, 'rgba(0, 229, 255, 0)');
    const confidenceGradient = ctx.createLinearGradient(0, 0, 0, 280);
    confidenceGradient.addColorStop(0, 'rgba(168, 85, 247, 0.2)');
    confidenceGradient.addColorStop(1, 'rgba(168, 85, 247, 0)');

    let stressColor = 'rgba(75, 192, 192, 1)';
    if (latestData.stress_level === 'high') {
        stressColor = 'rgba(255, 79, 123, 1)';
    } else if (latestData.stress_level === 'medium') {
        stressColor = 'rgba(245, 158, 11, 1)';
    } else if (latestData.stress_level === 'low') {
        stressColor = 'rgba(16, 217, 168, 1)';
    }

    appState.stressTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Stress Level',
                    data: stressData,
                    backgroundColor: chartAreaGradient,
                    borderColor: stressColor,
                    borderWidth: 3,
                    tension: 0.48,
                    cubicInterpolationMode: 'monotone',
                    fill: true,
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    pointBackgroundColor: stressColor,
                    pointBorderColor: 'rgba(255, 255, 255, 0.85)',
                    pointBorderWidth: 2
                },
                {
                    label: 'Confidence (scaled)',
                    data: confidenceData,
                    backgroundColor: confidenceGradient,
                    borderColor: 'rgba(168, 85, 247, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.48,
                    cubicInterpolationMode: 'monotone',
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    pointBackgroundColor: 'rgba(168, 85, 247, 1)',
                    pointBorderColor: 'rgba(255, 255, 255, 0.85)',
                    pointBorderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 950,
                easing: 'easeInOutQuart'
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 3,
                    ticks: {
                        color: textColor,
                        callback: function (value) {
                            if (value === 1) return 'Low';
                            if (value === 2) return 'Medium';
                            if (value === 3) return 'High';
                            return '';
                        }
                    },
                    grid: {
                        color: gridColor,
                        drawBorder: false,
                        lineWidth: 1
                    }
                },
                x: {
                    display: true,
                    ticks: {
                        color: textColor
                    },
                    grid: {
                        color: gridColor,
                        drawBorder: false,
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Time',
                        color: textColor
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor,
                        usePointStyle: true,
                        boxWidth: 8,
                        padding: 18
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(5, 7, 15, 0.92)',
                    titleColor: '#f8fafc',
                    bodyColor: '#dbeafe',
                    borderColor: 'rgba(255,255,255,0.14)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            if (context.dataset.label === 'Stress Level') {
                                const labels = { 1: 'Low', 2: 'Medium', 3: 'High' };
                                return `Stress Level: ${labels[context.parsed.y] || 'Unknown'}`;
                            }
                            return `Confidence: ${((context.parsed.y / 3) * 100).toFixed(1)}%`;
                        }
                    }
                },
                title: {
                    display: true,
                    text: `Latest: ${latestData.emotion || 'Unknown'} - ${latestData.stress_level || 'N/A'}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    color: textColor
                }
            }
        }
    });
}

// Modal Functions
function openModal(id) {
    const modal = getById(id);
    if (modal) {
        modal.classList.remove('hidden');
    }
}

function closeModal(id) {
    const modal = getById(id);
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Terms Acceptance
// In Demo Mode, automatically allow access (no real data processing)
// In Real ML Mode, require explicit checkbox acceptance
function validateTerms() {
    const checkbox = getById('acceptTerms');
    
    // In Demo Mode, automatically accept (no actual data processing)
    if (!appState.isLiveMode) {
        return true;
    }
    
    // In Real ML Mode, require explicit acceptance
    const accepted = checkbox ? checkbox.checked : false;

    if (!accepted) {
        showError('Please accept Terms & Conditions to continue');
        openModal('termsModal');
        const footer = document.querySelector('.app-footer');
        if (footer) footer.scrollIntoView({ behavior: 'smooth' });
    }
    return accepted;
}

// Silent backend warm-up on page load
function warmUpBackend() {
    fetch(`${APP_CONFIG.apiBase}/docs`).catch(() => {});
}

function initializePredictionEventListeners() {
    bindEvent(elements.mlModeToggle, 'change', toggleMLMode);
    bindEvent(elements.startCameraBtn, 'click', startCamera);
    bindEvent(elements.stopCameraBtn, 'click', stopCamera);
    bindEvent(elements.captureFrameBtn, 'click', captureFrame);
    bindEvent(elements.manualForm, 'submit', submitManual);
    bindEvent(elements.sensorForm, 'submit', submitSensor);
    bindEvent(elements.exportResultsBtn, 'click', exportReport);
}

function onDomReady(callback) {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', callback, { once: true });
        return;
    }

    callback();
}

// =============================================================================
// Event Flow Initialization
// =============================================================================

onDomReady(() => {
    initializePredictionEventListeners();

    // Initialize Feather Icons
    const initFeatherIcons = () => {
        if (typeof feather !== 'undefined') {
            feather.replace();
        } else {
            setTimeout(initFeatherIcons, 100);
        }
    };
    initFeatherIcons();

    // Silent backend warm-up
    warmUpBackend();

    // Modal event listeners
    const openTerms = getById('openTerms');
    const openPrivacy = getById('openPrivacy');
    const agreeBtn = getById('agreeBtn');
    const openAbout = getById('openAbout');
    const navHelp = getById('navHelp');

    if (openTerms) openTerms.addEventListener('click', (e) => { e.preventDefault(); openModal('termsModal'); });
    if (openPrivacy) openPrivacy.addEventListener('click', (e) => { e.preventDefault(); openModal('privacyModal'); });
    if (openAbout) openAbout.addEventListener('click', (e) => { e.preventDefault(); openModal('aboutModal'); });

    if (agreeBtn) {
        agreeBtn.addEventListener('click', () => {
            const checkbox = getById('acceptTerms');
            if (checkbox) {
                checkbox.checked = true;
                closeModal('termsModal');
                showSuccess('Terms & Conditions accepted');
            }
        });
    }

    if (navHelp) navHelp.addEventListener('click', () => openModal('helpModal'));
    // Settings button opens the right sidebar panel in dashboard-ui.js.

    // Close modal on backdrop click
    document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
        backdrop.addEventListener('click', () => {
            const modal = backdrop.closest('.modal');
            if (modal) closeModal(modal.id);
        });
    });

    // Close buttons
    document.querySelectorAll('[data-modal-close]').forEach(btn => {
        btn.addEventListener('click', () => closeModal(btn.getAttribute('data-modal-close')));
    });

    document.querySelectorAll('.notification-close').forEach(btn => {
        btn.addEventListener('click', () => {
            const container = btn.closest('.notification-container');
            if (container) container.classList.add('hidden');
        });
    });

    // Tab navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            const tabPane = getById(`tab-${btn.dataset.tab}`);
            if (tabPane) tabPane.classList.add('active');
        });
    });

    // Image preview
    const imageUpload = elements.imageUpload;
    const previewImg = elements.previewImg;
    const imagePreview = elements.imagePreview;
    const uploadContent = elements.uploadContent;
    const removeImage = elements.removeImageBtn;

    if (imageUpload && previewImg && imagePreview && uploadContent) {
        imageUpload.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function (e) {
                    previewImg.src = e.target.result;
                    uploadContent.classList.add('hidden');
                    imagePreview.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            }
        });

        if (removeImage) {
            removeImage.addEventListener('click', () => {
                imageUpload.value = '';
                previewImg.src = '';
                imagePreview.classList.add('hidden');
                uploadContent.classList.remove('hidden');
            });
        }
    }

    // Slider value displays
    const stressSlider = getById('stress_scale');
    const stressValue = getById('stressValue');
    const stressValueDisplay = getById('stressValueDisplay');
    if (stressSlider && stressValue) {
        const updateStress = () => {
            stressValue.textContent = stressSlider.value;
            if (stressValueDisplay) stressValueDisplay.textContent = stressSlider.value;
        };
        stressSlider.addEventListener('input', updateStress);
        updateStress();
    }

    const sensorStressSlider = getById('sensor_stress_scale');
    const sensorStressValue = getById('sensorStressValue');
    const sensorStressValueDisplay = getById('sensorStressValueDisplay');
    if (sensorStressSlider && sensorStressValue) {
        const updateSensorStress = () => {
            sensorStressValue.textContent = sensorStressSlider.value;
            if (sensorStressValueDisplay) sensorStressValueDisplay.textContent = sensorStressSlider.value;
        };
        sensorStressSlider.addEventListener('input', updateSensorStress);
        updateSensorStress();
    }

    const activitySlider = getById('activity_level');
    const activityValue = getById('activityValue');
    const activityValueDisplay = getById('activityValueDisplay');
    if (activitySlider && activityValue) {
        const updateActivity = () => {
            activityValue.textContent = activitySlider.value;
            if (activityValueDisplay) activityValueDisplay.textContent = activitySlider.value;
        };
        activitySlider.addEventListener('input', updateActivity);
        updateActivity();
    }

    // Analyze Image button
    const analyzeImageBtn = elements.analyzeImageBtn;
    if (analyzeImageBtn && imageUpload) {
        analyzeImageBtn.addEventListener('click', () => {
            if (imageUpload.files && imageUpload.files[0]) {
                uploadImage({ target: imageUpload, preventDefault: () => {} });
            } else {
                showError('Please select an image first');
            }
        });
    }

    toggleMLMode();

    console.info('MindCare app loaded');
    console.info('API Base URL:', APP_CONFIG.apiBase);
});

// Silence non-critical UI errors in the console
window.addEventListener('error', (e) => {
    console.warn('Non-critical UI warning:', e.message);
});
