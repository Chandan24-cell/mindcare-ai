/**
 * Compatibility loader for the relocated MindCare frontend application.
 *
 * The production source now lives in /frontend/js/app.js. This file preserves
 * the historical /frontend/script.js URL for cached pages and direct links.
 */
(function loadMindCareApp() {
    const source = '/frontend/js/app.js';

    if (document.currentScript && document.readyState === 'loading') {
        document.write(`<script src="${source}"><\/script>`);
        return;
    }

    const script = document.createElement('script');
    script.src = source;
    script.defer = true;
    document.head.appendChild(script);
}());
