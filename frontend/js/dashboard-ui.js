/**
 * MindCare dashboard chrome.
 *
 * Handles greeting rotation, settings persistence, native browser
 * notifications, and the optional voice assistant. Prediction flows stay in
 * app.js so UI chrome can evolve independently from API orchestration.
 */
(function initializeDashboardChrome() {
    const STORAGE_KEYS = Object.freeze({
        theme: 'mindcare-theme',
        voice: 'mindcare-voice',
        notifications: 'mindcare-notify'
    });

    const TIMING = Object.freeze({
        greetingRotationMs: 5000,
        greetingFadeMs: 500,
        voiceTypingMs: 150,
        assistantHideDelayMs: 1000,
        initialGreetingDelayMs: 1200
    });

    const browserNotificationsSupported = typeof Notification !== 'undefined';
    let voiceEnabled = localStorage.getItem(STORAGE_KEYS.voice) !== 'off';
    let browserNotificationsEnabled = localStorage.getItem(STORAGE_KEYS.notifications) !== 'off';

    function getById(id) {
        return document.getElementById(id);
    }

    function bindEvent(target, eventName, handler) {
        if (target) target.addEventListener(eventName, handler);
    }

    function getTimeBasedMessages() {
        const hour = new Date().getHours();

        if (hour >= 5 && hour < 12) {
            return [
                "🌅 Good Morning | 🧠 Start Your Day with Mental Clarity",
                "🚀 MindCare AI is Ready to Analyze Your Emotional Patterns",
                "🌿 A Calm Mind Creates a Productive Day"
            ];
        }

        if (hour >= 12 && hour < 17) {
            return [
                "☀️ Good Afternoon | 📊 Stay Focused & Balanced",
                "🤖 Real-Time Emotional Insights Active",
                "💡 Small Mental Breaks Improve Big Results"
            ];
        }

        if (hour >= 17 && hour < 21) {
            return [
                "🌇 Good Evening | 🌿 Time to Reflect",
                "🧘 Relax & Let MindCare Guide Your Wellness",
                "📈 Reviewing Today's Emotional Patterns"
            ];
        }

        return [
            "🌙 Good Night | 💙 Prioritize Rest",
            "🧠 Mental Recovery is Essential for Growth",
            "✨ MindCare Supporting Your Inner Peace"
        ];
    }

    function getGreetingMessage() {
        const hour = new Date().getHours();

        if (hour >= 5 && hour < 12) {
            return "Good morning. Welcome to MindCare. Your mental wellness dashboard is now ready.";
        }

        if (hour >= 12 && hour < 17) {
            return "Good afternoon. Welcome to MindCare. Your emotional insights are prepared.";
        }

        if (hour >= 17 && hour < 21) {
            return "Good evening. Relax. Your personalized analysis is ready.";
        }

        return "Good night. Your journey toward mental clarity begins now.";
    }

    function applyTheme(theme, toggleDarkMode) {
        const isDarkTheme = theme !== 'light';

        document.body.classList.toggle('light-mode', !isDarkTheme);
        document.body.classList.toggle('dark-mode', isDarkTheme);
        document.documentElement.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
        document.body.style.colorScheme = isDarkTheme ? 'dark' : 'light';
        localStorage.setItem(STORAGE_KEYS.theme, isDarkTheme ? 'dark' : 'light');

        if (toggleDarkMode) toggleDarkMode.checked = isDarkTheme;
    }

    function typeText(text, callback) {
        const aiMessage = getById('aiMessage');
        if (!aiMessage) {
            callback();
            return;
        }

        const words = text.split(' ');
        let wordIndex = 0;

        aiMessage.textContent = '';
        const typingInterval = setInterval(() => {
            aiMessage.textContent += `${words[wordIndex]} `;
            wordIndex += 1;

            if (wordIndex >= words.length) {
                clearInterval(typingInterval);
                callback();
            }
        }, TIMING.voiceTypingMs);
    }

    function speakMessage(text) {
        if (!voiceEnabled || !text) return;

        const aiBox = getById('aiAssistantBox');
        const voiceAnimation = getById('voiceAnimation');
        if (!aiBox) return;

        aiBox.classList.remove('hide');
        typeText(text, () => {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1;
            utterance.pitch = 1;

            if (voiceAnimation) voiceAnimation.style.display = 'flex';

            speechSynthesis.cancel();
            speechSynthesis.speak(utterance);

            utterance.onend = () => {
                if (voiceAnimation) voiceAnimation.style.display = 'none';
                setTimeout(() => aiBox.classList.add('hide'), TIMING.assistantHideDelayMs);
            };
        });
    }

    function showNativeNotification(message) {
        if (!browserNotificationsEnabled || !browserNotificationsSupported) return;

        if (Notification.permission === 'granted') {
            new Notification('MindCare AI', { body: message });
        }
    }

    function initializeGreetingRotation() {
        const greetingElement = getById('dynamicGreeting');
        if (!greetingElement) return;

        const messages = getTimeBasedMessages();
        let messageIndex = 0;

        function rotateGreeting() {
            greetingElement.classList.add('fade-out');

            setTimeout(() => {
                greetingElement.innerText = messages[messageIndex];
                greetingElement.classList.remove('fade-out');
                greetingElement.classList.add('fade-in');
                messageIndex = (messageIndex + 1) % messages.length;
            }, TIMING.greetingFadeMs);
        }

        greetingElement.innerText = messages[0];
        setInterval(rotateGreeting, TIMING.greetingRotationMs);
    }

    function initializeSettingsPanel() {
        const settingsBtn = getById('settingsBtn');
        const settingsPanel = getById('settingsPanel');
        const closeSettings = getById('closeSettings');

        bindEvent(settingsBtn, 'click', () => settingsPanel?.classList.add('active'));
        bindEvent(closeSettings, 'click', () => settingsPanel?.classList.remove('active'));
    }

    function initializeSettingsToggles() {
        const toggleVoice = getById('toggleVoice');
        const toggleNotifications = getById('toggleNotifications');
        const toggleDarkMode = getById('toggleDarkMode');

        applyTheme(localStorage.getItem(STORAGE_KEYS.theme) || 'dark', toggleDarkMode);

        if (toggleVoice) {
            toggleVoice.checked = voiceEnabled;
            bindEvent(toggleVoice, 'change', function handleVoiceToggle() {
                voiceEnabled = this.checked;
                localStorage.setItem(STORAGE_KEYS.voice, voiceEnabled ? 'on' : 'off');

                if (!voiceEnabled) {
                    speechSynthesis.cancel();
                    console.debug('Voice Assistant Disabled');
                } else {
                    console.debug('Voice Assistant Enabled');
                }
            });
        }

        if (toggleNotifications) {
            toggleNotifications.checked = browserNotificationsEnabled && browserNotificationsSupported;
            bindEvent(toggleNotifications, 'change', function handleNotificationToggle() {
                browserNotificationsEnabled = browserNotificationsSupported && this.checked;
                localStorage.setItem(
                    STORAGE_KEYS.notifications,
                    browserNotificationsEnabled ? 'on' : 'off'
                );

                if (browserNotificationsEnabled && browserNotificationsSupported) {
                    if (Notification.permission !== 'granted' && Notification.permission !== 'denied') {
                        Notification.requestPermission();
                    }
                    console.debug('Notifications Enabled');
                } else {
                    console.debug('Notifications Disabled');
                }
            });
        }

        if (browserNotificationsSupported && Notification.permission === 'default') {
            Notification.requestPermission();
        }

        bindEvent(toggleDarkMode, 'change', function handleThemeToggle() {
            applyTheme(this.checked ? 'dark' : 'light', toggleDarkMode);
        });
    }

    function initializeVoiceAssistant() {
        const closeBtn = getById('closeAiAssistant');

        bindEvent(closeBtn, 'click', () => {
            speechSynthesis.cancel();

            const voiceAnimation = getById('voiceAnimation');
            const aiBox = getById('aiAssistantBox');
            if (voiceAnimation) voiceAnimation.style.display = 'none';
            if (aiBox) aiBox.classList.add('hide');
        });

        window.addEventListener('load', () => {
            setTimeout(() => speakMessage(getGreetingMessage()), TIMING.initialGreetingDelayMs);
        });
    }

    initializeGreetingRotation();
    initializeSettingsPanel();
    initializeSettingsToggles();
    initializeVoiceAssistant();

    window.speakMessage = speakMessage;
    window.showNativeNotification = showNativeNotification;
}());
