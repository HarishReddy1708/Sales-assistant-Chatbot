document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const messagesContainer = document.getElementById('messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const nameScreen = document.getElementById('name-screen');
    const menuScreen = document.getElementById('menu-screen');
    const chatScreen = document.getElementById('chat-screen');
    const chatInput = document.getElementById('chat-input');
    const backToMenuButton = document.getElementById('back-to-menu');
    const nameInput = document.getElementById('name-input');
    const startChatButton = document.getElementById('start-chat');
    const menuOptions = document.querySelectorAll('.menu-option');

    let userName = '';
    let currentMenuOption = '';
    let isProcessing = false;
    let selectedModel = '';

    // Porsche website links
    const PORSCHE_LINKS = {
        models: {
            '911': 'https://www.porsche.com/usa/models/911/',
            'taycan': 'https://www.porsche.com/usa/models/taycan/',
            'macan': 'https://www.porsche.com/usa/models/macan/',
            'cayenne': 'https://www.porsche.com/usa/models/cayenne/',
            'panamera': 'https://www.porsche.com/usa/models/panamera/',
            '718': 'https://www.porsche.com/usa/models/718/'
        },
        dealership: 'https://www.porsche.com/usa/dealer-locator/',
        testDrive: 'https://www.porsche.com/usa/events-and-experiences/',
        build: 'https://www.porsche.com/usa/modelstart/'
    };

    // Menu option responses with interactive elements
    const menuResponses = {
        models: {
            message: "Here are our current Porsche models. Click on any model to know more:",
            options: [
                { name: "911", link: PORSCHE_LINKS.models['911'] },
                { name: "Taycan", link: PORSCHE_LINKS.models['taycan'] },
                { name: "Macan", link: PORSCHE_LINKS.models['macan'] },
                { name: "Cayenne", link: PORSCHE_LINKS.models['cayenne'] },
                { name: "Panamera", link: PORSCHE_LINKS.models['panamera'] },
                { name: "718", link: PORSCHE_LINKS.models['718'] }
            ]
        },
        dealerships: {
            message: " Please enter nearest City/ZipCode or You can find your nearest Porsche dealership at below link:",
            options: [
                { name: "Dealer Locator", link: PORSCHE_LINKS.dealership }
            ]
        },
        'test-drive': {
            message: "Experience Porsche:",
            options: [
                { name: "Schedule a Test Drive", link: PORSCHE_LINKS.testDrive },
                { name: "Porsche Experience Center", link: "https://www.porsche.com/usa/events-and-experiences/porsche-experience-center/" }
            ]
        },
        build: {
            message: "Build your dream Porsche:",
            options: [
                { name: "Porsche Configurator", link: PORSCHE_LINKS.build }
            ]
        }
    };

    function showScreen(screen) {
        nameScreen.classList.add('hidden');
        menuScreen.classList.add('hidden');
        chatScreen.classList.add('hidden');
        chatInput.classList.add('hidden');
        backToMenuButton.classList.add('hidden');
        
        screen.classList.remove('hidden');
        if (screen === chatScreen) {
            chatInput.classList.remove('hidden');
            backToMenuButton.classList.remove('hidden');
        }
    }

    function addMessage(message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `flex items-start message ${isUser ? 'justify-end' : ''}`;
        
        const avatar = isUser ? '👤' : '🤖';
        const bgColor = isUser ? 'bg-blue-100' : 'bg-gray-100';
        
        messageDiv.innerHTML = `
            ${!isUser ? `
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                        ${avatar}
                    </div>
                </div>
            ` : ''}
            <div class="${isUser ? 'mr-3' : 'ml-3'} ${bgColor} rounded-lg p-3 max-w-[70%]">
                <p class="text-sm text-gray-800">${message}</p>
            </div>
            ${isUser ? `
                <div class="flex-shrink-0">
                    <div class="h-8 w-8 rounded-full bg-blue-300 flex items-center justify-center">
                        ${avatar}
                    </div>
                </div>
            ` : ''}
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function addInteractiveOptions(options) {
        const optionsDiv = document.createElement('div');
        optionsDiv.className = 'flex flex-col space-y-2 mt-2';
        
        options.forEach(option => {
            const button = document.createElement('button');
            button.className = 'text-left p-2 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors';
            button.innerHTML = `
                <span class="text-blue-600">${option.name}</span>
                <span class="text-xs text-gray-500 ml-2">→</span>
            `;
            button.onclick = () => {
                window.open(option.link, '_blank');
                if (currentMenuOption === 'models') {
                    selectedModel = option.name.toLowerCase();
                    addMessage(`You selected the ${option.name}. What would you like to know about it?`);
                }
            };
            optionsDiv.appendChild(button);
        });
        
        messagesContainer.appendChild(optionsDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'flex items-start message';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="flex-shrink-0">
                <div class="h-8 w-8 rounded-full bg-gray-300 flex items-center justify-center">
                    🤖
                </div>
            </div>
            <div class="ml-3 bg-gray-100 rounded-lg p-3">
                <div class="typing-indicator"></div>
            </div>
        `;
        messagesContainer.appendChild(indicator);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function setProcessingState(processing) {
        isProcessing = processing;
        userInput.disabled = processing;
        sendButton.disabled = processing;
        if (processing) {
            sendButton.classList.add('opacity-50', 'cursor-not-allowed');
            userInput.classList.add('opacity-50', 'cursor-not-allowed');
        } else {
            sendButton.classList.remove('opacity-50', 'cursor-not-allowed');
            userInput.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (message && !isProcessing) {
            addMessage(message, true);
            userInput.value = '';
            showTypingIndicator();
            setProcessingState(true);
            
            // Send message with current menu context and selected model
            socket.emit('message', {
                message: message,
                menu_context: currentMenuOption,
                selected_model: selectedModel
            });
        }
    }

    // Name input handling
    startChatButton.addEventListener('click', () => {
        userName = nameInput.value.trim();
        if (userName) {
            showScreen(menuScreen);
            addMessage(`Welcome, ${userName}! How can I assist you today?`);
        }
    });

    nameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            startChatButton.click();
        }
    });

    // Menu option handling
    menuOptions.forEach(option => {
        option.addEventListener('click', () => {
            currentMenuOption = option.dataset.option;
            showScreen(chatScreen);
            const response = menuResponses[currentMenuOption];
            addMessage(response.message);
            addInteractiveOptions(response.options);
        });
    });

    // Back to menu handling
    backToMenuButton.addEventListener('click', () => {
        showScreen(menuScreen);
        currentMenuOption = '';
        selectedModel = '';
    });

    // Chat input handling
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isProcessing) {
            sendMessage();
        }
    });

    socket.on('response', (response) => {
        removeTypingIndicator();
        addMessage(response);
        setProcessingState(false);
    });

    socket.on('error', (error) => {
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again.');
        setProcessingState(false);
        console.error('Socket error:', error);
    });
}); 