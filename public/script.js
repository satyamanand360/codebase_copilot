document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('prompt-input');
    const history = document.getElementById('chat-history');

    // Auto-resize textarea
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = 'auto';
    });

    // Handle enter to submit (shift+enter for new line)
    input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form.dispatchEvent(new Event('submit'));
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = input.value.trim();
        if (!message) return;

        // 1. Add User Message
        appendMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        // 2. Add Loading Indicator
        const loadingId = 'loading-' + Date.now();
        appendLoading(loadingId);
        scrollToBottom();

        try {
            // 3. Fetch from FastAPI Backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            
            // 4. Remove Loading and Add AI Message
            document.getElementById(loadingId).remove();
            
            if (response.ok) {
                appendMessage('system', data.response, data.intent);
            } else {
                appendMessage('system', `**Error:** ${data.detail || 'Failed to communicate with API.'}`);
            }

        } catch (error) {
            document.getElementById(loadingId).remove();
            appendMessage('system', `**Connection Error:** ${error.message}`);
        }

        scrollToBottom();
    });

    function appendMessage(role, text, intent = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? 'U' : 'AI';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        // Add intent badge if provided
        if (intent && intent !== 'unknown') {
            const badge = document.createElement('div');
            badge.className = 'intent-badge';
            badge.textContent = `Intent: ${intent}`;
            bubble.appendChild(badge);
        }

        // Render Markdown
        const content = document.createElement('div');
        content.innerHTML = marked.parse(text);
        
        // Apply syntax highlighting
        content.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        bubble.appendChild(content);
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);
        
        history.appendChild(msgDiv);
    }

    function appendLoading(id) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message system loading`;
        msgDiv.id = id;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = 'AI';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        const dotContainer = document.createElement('div');
        dotContainer.style.position = 'relative';
        dotContainer.style.width = '24px';
        dotContainer.style.height = '6px';
        dotContainer.style.margin = '0 10px';
        
        const dot = document.createElement('div');
        dot.className = 'dot-flashing';
        
        dotContainer.appendChild(dot);
        bubble.appendChild(dotContainer);
        
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(bubble);
        
        history.appendChild(msgDiv);
    }

    function scrollToBottom() {
        const container = document.querySelector('.chat-container');
        container.scrollTop = container.scrollHeight;
    }
});
