const GATEWAY_URL = 'http://localhost:8000';

async function generateText() {
    const button = document.getElementById('generateBtn');
    const output = document.getElementById('output');
    const metrics = document.getElementById('metrics');
    const prompt = document.getElementById('prompt').value;
    const maxTokens = parseInt(document.getElementById('maxTokens').value);
    const temperature = parseFloat(document.getElementById('temperature').value);

    if (!prompt.trim()) {
        alert('Please enter a prompt');
        return;
    }

    // Disable button and show loading
    button.disabled = true;
    button.innerHTML = 'Generating... <span class="loading"></span>';
    output.innerHTML = '<p class="placeholder">Generating response...</p>';
    metrics.style.display = 'none';

    const startTime = Date.now();

    try {
        const response = await fetch(`${GATEWAY_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temperature
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const responseTime = Date.now() - startTime;

        // Display generated text
        output.innerHTML = `<p>${data.generated_text || data.total_text || 'Response generated successfully!'}</p>`;

        // Display metrics
        metrics.style.display = 'block';
        document.getElementById('worker').textContent = data.worker_id || data.processed_by || 'N/A';
        document.getElementById('responseTime').textContent = `${responseTime}ms`;
        document.getElementById('cacheHit').textContent = data.cache_hit ? '✅ Yes' : '❌ No';
        document.getElementById('tokens').textContent = data.tokens_generated || 'N/A';

    } catch (error) {
        console.error('Error:', error);
        output.innerHTML = `<p style="color: red;">Error: ${error.message}</p>
                           <p style="margin-top: 10px; font-size: 14px;">
                           Make sure the system is running: <code>./scripts/start_system.sh</code>
                           </p>`;
    } finally {
        button.disabled = false;
        button.textContent = 'Generate Text';
    }
}

// Test connection on load
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${GATEWAY_URL}/health`);
        if (response.ok) {
            console.log('✅ Connected to gateway');
        }
    } catch (error) {
        console.warn('⚠️ Gateway not responding. Please start the system.');
        const output = document.getElementById('output');
        output.innerHTML = `
            <p style="color: orange;">⚠️ System not running</p>
            <p style="margin-top: 10px;">Start the demo with:</p>
            <pre style="background: #f0f0f0; padding: 10px; border-radius: 4px; margin-top: 10px;">
docker-compose -f docker-compose.demo.yml up
            </pre>
        `;
    }
});
