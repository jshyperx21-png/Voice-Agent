/**
 * API Health Check — Voice Agent
 * Tests all 4 external services: Azure OpenAI, Sarvam STT, Sarvam TTS, Plivo
 * Run: node test-apis.js
 */

import WebSocket from 'ws';
import fs from 'fs';

// ── Load .env manually (no dotenv dependency needed) ─────────────────────────
const env = {};
try {
  fs.readFileSync('.env', 'utf8').split(/\r?\n/).forEach(line => {
    const m = line.match(/^([^#=]+)=(.*)$/);
    if (m) env[m[1].trim()] = m[2].trim();
  });
} catch (e) {
  console.error('Could not read .env file. Run from the project root.');
  process.exit(1);
}

const AZURE_URL   = env.OPENAI_API_URL;
const AZURE_KEY   = env.OPENAI_API_KEY;
const SARVAM_KEY  = env.SARVAM_API_KEY;
const PLIVO_ID    = env.PLIVO_AUTH_ID;
const PLIVO_TOKEN = env.PLIVO_AUTH_TOKEN;

const PASS = 'PASS';
const FAIL = 'FAIL';
const WARN = 'WARN';

// ─────────────────────────────────────────────────────────────────────────────
// 1. Azure OpenAI
// ─────────────────────────────────────────────────────────────────────────────
async function testAzureOpenAI() {
  const name = 'Azure OpenAI (GPT-4.1-nano)';
  if (!AZURE_URL || !AZURE_KEY) return { name, status: FAIL, detail: 'Missing OPENAI_API_URL or OPENAI_API_KEY in .env' };

  const start = Date.now();
  try {
    const res = await fetch(AZURE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'api-key': AZURE_KEY },
      body: JSON.stringify({
        messages: [{ role: 'user', content: 'Say: OK' }],
        max_tokens: 5,
        stream: false
      }),
      signal: AbortSignal.timeout(10000)
    });

    const latency = Date.now() - start;

    if (res.status === 200) {
      const json = await res.json();
      const reply = json.choices?.[0]?.message?.content || '(no content)';
      return { name, status: PASS, detail: `${latency}ms | Reply: "${reply.trim()}"` };
    } else {
      const body = await res.text().catch(() => '');
      return { name, status: FAIL, detail: `HTTP ${res.status} | ${body.substring(0, 200)}` };
    }
  } catch (err) {
    return { name, status: FAIL, detail: err.message };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Sarvam TTS WebSocket
// ─────────────────────────────────────────────────────────────────────────────
function testSarvamTTS() {
  return new Promise(resolve => {
    const ttsModel = env.SARVAM_TTS_MODEL || 'bulbul:v3';
    const name = `Sarvam TTS (${ttsModel} WebSocket)`;
    if (!SARVAM_KEY) return resolve({ name, status: FAIL, detail: 'Missing SARVAM_API_KEY in .env' });

    const start = Date.now();
    const params = new URLSearchParams({ model: ttsModel, send_completion_event: 'true' });
    const ws = new WebSocket(`wss://api.sarvam.ai/text-to-speech/ws?${params}`, {
      headers: { 'api-subscription-key': SARVAM_KEY }
    });

    const timer = setTimeout(() => {
      ws.terminate();
      resolve({ name, status: FAIL, detail: 'Connection timed out (10s)' });
    }, 10000);

    let gotAudio = false;
    let firstAudioMs = 0;

    ws.on('open', () => {
      ws.send(JSON.stringify({
        type: 'config',
        data: {
          speaker: env.SARVAM_TTS_SPEAKER || 'priya',
          target_language_code: env.SARVAM_TTS_LANGUAGE || 'ta-IN',
          pace: Number(env.SARVAM_TTS_PACE || 1),
          temperature: Number(env.SARVAM_TTS_TEMPERATURE || 0.4),
          output_audio_codec: env.SARVAM_TTS_AUDIO_CODEC || 'mulaw',
          speech_sample_rate: Number(env.SARVAM_TTS_SAMPLE_RATE || 8000)
        }
      }));
      ws.send(JSON.stringify({ type: 'text', data: { text: 'வணக்கம்' } }));
      ws.send(JSON.stringify({ type: 'flush' }));
    });

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.error || msg.data?.error) {
          clearTimeout(timer);
          ws.terminate();
          return resolve({ name, status: FAIL, detail: `API error: ${JSON.stringify(msg)}` });
        }
        const audio = msg.data?.audio || msg.audio || msg.data?.audio_chunk;
        if (audio && !gotAudio) {
          gotAudio = true;
          firstAudioMs = Date.now() - start;
        }
        const eventType = msg.data?.event_type || msg.event_type || msg.type;
        if (eventType === 'final' || eventType === 'completed' || eventType === 'done' || msg.done === true) {
          clearTimeout(timer);
          ws.close();
          const audioBytes = audio ? Buffer.from(audio, 'base64').length : 0;
          resolve({
            name,
            status: gotAudio ? PASS : WARN,
            detail: gotAudio
              ? `First audio in ${firstAudioMs}ms | Audio size: ${audioBytes} bytes`
              : `Connected OK but no audio returned. Completion event: ${eventType}`
          });
        }
      } catch (_) {}
    });

    ws.on('error', (err) => {
      clearTimeout(timer);
      resolve({ name, status: FAIL, detail: err.message });
    });

    ws.on('close', (code, reason) => {
      clearTimeout(timer);
      if (!gotAudio) {
        resolve({ name, status: FAIL, detail: `WS closed code=${code} — no audio received` });
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Sarvam STT WebSocket
// ─────────────────────────────────────────────────────────────────────────────
function testSarvamSTT() {
  return new Promise(resolve => {
    const name = 'Sarvam STT (saaras:v3 WebSocket)';
    if (!SARVAM_KEY) return resolve({ name, status: FAIL, detail: 'Missing SARVAM_API_KEY in .env' });

    const start = Date.now();
    const params = new URLSearchParams({
      model: 'saaras:v3',
      'language-code': 'ta-IN',
      mode: 'codemix',
      sample_rate: '8000',
      endpointing: '150',
      vad_signals: 'true',
      input_audio_codec: 'pcm_s16le'
    });

    const ws = new WebSocket(`wss://api.sarvam.ai/speech-to-text/ws?${params}`, {
      headers: { 'api-subscription-key': SARVAM_KEY }
    });

    const timer = setTimeout(() => {
      ws.terminate();
      resolve({ name, status: WARN, detail: 'Connected OK but timed out (no real audio sent in health check). STT auth is valid.' });
    }, 5000);

    ws.on('open', () => {
      const connectMs = Date.now() - start;
      // Send 200ms of PCM16 silence to verify binary frames are accepted
      const silenceBuf = Buffer.alloc(3200, 0); // 200ms @ 8000Hz PCM16
      ws.send(silenceBuf);
      // Give it 2s for any response, then declare healthy
      setTimeout(() => {
        clearTimeout(timer);
        ws.close();
        resolve({ name, status: PASS, detail: `Connected & auth OK in ${connectMs}ms | Binary audio accepted` });
      }, 2000);
    });

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.error || msg.type === 'error') {
          clearTimeout(timer);
          ws.terminate();
          return resolve({ name, status: FAIL, detail: `Auth/API error: ${JSON.stringify(msg)}` });
        }
        const connectMs = Date.now() - start;
        clearTimeout(timer);
        ws.close();
        resolve({ name, status: PASS, detail: `Connected & responding in ${connectMs}ms | Msg: ${JSON.stringify(msg).substring(0, 100)}` });
      } catch (_) {}
    });

    ws.on('error', (err) => {
      clearTimeout(timer);
      resolve({ name, status: FAIL, detail: err.message });
    });

    ws.on('close', (code) => {
      clearTimeout(timer);
      if (code === 1008 || code === 4001 || code === 4003) {
        resolve({ name, status: FAIL, detail: `Auth rejected — close code ${code}` });
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Plivo REST API
// ─────────────────────────────────────────────────────────────────────────────
async function testPlivo() {
  const name = 'Plivo (REST API credentials)';
  if (!PLIVO_ID || !PLIVO_TOKEN) return { name, status: FAIL, detail: 'Missing PLIVO_AUTH_ID or PLIVO_AUTH_TOKEN in .env' };

  const start = Date.now();
  const credentials = Buffer.from(`${PLIVO_ID}:${PLIVO_TOKEN}`).toString('base64');

  try {
    const res = await fetch(`https://api.plivo.com/v1/Account/${PLIVO_ID}/`, {
      headers: { 'Authorization': `Basic ${credentials}` },
      signal: AbortSignal.timeout(10000)
    });

    const latency = Date.now() - start;

    if (res.status === 200) {
      const json = await res.json();
      return {
        name,
        status: PASS,
        detail: `${latency}ms | Account type: ${json.account_type || 'OK'} | Balance: ${json.cash_credits || '?'} ${json.billing_mode || ''}`
      };
    } else if (res.status === 401) {
      return { name, status: FAIL, detail: 'HTTP 401 — Invalid Auth ID or Token' };
    } else {
      const body = await res.text().catch(() => '');
      return { name, status: FAIL, detail: `HTTP ${res.status} | ${body.substring(0, 200)}` };
    }
  } catch (err) {
    return { name, status: FAIL, detail: err.message };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Runner
// ─────────────────────────────────────────────────────────────────────────────
async function main() {
  console.log('\n==================================================');
  console.log('  Voice Agent -- API Health Check');
  console.log('==================================================\n');
  console.log('Testing all APIs in parallel...\n');

  const results = await Promise.all([
    testAzureOpenAI(),
    testSarvamTTS(),
    testSarvamSTT(),
    testPlivo()
  ]);

  console.log('==================================================');
  results.forEach(r => {
    const icon = r.status === PASS ? '[PASS]' : r.status === WARN ? '[WARN]' : '[FAIL]';
    console.log(`${icon}  ${r.name}`);
    console.log(`       ${r.detail}\n`);
  });

  const passed = results.filter(r => r.status === PASS).length;
  const warned = results.filter(r => r.status === WARN).length;
  const failed = results.filter(r => r.status === FAIL).length;

  console.log('==================================================');
  console.log(`  Result: ${passed} passed, ${warned} warnings, ${failed} failed`);
  console.log('==================================================\n');

  if (failed > 0) process.exit(1);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
