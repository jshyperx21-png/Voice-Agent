/**
 * Sarvam TTS deep debug — logs every single message received
 */
import WebSocket from 'ws';
import fs from 'fs';

const env = {};
fs.readFileSync('.env', 'utf8').split(/\r?\n/).forEach(line => {
  const m = line.match(/^([^#=]+)=(.*)$/);
  if (m) env[m[1].trim()] = m[2].trim();
});

const KEY = env.SARVAM_API_KEY;

const params = new URLSearchParams({ model: env.SARVAM_TTS_MODEL || 'bulbul:v3', send_completion_event: 'true' });
const ws = new WebSocket(`wss://api.sarvam.ai/text-to-speech/ws?${params}`, {
  headers: { 'api-subscription-key': KEY }
});

console.log('[TTS Debug] Connecting to Sarvam TTS...');

ws.on('open', () => {
  console.log('[TTS Debug] Connected. Sending config...');
  ws.send(JSON.stringify({
    type: 'config',
    data: {
      speaker: env.SARVAM_TTS_SPEAKER || 'priya',
      target_language_code: env.SARVAM_TTS_LANGUAGE || 'ta-IN',
      pace: Number(env.SARVAM_TTS_PACE || 1),
      temperature: Number(env.SARVAM_TTS_TEMPERATURE || 0.4),
      min_buffer_size: Number(env.SARVAM_TTS_MIN_BUFFER_SIZE || 50),
      max_chunk_length: Number(env.SARVAM_TTS_MAX_CHUNK_LENGTH || 200),
      output_audio_codec: env.SARVAM_TTS_AUDIO_CODEC || 'mulaw',
      speech_sample_rate: Number(env.SARVAM_TTS_SAMPLE_RATE || 8000)
    }
  }));

  console.log('[TTS Debug] Sending text: வணக்கம்...');
  ws.send(JSON.stringify({ type: 'text', data: { text: 'வணக்கம், நான் Karthika பேசுறேன்.' } }));

  console.log('[TTS Debug] Sending flush...');
  ws.send(JSON.stringify({ type: 'flush' }));
});

ws.on('message', (data) => {
  try {
    const msg = JSON.parse(data.toString());
    const audio = msg.data?.audio || msg.audio || msg.data?.audio_chunk;
    if (audio) {
      const bytes = Buffer.from(audio, 'base64').length;
      console.log(`[TTS Debug] Got AUDIO chunk: ${bytes} bytes`);
    } else {
      console.log('[TTS Debug] Got message:', JSON.stringify(msg).substring(0, 300));
    }
  } catch (e) {
    console.log('[TTS Debug] Non-JSON binary frame:', data.length, 'bytes');
  }
});

ws.on('close', (code, reason) => {
  console.log(`[TTS Debug] Closed. Code: ${code}, Reason: ${reason?.toString()}`);
  process.exit(0);
});

ws.on('error', (err) => {
  console.error('[TTS Debug] Error:', err.message);
  process.exit(1);
});

setTimeout(() => {
  console.log('[TTS Debug] Timeout — closing.');
  ws.close();
}, 15000);
