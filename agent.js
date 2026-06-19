import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

// ── Phase 6: Precomputed μ-law → PCM16 lookup table ──────────────────────────
// Built once at module load. Replaces per-byte math in the hot audio path.
const MULAW_TO_PCM16 = (() => {
  const table = new Int16Array(256);
  const BIAS = 0x84;
  for (let i = 0; i < 256; i++) {
    let value = (~i) & 0xff;
    const sign = value & 0x80;
    const exponent = (value >> 4) & 0x07;
    const mantissa = value & 0x0f;
    let sample = ((mantissa << 3) + BIAS) << exponent;
    sample -= BIAS;
    table[i] = sign ? -sample : sample;
  }
  return table;
})();

export class VoiceAgent {
  constructor(plivoWs, statusCallbacks) {
    this.plivoWs = plivoWs;
    this.statusCallbacks = statusCallbacks || {
      onStateChange: () => {},
      onUserTranscript: () => {},
      onAiTranscript: () => {},
      onAiTiming: () => {},
      onInterruption: () => {},
      onError: () => {}
    };

    this.streamId = null;
    this.history = [];
    this.isSpeaking = false;

    // Outbound audio buffer (kept for legacy reference)
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.playbackInterval = null;
    this.playbackDoneTimeout = null;
    this.outboundBytesSent = 0;

    // Active connection references
    this.sarvamSttWs = null;
    this.currentLlmController = null;

    this.userUtteranceBuffer = '';
    this.silenceTimeout = null;
    this.silenceTimeoutMs = Number(process.env.SILENCE_TIMEOUT_MS || 5000);
    this.endCallAfterSpeech = false;

    // Lifecycle state
    this.isClosed = false;
    this.greetingSpoken = false;
    this.sarvamSttFatalError = false;

    // Voice / LLM config
    this.openaiModelId = process.env.OPENAI_MODEL || 'gpt-4.1-nano';
    this.openaiApiUrl = process.env.OPENAI_API_URL || 'https://api.openai.com/v1/chat/completions';
    this.openaiProvider = process.env.OPENAI_PROVIDER || 'openai';
    this.isResponsesApi = /\/responses(?:\?|$)/.test(this.openaiApiUrl);
    this.isAzureOpenAI = this.openaiProvider.startsWith('azure') ||
                          this.openaiApiUrl.includes('.openai.azure.com') ||
                          this.openaiApiUrl.includes('.services.ai.azure.com');
    this.enableBargeIn = process.env.ENABLE_BARGE_IN === 'true';

    // Sarvam STT config
    this.sarvamApiKey = process.env.SARVAM_API_KEY;
    this.sarvamSttLanguage = process.env.SARVAM_STT_LANGUAGE || 'ta-IN';
    this.sarvamSttMode = process.env.SARVAM_STT_MODE || 'codemix';

    // Sarvam TTS config
    this.sarvamTtsApiKey = process.env.SARVAM_TTS_API_KEY || this.sarvamApiKey;
    this.sarvamTtsModel = process.env.SARVAM_TTS_MODEL || 'bulbul:v3';
    this.sarvamTtsLanguage = process.env.SARVAM_TTS_LANGUAGE || 'ta-IN';
    this.sarvamTtsSpeaker = process.env.SARVAM_TTS_SPEAKER || 'anushka';
    this.sarvamTtsPace = Number(process.env.SARVAM_TTS_PACE || 1);
    this.sarvamTtsMinBufferSize = Number(process.env.SARVAM_TTS_MIN_BUFFER_SIZE || 50);
    this.sarvamTtsMaxChunkLength = Number(process.env.SARVAM_TTS_MAX_CHUNK_LENGTH || 200);
    this.sarvamTtsAudioCodec = process.env.SARVAM_TTS_AUDIO_CODEC || 'mulaw';
    this.sarvamTtsSampleRate = Number(process.env.SARVAM_TTS_SAMPLE_RATE || 8000);

    // ── Phase 2: Config cache — no disk I/O per utterance ────────────────────
    this.cachedSystemPrompt = '';
    this.cachedKnowledgeBase = '';
    const { systemPrompt, knowledgeBase } = this.loadAgentFiles();
    this.cachedSystemPrompt = systemPrompt;
    this.cachedKnowledgeBase = knowledgeBase;

    // ── Phase 3: LLM → TTS streaming queue ───────────────────────────────────
    this.ttsQueue = [];              // Pending text chunks to synthesize
    this.isTtsProcessing = false;    // Is drainTtsQueue currently running?
    this.llmStreamFinished = false;  // Has the LLM stream fully completed?
    this.ttsStreamStartedAt = 0;     // Timestamp when first TTS chunk was queued
    this.ttsStreamTotalBytesSent = 0; // Accumulated audio bytes across all chunks

    // ── Phase 4: Persistent TTS WebSocket ────────────────────────────────────
    this.sarvamTtsWs = null;               // Persistent TTS WebSocket reference
    this.sarvamTtsWsConfigSent = false;    // Config frame sent on this connection?
    this.sarvamTtsCurrentResolve = null;   // Resolve fn for in-flight chunk
    this.sarvamTtsCurrentBytesSent = 0;    // Bytes received for current chunk

    // ── Response Timing Tracking ──────────────────────────────────────────────
    // Tracks exact latency for each conversational turn:
    //   turnStartedAt     → when user utterance is received by handleUserUtterance
    //   llmFirstTokenAt   → when the very first LLM token arrives from OpenAI
    //   firstAudioSentAt  → when the first audio chunk is sent to Plivo
    this.turnStartedAt = 0;
    this.llmFirstTokenAt = 0;
    this.firstAudioSentAt = 0;
    this.turnFirstAudioRecorded = false; // gate to fire onAiTiming only once per turn
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Start the AI connections.
   */
  async start() {
    console.log('[Agent] Initializing AI streams...');
    this.connectSarvamStt();
  }

  /**
   * Play the initial greeting automatically when the call connects.
   */
  speakGreeting() {
    if (this.isClosed) return;
    const greetingText = this.getGreetingText();
    console.log(`[Agent] Speaking greeting: "${greetingText}"`);

    this.isSpeaking = true;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = 0;

    this.history.push({ role: 'assistant', content: greetingText });
    this.statusCallbacks.onAiTranscript(greetingText, true);
    this.statusCallbacks.onStateChange('active');

    this.synthesizeSpeech(greetingText);
  }

  /**
   * Read system prompt and knowledge base from disk.
   * Called once in constructor and on hot-reload.
   */
  loadAgentFiles() {
    let systemPrompt = 'You are a warm and helpful voice assistant. Keep replies short and ask one question at a time.';
    let knowledgeBase = '';
    try {
      systemPrompt = fs.readFileSync(path.resolve('./system_prompt.txt'), 'utf8');
      knowledgeBase = fs.readFileSync(path.resolve('./knowledge_base.txt'), 'utf8');
    } catch (e) {
      console.warn('[Agent] Could not load system prompt or knowledge base files, using defaults.');
    }
    return { systemPrompt, knowledgeBase };
  }

  /**
   * Phase 2: Hot-reload config from disk without restarting the server.
   * Called by server.js POST /api/config after saving files.
   */
  reloadConfig() {
    const { systemPrompt, knowledgeBase } = this.loadAgentFiles();
    this.cachedSystemPrompt = systemPrompt;
    this.cachedKnowledgeBase = knowledgeBase;
    console.log('[Agent] System prompt and knowledge base reloaded from disk.');
  }

  /**
   * Phase 2: Read greeting from cached system prompt.
   */
  getGreetingText() {
    const greetingLine = this.cachedSystemPrompt
      .split(/\r?\n/)
      .map(line => line.trim())
      .find(line => /^GREETING\s*:/i.test(line));
    if (greetingLine) return greetingLine.replace(/^GREETING\s*:/i, '').trim();
    return 'Vanakkam, naan AI assistant pesuren. Eppadi help pannalam?';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Sarvam STT (Speech-to-Text)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Connect to Sarvam Saaras v3 streaming STT.
   */
  connectSarvamStt() {
    if (this.isClosed || this.sarvamSttFatalError) return;
    if (!this.sarvamApiKey) {
      console.error('[Sarvam STT] Missing API key in environment variables.');
      this.statusCallbacks.onError('Sarvam API Key is missing.');
      return;
    }

    const params = new URLSearchParams({
      model: 'saaras:v3',
      'language-code': this.sarvamSttLanguage,
      mode: this.sarvamSttMode,
      sample_rate: '8000',
      endpointing: '150',          // Phase 1: reduced from 250ms → 150ms
      vad_signals: 'true',
      high_vad_sensitivity: 'true',
      input_audio_codec: 'pcm_s16le'
    });

    const url = `wss://api.sarvam.ai/speech-to-text/ws?${params.toString()}`;
    console.log(`[Sarvam STT] Connecting. Language: ${this.sarvamSttLanguage}, Mode: ${this.sarvamSttMode}`);

    this.sarvamSttWs = new WebSocket(url, {
      headers: { 'api-subscription-key': this.sarvamApiKey }
    });

    this.sarvamSttWs.on('open', () => {
      console.log('[Sarvam STT] WebSocket established successfully.');
      if (!this.greetingSpoken) {
        this.greetingSpoken = true;
        setTimeout(() => this.speakGreeting(), 500);
      }
    });

    this.sarvamSttWs.on('message', (data) => {
      try {
        const response = JSON.parse(data);

        if (response.type === 'error' || response.error) {
          console.error('[Sarvam STT] API error:', response);
          this.statusCallbacks.onError(response.error || response.message || 'Sarvam STT error.');
          this.sarvamSttFatalError = true;
          return;
        }

        const transcript = (response.data?.transcript || response.transcript || '').trim();
        if (!transcript) return;

        const normalizedTranscript = this.normalizeTranscript(transcript);
        this.statusCallbacks.onUserTranscript(normalizedTranscript, true);
        console.log(`[Sarvam STT] Final Transcript: "${normalizedTranscript}"`);

        if (!this.isActionableUtterance(normalizedTranscript)) {
          console.log(`[Sarvam STT] Ignoring short/noisy transcript: "${normalizedTranscript}"`);
          return;
        }

        if (this.enableBargeIn && this.isSpeaking) {
          this.handleInterruption();
        }

        if (!this.isSpeaking) {
          this.handleUserUtterance(normalizedTranscript);
        } else {
          console.log(`[Sarvam STT] Ignoring transcript while agent is speaking: "${normalizedTranscript}"`);
        }
      } catch (err) {
        console.error('[Sarvam STT] Error parsing incoming message:', err);
      }
    });

    this.sarvamSttWs.on('close', (code, reason) => {
      console.log(`[Sarvam STT] Connection closed. Code: ${code}, Reason: ${reason}`);
      if (!this.isClosed && !this.sarvamSttFatalError) {
        setTimeout(() => this.connectSarvamStt(), 1000);
      }
    });

    this.sarvamSttWs.on('error', (error) => {
      console.error('[Sarvam STT] WebSocket Error:', error);
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Audio conversion (Phase 6: lookup table)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Phase 6: Convert single μ-law byte → PCM16 sample using lookup table.
   */
  mulawByteToPcm16(muLawByte) {
    return MULAW_TO_PCM16[muLawByte];
  }

  /**
   * Phase 6: Convert μ-law buffer → PCM16 buffer using lookup table.
   * ~10-30% faster than the per-byte math implementation.
   */
  mulawBufferToPcm16Buffer(muLawBuffer) {
    const pcmBuffer = Buffer.alloc(muLawBuffer.length * 2);
    for (let i = 0; i < muLawBuffer.length; i++) {
      pcmBuffer.writeInt16LE(MULAW_TO_PCM16[muLawBuffer[i]], i * 2);
    }
    return pcmBuffer;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Sarvam TTS — Phase 4: Persistent WebSocket
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Phase 4: Ensure the persistent Sarvam TTS WebSocket is open and configured.
   * Reuses existing connection if healthy. Opens a new one otherwise.
   * The config frame (speaker, language, etc.) is sent only once on open.
   */
  async ensureSarvamTtsConnected() {
    // Return immediately if already connected and configured
    if (
      this.sarvamTtsWs &&
      this.sarvamTtsWs.readyState === WebSocket.OPEN &&
      this.sarvamTtsWsConfigSent
    ) {
      return;
    }

    // Clean up any stale connection
    if (this.sarvamTtsWs) {
      try { this.sarvamTtsWs.terminate(); } catch (_) {}
      this.sarvamTtsWs = null;
      this.sarvamTtsWsConfigSent = false;
    }

    console.log(`[Sarvam TTS] Opening persistent connection. Model: ${this.sarvamTtsModel}, Speaker: ${this.sarvamTtsSpeaker}`);

    return new Promise((resolve, reject) => {
      const params = new URLSearchParams({
        model: this.sarvamTtsModel,
        send_completion_event: 'true'
      });
      const ws = new WebSocket(`wss://api.sarvam.ai/text-to-speech/ws?${params}`, {
        headers: { 'api-subscription-key': this.sarvamTtsApiKey }
      });

      const openTimeout = setTimeout(() => {
        console.error('[Sarvam TTS] Connection timed out.');
        try { ws.terminate(); } catch (_) {}
        reject(new Error('Sarvam TTS connection timed out.'));
      }, 10000);

      ws.on('open', () => {
        clearTimeout(openTimeout);
        // Send config frame once on this connection
        ws.send(JSON.stringify({
          type: 'config',
          data: {
            speaker: this.sarvamTtsSpeaker,
            target_language_code: this.sarvamTtsLanguage,
            pace: this.sarvamTtsPace,
            min_buffer_size: this.sarvamTtsMinBufferSize,
            max_chunk_length: this.sarvamTtsMaxChunkLength,
            output_audio_codec: this.sarvamTtsAudioCodec,
            speech_sample_rate: this.sarvamTtsSampleRate
          }
        }));
        this.sarvamTtsWsConfigSent = true;
        console.log('[Sarvam TTS] Persistent connection established and configured.');
        resolve();
      });

      // Persistent message handler — routes audio and completion events
      // to the currently pending chunk resolve callback
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());

          if (message.type === 'error' || message.error || message.data?.error) {
            console.error('[Sarvam TTS] API error:', message);
            if (this.sarvamTtsCurrentResolve) {
              const res = this.sarvamTtsCurrentResolve;
              this.sarvamTtsCurrentResolve = null;
              res(0);
            }
            return;
          }

          const audioBase64 = message.data?.audio || message.audio || message.data?.audio_chunk;
          if (audioBase64) {
            const audioChunk = Buffer.from(audioBase64, 'base64');
            if (audioChunk.length > 0 && this.isSpeaking && !this.isClosed) {
              this.sarvamTtsCurrentBytesSent += audioChunk.length;
              this.sendAudioToPlivo(audioChunk);

              // ── Timing: record first audio sent this turn ──────────────────
              if (!this.turnFirstAudioRecorded && this.turnStartedAt > 0) {
                this.turnFirstAudioRecorded = true;
                this.firstAudioSentAt = Date.now();
                const totalMs = this.firstAudioSentAt - this.turnStartedAt;
                const llmMs = this.llmFirstTokenAt
                  ? this.llmFirstTokenAt - this.turnStartedAt
                  : 0;
                const ttsMs = this.llmFirstTokenAt
                  ? this.firstAudioSentAt - this.llmFirstTokenAt
                  : totalMs;
                console.log(`[Timing] ⚡ Total: ${totalMs}ms | LLM first token: ${llmMs}ms | TTS first audio: ${ttsMs}ms`);
                this.statusCallbacks.onAiTiming({ llmMs, ttsMs, totalMs });
              }
            }
          }

          const eventType = message.data?.event_type || message.event_type || message.type;
          if (
            eventType === 'final' || eventType === 'completed' ||
            eventType === 'done' || message.done === true
          ) {
            if (this.sarvamTtsCurrentResolve) {
              const res = this.sarvamTtsCurrentResolve;
              const bytesSent = this.sarvamTtsCurrentBytesSent;
              this.sarvamTtsCurrentResolve = null;
              this.sarvamTtsCurrentBytesSent = 0;
              res(bytesSent);
            }
          }
        } catch (err) {
          console.error('[Sarvam TTS] Message parse error:', err);
        }
      });

      ws.on('error', (err) => {
        console.error('[Sarvam TTS] Persistent WS error:', err);
        this.sarvamTtsWs = null;
        this.sarvamTtsWsConfigSent = false;
        // Unblock any in-flight chunk
        if (this.sarvamTtsCurrentResolve) {
          const res = this.sarvamTtsCurrentResolve;
          this.sarvamTtsCurrentResolve = null;
          res(0);
        }
        clearTimeout(openTimeout);
        reject(err);
      });

      ws.on('close', () => {
        console.log('[Sarvam TTS] Persistent WS closed.');
        this.sarvamTtsWs = null;
        this.sarvamTtsWsConfigSent = false;
        // Unblock any in-flight chunk
        if (this.sarvamTtsCurrentResolve) {
          const res = this.sarvamTtsCurrentResolve;
          this.sarvamTtsCurrentResolve = null;
          res(0);
        }
      });

      this.sarvamTtsWs = ws;
    });
  }

  /**
   * Phase 4: Synthesize a single text chunk using the persistent TTS WebSocket.
   * Sends text + flush, awaits the 'final' event, returns number of bytes sent.
   * Used by Phase 3's drainTtsQueue().
   */
  async synthesizeChunkWithSarvam(text) {
    if (!this.sarvamTtsApiKey) {
      console.error('[Sarvam TTS] Missing API key.');
      return 0;
    }
    if (!text || !text.trim()) return 0;

    try {
      await this.ensureSarvamTtsConnected();
    } catch (err) {
      console.error('[Sarvam TTS] Failed to establish connection:', err);
      return 0;
    }

    if (!this.sarvamTtsWs || this.sarvamTtsWs.readyState !== WebSocket.OPEN) {
      return 0;
    }

    // Reset per-chunk tracking
    this.sarvamTtsCurrentBytesSent = 0;

    return new Promise((resolve) => {
      let settled = false;

      const chunkTimeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        console.error('[Sarvam TTS] Chunk synthesis timed out, continuing.');
        if (this.sarvamTtsCurrentResolve) {
          this.sarvamTtsCurrentResolve = null;
        }
        resolve(0);
      }, 15000);

      this.sarvamTtsCurrentResolve = (bytesSent) => {
        if (settled) return;
        settled = true;
        clearTimeout(chunkTimeout);
        resolve(bytesSent);
      };

      // Send text + flush to trigger synthesis of this chunk
      this.sarvamTtsWs.send(JSON.stringify({ type: 'text', data: { text: text.trim() } }));
      this.sarvamTtsWs.send(JSON.stringify({ type: 'flush' }));
    });
  }

  /**
   * Entry point for TTS. Used by speakGreeting() and speakAndMaybeEnd().
   * Synthesizes full text, then waits for estimated playback to complete before
   * marking the agent idle (so the caller's next words are captured).
   */
  async synthesizeSpeech(text) {
    const cleanText = text.replace(/\s+/g, ' ').trim();
    if (!cleanText || this.isClosed || !this.isSpeaking) return false;

    const startedAt = Date.now();
    const bytesSent = await this.synthesizeChunkWithSarvam(cleanText);

    if (bytesSent > 0 && this.isSpeaking && !this.isClosed) {
      const elapsedMs = Date.now() - startedAt;
      const playbackMs = Math.ceil((bytesSent / this.sarvamTtsSampleRate) * 1000);
      const remainingMs = Math.max(0, playbackMs - elapsedMs) + 250;
      setTimeout(() => {
        const secondsSent = (bytesSent / this.sarvamTtsSampleRate).toFixed(2);
        this.markAgentIdle(`[Sarvam TTS] Audio complete (${secondsSent}s). Agent idle.`);
      }, remainingMs);
      return true;
    } else {
      if (this.isSpeaking && !this.isClosed) {
        this.markAgentIdle('[Sarvam TTS] No audio produced. Agent idle.');
      }
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 3: LLM → TTS streaming pipeline
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Phase 3: Detect Tamil/Tanglish/English sentence boundaries in the LLM token stream.
   * Returns true when the accumulated buffer should be flushed to TTS immediately.
   *
   * Rules:
   *  - Hard boundary (.!?) with ≥1 word  → speak immediately (catches "சரி.", "Okay.")
   *  - Soft boundary (,;।\n) with ≥3 words → speak (avoids super-short fragments)
   *  - 12+ words with no boundary        → force-flush (prevents runaway buffers)
   */
  isSpeakableBoundary(buffer) {
    const trimmed = buffer.trim();
    if (!trimmed) return false;
    const wordCount = trimmed.split(/\s+/).filter(Boolean).length;
    if (wordCount >= 12) return true; // force flush
    if (/[.!?]$/.test(trimmed) && wordCount >= 1) return true;
    if (/[,;।\n]$/.test(trimmed) && wordCount >= 3) return true;
    return false;
  }

  /**
   * Phase 3: Push a text chunk into the TTS queue.
   * Starts the drain loop if not already running.
   */
  enqueueTtsChunk(text) {
    if (!text || !text.trim() || this.isClosed || !this.isSpeaking) return;
    this.ttsQueue.push(text.trim());
    if (!this.isTtsProcessing) {
      this.drainTtsQueue();
    }
  }

  /**
   * Phase 3: Drain the TTS queue serially.
   * Processes one chunk at a time so audio plays in order.
   * After the queue empties AND the LLM stream is finished, marks agent idle.
   */
  async drainTtsQueue() {
    if (this.isTtsProcessing) return; // Already draining
    this.isTtsProcessing = true;

    while (this.ttsQueue.length > 0 && this.isSpeaking && !this.isClosed) {
      const chunk = this.ttsQueue.shift();
      const bytesSent = await this.synthesizeChunkWithSarvam(chunk);
      if (bytesSent > 0) this.ttsStreamTotalBytesSent += bytesSent;
    }

    this.isTtsProcessing = false;

    // Both conditions must be true to declare idle:
    // 1. LLM has fully finished streaming
    // 2. TTS queue is fully drained
    if (this.llmStreamFinished && this.ttsQueue.length === 0 && this.isSpeaking && !this.isClosed) {
      const elapsedMs = Date.now() - this.ttsStreamStartedAt;
      const playbackMs = Math.ceil((this.ttsStreamTotalBytesSent / this.sarvamTtsSampleRate) * 1000);
      const remainingMs = Math.max(0, playbackMs - elapsedMs) + 250;
      setTimeout(() => {
        this.markAgentIdle('[Agent] All streaming TTS chunks complete. Agent idle.');
      }, remainingMs);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Transcript helpers
  // ─────────────────────────────────────────────────────────────────────────

  normalizeTranscript(text) {
    return text.replace(/\s+/g, ' ').trim();
  }

  /**
   * Drop tiny STT fragments without blocking common short caller intents.
   */
  isActionableUtterance(text) {
    const normalized = text.trim().toLowerCase();
    const words = normalized.split(/\s+/).filter(Boolean);
    const tamilCharCount = (normalized.match(/[\u0B80-\u0BFF]/g) || []).length;
    const shortTamilIntents = new Set([
      '\u0BAE\u0BCD',             // ம்
      '\u0BAE\u0BCD\u0BAE\u0BCD', // ம்ம்
      '\u0B86',                   // ஆ
      '\u0B86\u0BAE\u0BCD',       // ஆம்
      '\u0B86\u0BAE\u0BBE',       // ஆமா
      '\u0B9A\u0BB0\u0BBF',       // சரி
      '\u0B87\u0BB2\u0BCD\u0BB2\u0BC8', // இல்லை
      '\u0B93\u0B95\u0BC7'        // ஓகே
    ]);

    if (/^(hmm|hm|um|uh|mmm|mm)$/i.test(normalized)) return false;
    if (shortTamilIntents.has(normalized)) return true;
    if (words.length >= 3) return true;
    if (normalized.length >= 3) return true;
    if (tamilCharCount > 0 && words.length >= 2) return true;
    if (/^\d{1,3}$/.test(normalized)) return true;
    if (/^\d{1,2}(:\d{2})?\s?(am|pm)?$/i.test(normalized)) return true;
    return false;
  }

  extractOpenAIStreamToken(data) {
    if (!data || typeof data !== 'object') return '';
    if (typeof data.delta === 'string') return data.delta;
    if (typeof data.text === 'string' && data.type === 'response.output_text.delta') return data.text;
    if (typeof data.output_text === 'string') return data.output_text;
    const outputContent = data.response?.output?.[0]?.content?.[0];
    if (typeof outputContent?.text === 'string') return outputContent.text;
    return data.choices?.[0]?.delta?.content || '';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Silence & call flow helpers
  // ─────────────────────────────────────────────────────────────────────────

  clearSilenceTimer() {
    if (this.silenceTimeout) {
      clearTimeout(this.silenceTimeout);
      this.silenceTimeout = null;
    }
  }

  scheduleSilenceTimeout() {
    this.clearSilenceTimer();
    if (this.isClosed || this.endCallAfterSpeech) return;
    this.silenceTimeout = setTimeout(() => {
      if (this.isClosed || this.isSpeaking) return;
      this.speakAndMaybeEnd('Ungal response kekkala. Sales team contact pannuvanga. Nandri.', true);
    }, this.silenceTimeoutMs);
  }

  endCall() {
    if (this.isClosed) return;
    console.log('[Agent] Ending call by closing Plivo stream.');
    if (this.plivoWs && this.plivoWs.readyState === WebSocket.OPEN) {
      this.plivoWs.close(1000, 'Conversation completed');
    }
  }

  markAgentIdle(logMessage) {
    console.log(logMessage);
    this.stopOutboundPlaybackLoop();
    this.isSpeaking = false;
    this.statusCallbacks.onStateChange('active');
    if (this.endCallAfterSpeech) {
      setTimeout(() => this.endCall(), 300);
    } else {
      this.scheduleSilenceTimeout();
    }
  }

  async speakAndMaybeEnd(reply, shouldEnd = false) {
    if (this.isClosed) return;
    this.endCallAfterSpeech = shouldEnd;
    this.isSpeaking = true;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = 0;
    this.history.push({ role: 'assistant', content: reply });
    this.statusCallbacks.onAiTranscript(reply, true);
    await this.synthesizeSpeech(reply);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Phase 3: LLM call with streaming TTS pipeline
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Process a completed user utterance:
   * 1. Streams LLM response token-by-token.
   * 2. At each sentence boundary, immediately sends the chunk to the TTS queue.
   * 3. TTS queue drains serially in parallel with LLM streaming.
   * Result: first audio heard by caller ~600-900ms after they stop speaking.
   */
  async handleUserUtterance(transcript) {
    if (this.isClosed) return;
    console.log(`[Agent] Processing utterance: "${transcript}"`);

    this.isSpeaking = true;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = 0;
    this.history.push({ role: 'user', content: transcript });
    this.statusCallbacks.onStateChange('active');
    this.clearSilenceTimer();

    // Phase 3: Reset streaming state for this turn
    this.llmStreamFinished = false;
    this.ttsStreamStartedAt = Date.now();
    this.ttsStreamTotalBytesSent = 0;
    this.ttsQueue = [];
    this.isTtsProcessing = false;

    // Timing: reset all per-turn timing checkpoints
    this.turnStartedAt = Date.now();
    this.llmFirstTokenAt = 0;
    this.firstAudioSentAt = 0;
    this.turnFirstAudioRecorded = false;

    // Phase 2: Use cached config — zero disk I/O
    const systemPrompt = this.cachedSystemPrompt;
    const knowledgeBase = this.cachedKnowledgeBase;

    const voiceCostControlRules = `
VOICE COST AND FLOW RULES:
- Follow the business, industry, flow, and tone from system_prompt.txt.
- Use knowledge_base.txt as the only source of business facts.
- Maximum 12 words per reply.
- Ask exactly one question at the end.
- Understand meaning from natural Tamil/Tanglish, not exact yes/no keywords.
- Detect positive, negative, continue, refusal, confusion, and question intent from context.
- Treat phrases like "இல்லைங்க", "நா பண்ணல", "illanga", "pannala" as no.
- Treat phrases like "ஆமாங்க", "சரி", "சொல்லுங்க", "okay", "pannirukken" as yes/continue when context fits.
- Never ask the caller to say only yes or no.
- Prefer 6-8 words. Never give long explanations unless caller asks.
- Do not invent prices, services, slots, policies, addresses, or offers.
- If the answer is not in the knowledge base, say the team will confirm.
- Never repeat the same intro if already said.`;

    const combinedSystemContext = `${systemPrompt}\n\n${voiceCostControlRules}\n\nKNOWLEDGE BASE CONTEXT:\n${knowledgeBase}`;

    // Keep last 12 history entries to limit context size and keep LLM fast
    const historyWindow = this.history.slice(-12);
    const messages = [
      { role: 'system', content: combinedSystemContext },
      ...historyWindow
    ];

    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) {
      console.error('[OpenAI] Missing API Key.');
      this.statusCallbacks.onError('OpenAI API Key is missing.');
      this.isSpeaking = false;
      return;
    }

    this.currentLlmController = new AbortController();
    let completeAiResponseText = '';
    let ttsBuffer = ''; // Phase 3: accumulate tokens until sentence boundary

    try {
      console.log(`[OpenAI] Querying ${this.isResponsesApi ? 'responses endpoint' : this.isAzureOpenAI ? 'Azure deployment' : 'model'}: ${this.openaiModelId}`);

      const headers = {
        'Content-Type': 'application/json',
        ...(this.isAzureOpenAI
          ? { 'api-key': openaiKey }
          : { 'Authorization': `Bearer ${openaiKey}` })
      };

      const body = this.isResponsesApi
        ? { input: messages, stream: true, max_output_tokens: 32, temperature: 0.35 }
        : { messages: messages, stream: true, max_tokens: 32, temperature: 0.35 };

      if (!this.isAzureOpenAI && !this.isResponsesApi) {
        body.model = this.openaiModelId;
      }

      const response = await fetch(this.openaiApiUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: this.currentLlmController.signal
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => '');
        throw new Error(`OpenAI HTTP error: ${response.status} ${errorBody}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Save incomplete line for next iteration

        for (const line of lines) {
          const cleanedLine = line.trim();
          if (cleanedLine === '' || cleanedLine === 'data: [DONE]') continue;
          if (cleanedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(cleanedLine.substring(6));
              const token = this.extractOpenAIStreamToken(data);
              if (token) {
                // Timing: capture timestamp of very first LLM token this turn
                if (!this.llmFirstTokenAt) {
                  this.llmFirstTokenAt = Date.now();
                }
                completeAiResponseText += token;
                ttsBuffer += token;
                this.statusCallbacks.onAiTranscript(token, false);

                // ── Phase 3: Fire TTS as soon as a speakable chunk is ready ──
                if (this.isSpeakableBoundary(ttsBuffer)) {
                  const chunk = ttsBuffer.trim();
                  ttsBuffer = '';
                  if (chunk && this.isSpeaking && !this.isClosed) {
                    this.enqueueTtsChunk(chunk);
                  }
                }
              }
            } catch (_) {
              // Ignore SSE chunk boundary parse errors
            }
          }
        }
      }

      // Ensure fallback text if LLM returned nothing
      if (completeAiResponseText.trim().length === 0) {
        completeAiResponseText = 'Sorry, sariyaa kekkala. Innum oru thadava sollunga.';
      }

      console.log(`[OpenAI] AI Complete Response: "${completeAiResponseText}"`);

      if (completeAiResponseText.trim().length > 0) {
        this.history.push({ role: 'assistant', content: completeAiResponseText });
        this.statusCallbacks.onAiTranscript('', true); // Signal stream complete to UI
      }

      // ── Phase 3: Flush any remaining buffer after LLM stream ends ──────────
      if (ttsBuffer.trim() && this.isSpeaking && !this.isClosed) {
        this.enqueueTtsChunk(ttsBuffer.trim());
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('[OpenAI] Stream aborted by user interruption.');
      } else {
        console.error('[OpenAI] Fetch error:', err);
        const fallbackText = 'Sorry, technical issue. Team contact pannuvanga.';
        this.history.push({ role: 'assistant', content: fallbackText });
        this.statusCallbacks.onAiTranscript(fallbackText, true);
        if (this.isSpeaking) await this.synthesizeSpeech(fallbackText);
      }
    } finally {
      this.currentLlmController = null;

      // ── Phase 3: Always mark LLM done, then kick the drain loop ───────────
      this.llmStreamFinished = true;

      // If drain is not running (edge case: TTS finished before LLM), trigger it
      // so it can detect llmStreamFinished and call markAgentIdle.
      if (!this.isTtsProcessing) {
        this.drainTtsQueue();
      }
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Outbound audio playback loop (kept for legacy ElevenLabs REST path)
  // ─────────────────────────────────────────────────────────────────────────

  startOutboundPlaybackLoop() {
    const bytesPerSecond = 8000;
    this.stopOutboundPlaybackLoop();

    if (!this.isSpeaking || this.isClosed || this.outboundAudioBuffer.length === 0) {
      return;
    }

    const audioToSend = this.outboundAudioBuffer;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = audioToSend.length;
    this.sendAudioToPlivo(audioToSend);

    const playbackMs = Math.ceil((audioToSend.length / bytesPerSecond) * 1000);
    this.playbackDoneTimeout = setTimeout(() => {
      const secondsSent = (this.outboundBytesSent / bytesPerSecond).toFixed(2);
      this.playbackDoneTimeout = null;
      this.markAgentIdle(`[Agent] Audio sent to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
    }, playbackMs + 250);
  }

  stopOutboundPlaybackLoop() {
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }
    if (this.playbackDoneTimeout) {
      clearTimeout(this.playbackDoneTimeout);
      this.playbackDoneTimeout = null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Plivo audio I/O
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Send one G.711 μ-law audio chunk to Plivo for playback.
   */
  sendAudioToPlivo(binaryChunk) {
    if (this.plivoWs && this.plivoWs.readyState === WebSocket.OPEN) {
      const base64Payload = binaryChunk.toString('base64');
      const msg = {
        event: 'playAudio',
        media: {
          contentType: 'audio/x-mulaw',
          sampleRate: 8000,
          payload: base64Payload
        }
      };
      this.plivoWs.send(JSON.stringify(msg));
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Barge-in & interruption
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Handle barge-in: immediately halt all AI speech and cancel in-flight LLM/TTS.
   */
  handleInterruption() {
    console.log('[Interruption] Barge-in! Stopping speech and clearing buffers.');
    this.clearSilenceTimer();

    // Mark as not speaking to stop all isSpeaking guards in drainTtsQueue / synthesize
    this.isSpeaking = false;
    this.stopOutboundPlaybackLoop();
    this.outboundAudioBuffer = Buffer.alloc(0);

    // Phase 3: Discard any pending TTS chunks
    this.ttsQueue = [];
    this.llmStreamFinished = false;

    // Phase 4: Terminate persistent TTS WebSocket to stop in-flight audio immediately.
    // The 'close' event will resolve any pending synthesizeChunkWithSarvam promise with 0
    // which allows drainTtsQueue's while-loop to exit cleanly.
    if (this.sarvamTtsWs) {
      try { this.sarvamTtsWs.terminate(); } catch (_) {}
      this.sarvamTtsWs = null;
      this.sarvamTtsWsConfigSent = false;
    }

    // Cancel in-flight LLM stream
    if (this.currentLlmController) {
      this.currentLlmController.abort();
      this.currentLlmController = null;
    }

    // Send clearAudio to Plivo to flush its speaker buffer
    if (this.plivoWs && this.plivoWs.readyState === WebSocket.OPEN && this.streamId) {
      const clearMessage = { event: 'clearAudio', streamId: this.streamId };
      this.plivoWs.send(JSON.stringify(clearMessage));
      console.log('[Interruption] Dispatched clearAudio to Plivo for stream:', this.streamId);
    }

    this.statusCallbacks.onInterruption();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Inbound audio from Plivo
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Feed raw inbound audio buffer (μ-law 8kHz) from Plivo into Sarvam STT.
   */
  handleInboundAudio(base64Payload) {
    if (this.isClosed) return;
    if (this.sarvamSttWs && this.sarvamSttWs.readyState === WebSocket.OPEN) {
      const mulawAudio = Buffer.from(base64Payload, 'base64');
      const pcmAudio = this.mulawBufferToPcm16Buffer(mulawAudio);
      this.sarvamSttWs.send(JSON.stringify({
        audio: {
          data: pcmAudio.toString('base64'),
          sample_rate: 8000,
          encoding: 'audio/wav'
        }
      }));
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle helpers
  // ─────────────────────────────────────────────────────────────────────────

  setStreamId(streamId) {
    this.streamId = streamId;
  }

  /**
   * Destructor — clean up all resources on call teardown.
   */
  close() {
    if (this.isClosed) return;
    console.log('[Agent] Tearing down active conversation resources...');
    this.isClosed = true;
    this.isSpeaking = false;
    this.ttsQueue = [];

    this.stopOutboundPlaybackLoop();
    this.clearSilenceTimer();

    if (this.currentLlmController) {
      this.currentLlmController.abort();
    }

    // Phase 4: Close persistent TTS WebSocket
    if (this.sarvamTtsWs) {
      try { this.sarvamTtsWs.close(); } catch (_) {}
      this.sarvamTtsWs = null;
      this.sarvamTtsWsConfigSent = false;
    }

    if (this.sarvamSttWs) {
      if (this.sarvamSttWs.readyState === WebSocket.CONNECTING) {
        this.sarvamSttWs.terminate();
      } else {
        this.sarvamSttWs.close();
      }
      this.sarvamSttWs = null;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Legacy PCM16 ↔ μ-law conversion helpers (kept for Cartesia TTS reference)
  // ─────────────────────────────────────────────────────────────────────────

  pcm16SampleToMulaw(sample) {
    const BIAS = 0x84;
    const CLIP = 32635;
    let sign = 0;
    let magnitude = sample;
    if (magnitude < 0) { magnitude = -magnitude; sign = 0x80; }
    magnitude = Math.min(CLIP, magnitude) + BIAS;
    let exponent = 7;
    for (let mask = 0x4000; (magnitude & mask) === 0 && exponent > 0; mask >>= 1) { exponent--; }
    const mantissa = (magnitude >> (exponent + 3)) & 0x0f;
    return (~(sign | (exponent << 4) | mantissa)) & 0xff;
  }

  pcm16BufferToMulawBuffer(pcmBuffer) {
    const sampleCount = Math.floor(pcmBuffer.length / 2);
    const mulawBuffer = Buffer.alloc(sampleCount);
    for (let i = 0; i < sampleCount; i++) {
      mulawBuffer[i] = this.pcm16SampleToMulaw(pcmBuffer.readInt16LE(i * 2));
    }
    return mulawBuffer;
  }

  /*
   * Previous Cartesia TTS implementation kept commented for rollback/reference.
   * synthesizeWithCartesia() was used before Sarvam TTS was adopted.
   */
}
