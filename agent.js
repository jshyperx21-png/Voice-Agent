import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

// Helper to generate a random UUID
function generateUUID() {
  return crypto.randomUUID();
}

export class VoiceAgent {
  constructor(plivoWs, statusCallbacks) {
    this.plivoWs = plivoWs;
    this.statusCallbacks = statusCallbacks || {
      onStateChange: () => {},
      onUserTranscript: () => {},
      onAiTranscript: () => {},
      onInterruption: () => {},
      onError: () => {}
    };

    this.streamId = null;
    this.history = [];
    this.isSpeaking = false;
    this.cartesiaDone = false;
    
    // Outbound audio buffer (mu-law G.711, 8kHz raw bytes)
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.playbackInterval = null;
    this.playbackDoneTimeout = null;
    this.outboundBytesSent = 0;
    this.prebufferMs = Number(process.env.TTS_PREBUFFER_MS || 700);
    this.prebufferBytes = Math.round(8000 * (this.prebufferMs / 1000)); // 8 kHz mu-law
    this.ttsChunkMs = Number(process.env.TTS_CHUNK_MS || 100);

    // Active connection references
    this.deepgramWs = null;
    this.cartesiaWs = null;
    this.sarvamSttWs = null;
    this.currentLlmController = null;
    
    // Track current speech turn IDs
    this.cartesiaContextId = null;
    this.userUtteranceBuffer = '';

    // Reconnection & lifecycle state
    this.isClosed = false;
    this.deepgramReconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.greetingSpoken = false;
    this.sarvamSttFatalError = false;

    // Load voice config
    this.voiceId = process.env.CARTESIA_VOICE_ID || 'f786b574-daa5-4673-aa0c-cbe3e8534c02';
    this.cartesiaModelId = process.env.CARTESIA_MODEL_ID || 'sonic-3.5';
    this.cartesiaLanguage = process.env.CARTESIA_LANGUAGE || 'ta';
    this.cartesiaVersion = process.env.CARTESIA_VERSION || '2026-03-01';
    this.openaiModelId = process.env.OPENAI_MODEL || 'gpt-4.1-mini';
    this.deepgramLanguage = process.env.DEEPGRAM_LANGUAGE || 'ta';
    this.deepgramEndpointing = process.env.DEEPGRAM_ENDPOINTING || '350';
    this.enableBargeIn = process.env.ENABLE_BARGE_IN === 'true';
    this.sarvamApiKey = process.env.SARVAM_API_KEY;
    this.sarvamSttLanguage = process.env.SARVAM_STT_LANGUAGE || 'ta-IN';
    this.sarvamSttMode = process.env.SARVAM_STT_MODE || 'codemix';
    this.sarvamTtsLanguage = process.env.SARVAM_TTS_LANGUAGE || 'ta-IN';
    this.sarvamTtsModel = process.env.SARVAM_TTS_MODEL || 'bulbul:v3';
    this.sarvamTtsSpeaker = process.env.SARVAM_TTS_SPEAKER || 'ritu';
    this.sarvamTtsPace = Number(process.env.SARVAM_TTS_PACE || 1);
    this.sarvamTtsTemperature = Number(process.env.SARVAM_TTS_TEMPERATURE || 0.35);
    this.deepgramKeyterms = [
      'Shanmuga Hospital',
      'Facebook',
      'full body checkup',
      'package',
      'packages',
      'Silver',
      'Gold',
      'Platinum',
      'சில்வர்',
      'கோல்டு',
      'பிளாட்டினம்',
      'பிளாட்டினம் package',
      'கோல்டு package',
      'சில்வர் package',
      'appointment',
      'doctor',
      'insurance',
      'CT scan',
      'MRI',
      'ECG',
      'echo',
      'cardiology',
      'cancer screening',
      'blood test',
      'fasting',
      'vitamin D',
      'thyroid',
      'calcium'
    ];
  }

  /**
   * Start the AI connections
   */
  async start() {
    console.log('[Agent] Initializing AI streams...');
    this.connectSarvamStt();
  }

  /**
   * Play the initial greeting automatically when the call connects
   */
  speakGreeting() {
    if (this.isClosed) return;

    const greetingText = "வணக்கம், நான் Shanmuga Hospitalல இருந்து Karthika பேசுறேன்.";
    console.log(`[Agent] Speaking greeting: "${greetingText}"`);

    this.isSpeaking = true;
    this.cartesiaDone = false;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = 0;
    this.cartesiaContextId = generateUUID();

    // Add greeting to LLM conversation history so the model knows it was spoken
    this.history.push({ role: 'assistant', content: greetingText });

    // Stream transcript update to browser UI
    this.statusCallbacks.onAiTranscript(greetingText, true);
    this.statusCallbacks.onStateChange('active');

    this.synthesizeWithSarvam(greetingText);
  }

  /**
   * Connect to Sarvam Saaras streaming STT.
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
      vad_signals: 'true',
      high_vad_sensitivity: 'true',
      input_audio_codec: 'pcm_s16le'
    });

    const url = `wss://api.sarvam.ai/speech-to-text/ws?${params.toString()}`;
    console.log(`[Sarvam STT] Connecting. Language: ${this.sarvamSttLanguage}, Mode: ${this.sarvamSttMode}`);

    this.sarvamSttWs = new WebSocket(url, {
      headers: {
        'api-subscription-key': this.sarvamApiKey
      }
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

  /**
   * Convert Plivo's G.711 mu-law bytes into signed 16-bit PCM for Sarvam STT.
   */
  mulawByteToPcm16(muLawByte) {
    const BIAS = 0x84;
    let value = (~muLawByte) & 0xff;
    const sign = value & 0x80;
    const exponent = (value >> 4) & 0x07;
    const mantissa = value & 0x0f;
    let sample = ((mantissa << 3) + BIAS) << exponent;
    sample -= BIAS;
    return sign ? -sample : sample;
  }

  mulawBufferToPcm16Buffer(muLawBuffer) {
    const pcmBuffer = Buffer.alloc(muLawBuffer.length * 2);
    for (let i = 0; i < muLawBuffer.length; i++) {
      pcmBuffer.writeInt16LE(this.mulawByteToPcm16(muLawBuffer[i]), i * 2);
    }
    return pcmBuffer;
  }

  /**
   * Connect to Deepgram Streaming STT
   */
  connectDeepgram() {
    if (this.isClosed) return;

    const apiKey = process.env.DEEPGRAM_API_KEY;
    if (!apiKey) {
      console.error('[Deepgram] Missing API key in environment variables.');
      this.statusCallbacks.onError('Deepgram API Key is missing.');
      return;
    }

    const params = new URLSearchParams({
      model: 'nova-3',
      language: this.deepgramLanguage,
      encoding: 'mulaw',
      sample_rate: '8000',
      channels: '1',
      smart_format: 'true',
      interim_results: 'false',
      endpointing: this.deepgramEndpointing,
      vad_events: 'true'
    });

    for (const keyterm of this.deepgramKeyterms) {
      params.append('keyterm', keyterm);
    }

    const url = `wss://api.deepgram.com/v1/listen?${params.toString()}`;

    console.log(`[Deepgram] Connecting to streaming server. Language: ${this.deepgramLanguage}, Endpointing: ${this.deepgramEndpointing}ms`);
    this.deepgramWs = new WebSocket(url, {
      headers: {
        Authorization: `Token ${apiKey}`
      }
    });

    this.deepgramWs.on('open', () => {
      console.log('[Deepgram] WebSocket established successfully.');
      this.deepgramReconnectAttempts = 0;
    });

    this.deepgramWs.on('message', (data) => {
      try {
        const response = JSON.parse(data);
        const transcript = response.channel?.alternatives?.[0]?.transcript || '';
        const isFinal = response.is_final;
        const speechFinal = response.speech_final;

        if (transcript.trim().length > 0) {
          // Treat only final caller speech as a barge-in. Interim STT can contain
          // short echo/noise while outbound audio is still being played.
          if (this.enableBargeIn && this.isSpeaking && isFinal && this.isActionableUtterance(transcript)) {
            this.handleInterruption();
          }

          // Broadcast transcript updates to status listeners (e.g. browser dashboard)
          this.statusCallbacks.onUserTranscript(transcript, isFinal);

          if (isFinal) {
            console.log(`[Deepgram] Final Transcript: "${transcript}"`);
            this.userUtteranceBuffer += (this.userUtteranceBuffer ? ' ' : '') + transcript;
          }
        }

        // When the caller stops speaking (silence detected), trigger the AI turn
        if (speechFinal && this.userUtteranceBuffer.trim().length > 0) {
          const completeUtterance = this.normalizeTranscript(this.userUtteranceBuffer.trim());
          this.userUtteranceBuffer = '';
          if (!this.isActionableUtterance(completeUtterance)) {
            console.log(`[Deepgram] Ignoring short/noisy transcript: "${completeUtterance}"`);
            return;
          }
          this.handleUserUtterance(completeUtterance);
        }
      } catch (err) {
        console.error('[Deepgram] Error parsing incoming message:', err);
      }
    });

    this.deepgramWs.on('close', (code, reason) => {
      console.log(`[Deepgram] Connection closed. Code: ${code}, Reason: ${reason}`);
      if (!this.isClosed) {
        this.reconnectDeepgram();
      }
    });

    this.deepgramWs.on('error', (error) => {
      console.error('[Deepgram] WebSocket Error:', error);
    });
  }

  /**
   * Reconnect to Deepgram with exponential backoff
   */
  reconnectDeepgram() {
    if (this.deepgramReconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[Deepgram] Max reconnection attempts reached. Failing call.');
      this.statusCallbacks.onError('Failed to reconnect to Deepgram.');
      return;
    }

    this.deepgramReconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.deepgramReconnectAttempts), 10000);
    console.log(`[Deepgram] Reconnecting in ${delay}ms (Attempt ${this.deepgramReconnectAttempts}/${this.maxReconnectAttempts})...`);
    
    setTimeout(() => {
      this.connectDeepgram();
    }, delay);
  }

  /**
   * Connect to Cartesia WebSocket TTS API
   */
  connectCartesia() {
    if (this.isClosed) return;

    const apiKey = process.env.CARTESIA_API_KEY;
    if (!apiKey) {
      console.error('[Cartesia] Missing API key in environment variables.');
      this.statusCallbacks.onError('Cartesia API Key is missing.');
      return;
    }

    // Connect to the streaming API with the version that supports Sonic 3.5.
    const url = `wss://api.cartesia.ai/tts/websocket?cartesia_version=${this.cartesiaVersion}`;

    console.log(`[Cartesia] Connecting to streaming server. Model: ${this.cartesiaModelId}, Language: ${this.cartesiaLanguage}, Version: ${this.cartesiaVersion}`);
    this.cartesiaWs = new WebSocket(url, {
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });

    this.cartesiaWs.on('open', () => {
      console.log('[Cartesia] WebSocket established successfully.');
      if (!this.greetingSpoken) {
        this.greetingSpoken = true;
        setTimeout(() => this.speakGreeting(), 500);
      }
    });

    this.cartesiaWs.on('message', (data) => {
      try {
        const response = JSON.parse(data);

        if (response.type === 'error' || response.status_code >= 400) {
          console.error('[Cartesia] API error:', response);
          this.statusCallbacks.onError(response.error || response.message || 'Cartesia TTS error.');
          this.cartesiaDone = true;
          return;
        }

        // Ignore packets belonging to orphaned contexts (from interruptions)
        if (response.context_id !== this.cartesiaContextId) {
          return;
        }

        if ((response.type === 'chunk' || response.type === 'audio') && response.data) {
          // Push Cartesia's raw mulaw G.711 chunks to our outbound queue
          const audioChunk = Buffer.from(response.data, 'base64');
          this.outboundAudioBuffer = Buffer.concat([this.outboundAudioBuffer, audioChunk]);

          // Start after a short jitter buffer so Plivo receives smooth audio.
          if (!this.playbackInterval && this.isSpeaking && this.outboundAudioBuffer.length >= this.prebufferBytes) {
            this.startOutboundPlaybackLoop();
          }
        } else if (response.type === 'done' || response.done === true) {
          console.log('[Cartesia] Finished streaming audio for context:', response.context_id);
          this.cartesiaDone = true;
        }
      } catch (err) {
        console.error('[Cartesia] Error handling message:', err);
      }
    });

    this.cartesiaWs.on('close', (code, reason) => {
      console.log(`[Cartesia] Connection closed. Code: ${code}, Reason: ${reason}`);
      if (!this.isClosed) {
        console.log('[Cartesia] Attempting immediate reconnect...');
        setTimeout(() => this.connectCartesia(), 1000);
      }
    });

    this.cartesiaWs.on('error', (error) => {
      console.error('[Cartesia] WebSocket Error:', error);
    });
  }

  /**
   * Send text to Cartesia using the current WebSocket generation schema.
   */
  sendTextToCartesia(transcript, shouldContinue) {
    if (!this.cartesiaWs || this.cartesiaWs.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (!transcript || transcript.trim().length === 0) {
      return false;
    }

    this.cartesiaWs.send(JSON.stringify({
      context_id: this.cartesiaContextId,
      model_id: this.cartesiaModelId,
      transcript,
      language: this.cartesiaLanguage,
      voice: {
        mode: 'id',
        id: this.voiceId
      },
      output_format: {
        container: 'raw',
        encoding: 'pcm_mulaw',
        sample_rate: 8000
      },
      continue: shouldContinue
    }));

    return true;
  }

  flushTextToCartesia(text, shouldContinue) {
    const cleanText = text.replace(/\s+/g, ' ').trim();
    if (!cleanText) return false;
    return this.sendTextToCartesia(cleanText, shouldContinue);
  }

  /**
   * Sarvam REST TTS returns one complete 8kHz mu-law audio payload.
   * That matches Plivo directly and avoids transcoding-related choppiness.
   */
  async synthesizeWithSarvam(text) {
    const cleanText = text.replace(/\s+/g, ' ').trim();
    if (!cleanText || this.isClosed || !this.isSpeaking) return false;

    if (!this.sarvamApiKey) {
      console.error('[Sarvam TTS] Missing API key in environment variables.');
      this.statusCallbacks.onError('Sarvam API Key is missing.');
      this.isSpeaking = false;
      return false;
    }

    try {
      console.log(`[Sarvam TTS] Synthesizing with ${this.sarvamTtsModel}, speaker: ${this.sarvamTtsSpeaker}`);
      const response = await fetch('https://api.sarvam.ai/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-subscription-key': this.sarvamApiKey
        },
        body: JSON.stringify({
          text: cleanText,
          target_language_code: this.sarvamTtsLanguage,
          speaker: this.sarvamTtsSpeaker,
          model: this.sarvamTtsModel,
          pace: this.sarvamTtsPace,
          speech_sample_rate: 8000,
          output_audio_codec: 'mulaw',
          enable_preprocessing: true,
          temperature: this.sarvamTtsTemperature
        })
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => '');
        throw new Error(`Sarvam TTS HTTP error: ${response.status} ${errorBody}`);
      }

      const data = await response.json();
      const audioBase64 =
        data.audios?.[0] ||
        data.audio ||
        data.audio_base64 ||
        data.data?.audio ||
        data.data?.audios?.[0];

      if (!audioBase64) {
        throw new Error(`Sarvam TTS response did not include audio: ${JSON.stringify(data)}`);
      }

      this.outboundAudioBuffer = Buffer.concat([
        this.outboundAudioBuffer,
        Buffer.from(audioBase64, 'base64')
      ]);
      this.cartesiaDone = true;

      if (!this.playbackInterval && this.isSpeaking) {
        this.startOutboundPlaybackLoop();
      }

      return true;
    } catch (err) {
      console.error('[Sarvam TTS] Error:', err);
      this.statusCallbacks.onError('Sarvam TTS error.');
      this.cartesiaDone = true;
      this.isSpeaking = false;
      this.stopOutboundPlaybackLoop();
      return false;
    }
  }

  normalizeTranscript(text) {
    return text
      .replace(/பிராண்டீனம்|பிராண்டீனம்னு|பிராண்டினம்|பிளாடினம்|பிளாட்டினம்‌/gi, 'பிளாட்டினம்')
      .replace(/கோல்ட்|கோல்டு|கோல்ட்னு/gi, 'கோல்டு')
      .replace(/சில்வர்|சில்வர்னு/gi, 'சில்வர்')
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Drop tiny STT fragments without blocking common short caller intents.
   */
  isActionableUtterance(text) {
    const normalized = text.trim().toLowerCase();
    const words = normalized.split(/\s+/).filter(Boolean);
    const tamilCharCount = (normalized.match(/[\u0B80-\u0BFF]/g) || []).length;

    const yesNoTamilSounds = [
      'எஸ்',
      'யெஸ்',
      'எசு',
      'ஓகே',
      'ஹலோ',
      'ஆமா',
      'ஆம்',
      'சரி',
      'இல்லை',
      'வேணாம்'
    ];

    if (yesNoTamilSounds.includes(text.trim())) return true;
    if (words.length >= 3) return true;
    if (tamilCharCount >= 4) return true;
    if (tamilCharCount > 0 && words.length >= 2) return true;

    const allowedShortIntents = new Set([
      'hi',
      'hello',
      'yes',
      'yeah',
      'ok',
      'okay',
      'no',
      'sure',
      'gold',
      'silver',
      'platinum',
      'package',
      'packages',
      'appointment',
      'scan',
      'insurance',
      'doctor',
      'சரி',
      'ஓகே',
      'ஹலோ',
      'ஆமா',
      'ஆம்',
      'இல்லை',
      'வேண்டும்',
      'சொல்லுங்க',
      'வணக்கம்',
      'கோல்டு',
      'சில்வர்',
      'பிளாட்டினம்'
    ]);

    if (allowedShortIntents.has(normalized)) return true;

    return false;
  }

  getFastReply(transcript) {
    const normalized = transcript.trim().toLowerCase();
    const lastAssistant = [...this.history].reverse().find((item) => item.role === 'assistant')?.content || '';

    if (/(gold|கோல்டு)/i.test(normalized)) {
      return 'Gold packageல scan, vitamin tests, doctor consultation இருக்கு. Booking பண்ணலாமா?';
    }

    if (/(silver|சில்வர்)/i.test(normalized)) {
      return 'Silver basic full body checkup package. Booking பண்ணலாமா?';
    }

    if (/(platinum|பிளாட்டினம்)/i.test(normalized)) {
      return 'Platinum advanced checkup package. Booking பண்ணலாமா?';
    }

    if (/(yes|yeah|ok|okay|sure|எஸ்|யெஸ்|ஓகே|ஆமா|ஆம்|சரி|சரிங்க|ம் சொல்லுங்க|சொல்லுங்க)/i.test(normalized)) {
      if (/booking|book|Booking|name|பெயர்|உங்க name/i.test(lastAssistant)) {
        return 'சரி, உங்க name சொல்லுங்க.';
      }

      if (/package|packages|Packages|Silver|Gold|Platinum|பேக்கேஜ்/i.test(lastAssistant)) {
        return 'Silver, Gold, Platinum இருக்கு. எது பார்க்கணும்?';
      }

      return 'Packages பற்றி சொல்லட்டுமா?';
    }

    if (/(hi|hello|ஹலோ|வணக்கம்)/i.test(normalized) && this.history.length > 2) {
      return 'சொல்லுங்க, எந்த package பார்க்கணும்?';
    }

    return null;
  }

  /**
   * Process a completed user utterance: sends context + history to OpenAI, then sends the response to Sarvam TTS.
   */
  async handleUserUtterance(transcript) {
    if (this.isClosed) return;

    console.log(`[Agent] Processing utterance: "${transcript}"`);
    
    // Mark states
    this.isSpeaking = true;
    this.cartesiaDone = false;
    this.outboundAudioBuffer = Buffer.alloc(0); // clear existing buffer
    this.outboundBytesSent = 0;
    this.cartesiaContextId = generateUUID(); // fresh ID for this turn
    
    // Add user message to history
    this.history.push({ role: 'user', content: transcript });

    this.statusCallbacks.onStateChange('active');

    const fastReply = this.getFastReply(transcript);
    if (fastReply) {
      console.log(`[Agent] Fast reply: "${fastReply}"`);
      this.history.push({ role: 'assistant', content: fastReply });
      this.statusCallbacks.onAiTranscript(fastReply, true);
      await this.synthesizeWithSarvam(fastReply);
      return;
    }

    // Load dynamic system prompt and knowledge base from flat files
    let systemPrompt = 'You are Katie, a warm and helpful voice assistant.';
    let knowledgeBase = '';
    try {
      systemPrompt = fs.readFileSync(path.resolve('./system_prompt.txt'), 'utf8');
      knowledgeBase = fs.readFileSync(path.resolve('./knowledge_base.txt'), 'utf8');
    } catch (e) {
      console.warn('[Agent] Could not load system prompt or knowledge base files, using default settings.');
    }

    const voiceCostControlRules = `
VOICE COST AND FLOW RULES:
- Reply in Tamil/Tanglish like a Tamil Nadu hospital receptionist.
- Maximum 12 words per reply.
- Ask exactly one question at the end.
- Do not list all package details unless the caller asks.
- If caller asks unknown details, say sales team will confirm and continue booking.
- Never repeat the same intro if already said.`;

    const combinedSystemContext = `${systemPrompt}\n\n${voiceCostControlRules}\n\nKNOWLEDGE BASE CONTEXT:\n${knowledgeBase}`;

    // Compile chat messages
    // We only keep the last 12 history entries to maintain high speed and prevent context length bloat
    const historyWindow = this.history.slice(-12);
    const messages = [
      { role: 'system', content: combinedSystemContext },
      ...historyWindow
    ];

    // Call OpenAI API with SSE streaming using Node fetch
    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) {
      console.error('[OpenAI] Missing API Key.');
      this.statusCallbacks.onError('OpenAI API Key is missing.');
      this.isSpeaking = false;
      return;
    }

    this.currentLlmController = new AbortController();
    let completeAiResponseText = '';

    try {
      console.log(`[OpenAI] Querying model: ${this.openaiModelId}`);
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`
        },
        body: JSON.stringify({
          model: this.openaiModelId,
          messages: messages,
          stream: true,
          max_tokens: 40, // Keep TTS character usage and latency low.
          temperature: 0.45
        }),
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
        buffer = lines.pop(); // save incomplete line

        for (const line of lines) {
          const cleanedLine = line.trim();
          if (cleanedLine === '' || cleanedLine === 'data: [DONE]') continue;
          if (cleanedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(cleanedLine.substring(6));
              const token = data.choices?.[0]?.delta?.content || '';
              if (token) {
                completeAiResponseText += token;
                
                // Stream text update to browser
                this.statusCallbacks.onAiTranscript(token, false);
              }
            } catch (err) {
              // Ignore parse errors from SSE chunk boundary splits
            }
          }
        }
      }

      if (completeAiResponseText.trim().length === 0) {
        completeAiResponseText = 'சாரி, சரியா கேட்கல. இன்னொரு தடவை சொல்ல முடியுமா?';
      }

      console.log(`[OpenAI] AI Complete Response: "${completeAiResponseText}"`);
      
      // Add final assistant message to conversation history
      if (completeAiResponseText.trim().length > 0) {
        this.history.push({ role: 'assistant', content: completeAiResponseText });
        this.statusCallbacks.onAiTranscript('', true); // mark finished
      }

      if (this.isSpeaking) {
        await this.synthesizeWithSarvam(completeAiResponseText);
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('[OpenAI] Stream aborted by user interruption.');
      } else {
        console.error('[OpenAI] Fetch completed with error:', err);
        const fallbackText = 'சாரி, கொஞ்சம் technical issue இருக்கு. இன்னொரு தடவை சொல்ல முடியுமா?';
        this.history.push({ role: 'assistant', content: fallbackText });
        this.statusCallbacks.onAiTranscript(fallbackText, true);
        if (this.isSpeaking) await this.synthesizeWithSarvam(fallbackText);
      }
    } finally {
      this.currentLlmController = null;
    }
  }

  /**
   * Send one complete G.711 mu-law payload to Plivo and let Plivo queue playback.
   * This avoids Node timer jitter causing broken speech.
   */
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
      console.log(`[Agent] Audio sent to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
      this.playbackDoneTimeout = null;
      this.isSpeaking = false;
      this.statusCallbacks.onStateChange('active');
    }, playbackMs + 250);
  }

  /**
   * Stop outbound loop
   */
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

  /**
   * Sends binary audio chunk to Plivo Web Socket
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

  /**
   * Handle Barge-In Interruption: Halt AI speaking, reset queue, notify Plivo
   */
  handleInterruption() {
    console.log('[Interruption] Barge-in! Stopping speech and clearing buffers.');

    // 1. Mark states immediately to halt active streams
    this.isSpeaking = false;
    this.cartesiaDone = false;
    this.stopOutboundPlaybackLoop();
    this.outboundAudioBuffer = Buffer.alloc(0);

    // Orphan the current Cartesia context ID
    this.cartesiaContextId = null;

    // 2. Cancel active LLM completion
    if (this.currentLlmController) {
      this.currentLlmController.abort();
      this.currentLlmController = null;
    }

    // 3. Send clearAudio event to Plivo
    if (this.plivoWs && this.plivoWs.readyState === WebSocket.OPEN && this.streamId) {
      const clearMessage = {
        event: 'clearAudio',
        streamId: this.streamId
      };
      this.plivoWs.send(JSON.stringify(clearMessage));
      console.log('[Interruption] Dispatched clearAudio to Plivo for stream:', this.streamId);
    }

    // 4. Notify callbacks of interruption
    this.statusCallbacks.onInterruption();
  }

  /**
   * Feed raw inbound audio buffer (mu-law 8kHz) from Plivo into Sarvam STT.
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

  /**
   * Set Plivo Stream ID
   */
  setStreamId(streamId) {
    this.streamId = streamId;
  }

  /**
   * Destructor to clean up resources on call teardown
   */
  close() {
    if (this.isClosed) return;
    console.log('[Agent] Tearing down active conversation resources...');
    this.isClosed = true;
    this.isSpeaking = false;

    this.stopOutboundPlaybackLoop();

    if (this.currentLlmController) {
      this.currentLlmController.abort();
    }

    if (this.deepgramWs) {
      this.deepgramWs.close();
      this.deepgramWs = null;
    }

    if (this.cartesiaWs) {
      this.cartesiaWs.close();
      this.cartesiaWs = null;
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
}
