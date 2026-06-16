import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

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
    
    // Outbound audio buffer (mu-law G.711, 8kHz raw bytes)
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.playbackInterval = null;
    this.playbackDoneTimeout = null;
    this.outboundBytesSent = 0;

    // Active connection references
    this.sarvamSttWs = null;
    this.currentLlmController = null;
    
    this.userUtteranceBuffer = '';
    this.silenceTimeout = null;
    this.silenceTimeoutMs = Number(process.env.SILENCE_TIMEOUT_MS || 15000);
    this.endCallAfterSpeech = false;

    // Reconnection & lifecycle state
    this.isClosed = false;
    this.greetingSpoken = false;
    this.sarvamSttFatalError = false;

    // Load voice config
    this.openaiModelId = process.env.OPENAI_MODEL || 'gpt-4.1';
    this.openaiApiUrl = process.env.OPENAI_API_URL || 'https://api.openai.com/v1/chat/completions';
    this.openaiProvider = process.env.OPENAI_PROVIDER || 'openai';
    this.isResponsesApi = /\/responses(?:\?|$)/.test(this.openaiApiUrl);
    this.isAzureOpenAI = this.openaiProvider.startsWith('azure') || this.openaiApiUrl.includes('.openai.azure.com') || this.openaiApiUrl.includes('.services.ai.azure.com');
    this.enableBargeIn = process.env.ENABLE_BARGE_IN === 'true';
    // Previous ElevenLabs TTS config kept for rollback/reference.
    // this.elevenLabsApiKey = process.env.ELEVENLABS_API_KEY;
    // this.elevenLabsVoiceId = process.env.ELEVENLABS_VOICE_ID || 'EXAVITQu4vr4xnSDxMaL';
    // this.elevenLabsModelId = process.env.ELEVENLABS_MODEL_ID || 'eleven_turbo_v2_5';
    // this.elevenLabsOutputFormat = process.env.ELEVENLABS_OUTPUT_FORMAT || 'ulaw_8000';
    // this.elevenLabsStreaming = process.env.ELEVENLABS_TTS_STREAMING !== 'false';
    // this.elevenLabsStability = Number(process.env.ELEVENLABS_STABILITY || 0.5);
    // this.elevenLabsSimilarityBoost = Number(process.env.ELEVENLABS_SIMILARITY_BOOST || 0.75);
    // Previous Cartesia TTS config kept for rollback/reference.
    // this.cartesiaApiKey = process.env.CARTESIA_API_KEY;
    // this.cartesiaModelId = process.env.CARTESIA_MODEL_ID || 'sonic-3.5';
    // this.cartesiaVoiceId = process.env.CARTESIA_VOICE_ID || '25d2c432-139c-4035-bfd6-9baaabcdd006';
    // this.cartesiaLanguage = process.env.CARTESIA_LANGUAGE || 'ta';
    // this.cartesiaVersion = process.env.CARTESIA_VERSION || '2026-03-01';
    // this.cartesiaSpeed = Number(process.env.CARTESIA_SPEED || 1);
    // this.cartesiaVolume = Number(process.env.CARTESIA_VOLUME || 1);
    // this.cartesiaEmotion = process.env.CARTESIA_EMOTION || 'calm';
    this.sarvamApiKey = process.env.SARVAM_API_KEY;
    this.sarvamSttLanguage = process.env.SARVAM_STT_LANGUAGE || 'ta-IN';
    this.sarvamSttMode = process.env.SARVAM_STT_MODE || 'codemix';
    this.sarvamTtsApiKey = process.env.SARVAM_TTS_API_KEY || this.sarvamApiKey;
    this.sarvamTtsModel = process.env.SARVAM_TTS_MODEL || 'bulbul:v3';
    this.sarvamTtsLanguage = process.env.SARVAM_TTS_LANGUAGE || 'ta-IN';
    this.sarvamTtsSpeaker = process.env.SARVAM_TTS_SPEAKER || 'anushka';
    this.sarvamTtsPace = Number(process.env.SARVAM_TTS_PACE || 1);
    this.sarvamTtsMinBufferSize = Number(process.env.SARVAM_TTS_MIN_BUFFER_SIZE || 50);
    this.sarvamTtsMaxChunkLength = Number(process.env.SARVAM_TTS_MAX_CHUNK_LENGTH || 200);
    this.sarvamTtsAudioCodec = process.env.SARVAM_TTS_AUDIO_CODEC || 'mulaw';
    this.sarvamTtsSampleRate = Number(process.env.SARVAM_TTS_SAMPLE_RATE || 8000);
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

    const greetingText = this.getGreetingText();
    console.log(`[Agent] Speaking greeting: "${greetingText}"`);

    this.isSpeaking = true;
    this.outboundAudioBuffer = Buffer.alloc(0);
    this.outboundBytesSent = 0;

    // Add greeting to LLM conversation history so the model knows it was spoken
    this.history.push({ role: 'assistant', content: greetingText });

    // Stream transcript update to browser UI
    this.statusCallbacks.onAiTranscript(greetingText, true);
    this.statusCallbacks.onStateChange('active');

    this.synthesizeSpeech(greetingText);
  }

  loadAgentFiles() {
    let systemPrompt = 'You are a warm and helpful voice assistant. Keep replies short and ask one question at a time.';
    let knowledgeBase = '';

    try {
      systemPrompt = fs.readFileSync(path.resolve('./system_prompt.txt'), 'utf8');
      knowledgeBase = fs.readFileSync(path.resolve('./knowledge_base.txt'), 'utf8');
    } catch (e) {
      console.warn('[Agent] Could not load system prompt or knowledge base files, using default settings.');
    }

    return { systemPrompt, knowledgeBase };
  }

  getGreetingText() {
    const { systemPrompt } = this.loadAgentFiles();
    const greetingLine = systemPrompt
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => /^GREETING\s*:/i.test(line));

    if (greetingLine) {
      return greetingLine.replace(/^GREETING\s*:/i, '').trim();
    }

    return 'Vanakkam, naan AI assistant pesuren. Eppadi help pannalam?';
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
      endpointing: '250',
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

  async synthesizeSpeech(text) {
    const cleanText = text.replace(/\s+/g, ' ').trim();
    if (!cleanText || this.isClosed || !this.isSpeaking) return false;
    const synthesized = await this.synthesizeWithSarvam(cleanText);
    if (!synthesized && this.isSpeaking && !this.isClosed) {
      this.markAgentIdle('[Sarvam TTS] Failed to synthesize audio. Agent idle.');
    }
    return synthesized;
  }

  async synthesizeWithSarvam(cleanText) {
    if (!this.sarvamTtsApiKey) {
      console.error('[Sarvam TTS] Missing API key in environment variables.');
      this.statusCallbacks.onError('Sarvam TTS API Key is missing.');
      return false;
    }

    const params = new URLSearchParams({
      model: this.sarvamTtsModel,
      send_completion_event: 'true'
    });
    const ws = new WebSocket(`wss://api.sarvam.ai/text-to-speech/ws?${params.toString()}`, {
      headers: {
        'api-subscription-key': this.sarvamTtsApiKey
      }
    });

    let totalBytesSent = 0;
    const startedAt = Date.now();

    console.log(`[Sarvam TTS] Streaming with ${this.sarvamTtsModel}, speaker: ${this.sarvamTtsSpeaker}, language: ${this.sarvamTtsLanguage}`);

    return new Promise((resolve) => {
      let settled = false;

      const finish = (ok) => {
        if (settled) return;
        settled = true;

        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }

        if (ok && totalBytesSent > 0 && this.isSpeaking && !this.isClosed) {
          const elapsedMs = Date.now() - startedAt;
          const playbackMs = Math.ceil((totalBytesSent / this.sarvamTtsSampleRate) * 1000);
          const remainingMs = Math.max(0, playbackMs - elapsedMs) + 250;

          setTimeout(() => {
            const secondsSent = (totalBytesSent / this.sarvamTtsSampleRate).toFixed(2);
            this.markAgentIdle(`[Sarvam TTS] Streamed audio to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
            resolve(true);
          }, remainingMs);
          return;
        }

        resolve(ok);
      };

      const timeout = setTimeout(() => {
        console.error('[Sarvam TTS] Timed out waiting for audio.');
        finish(false);
      }, 30000);

      ws.on('open', () => {
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

        ws.send(JSON.stringify({
          type: 'text',
          data: {
            text: cleanText
          }
        }));

        ws.send(JSON.stringify({
          type: 'flush'
        }));
      });

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());

          if (message.type === 'error' || message.error || message.data?.error) {
            console.error('[Sarvam TTS] API error:', message);
            clearTimeout(timeout);
            finish(false);
            return;
          }

          const audioBase64 = message.data?.audio || message.audio || message.data?.audio_chunk;
          if (audioBase64) {
            const audioChunk = Buffer.from(audioBase64, 'base64');
            if (audioChunk.length > 0 && this.isSpeaking && !this.isClosed) {
              totalBytesSent += audioChunk.length;
              this.sendAudioToPlivo(audioChunk);
            }
          }

          const eventType = message.data?.event_type || message.event_type || message.type;
          if (eventType === 'final' || eventType === 'completed' || eventType === 'done' || message.done === true) {
            clearTimeout(timeout);
            finish(totalBytesSent > 0);
          }
        } catch (err) {
          console.error('[Sarvam TTS] Error parsing WebSocket message:', err);
          clearTimeout(timeout);
          finish(false);
        }
      });

      ws.on('error', (err) => {
        console.error('[Sarvam TTS] WebSocket error:', err);
        clearTimeout(timeout);
        finish(false);
      });

      ws.on('close', () => {
        clearTimeout(timeout);
        if (!settled) finish(totalBytesSent > 0);
      });
    });
  }

  async synthesizeWithCartesia(cleanText) {
    if (!this.cartesiaApiKey) {
      console.error('[Cartesia TTS] Missing API key in environment variables.');
      this.statusCallbacks.onError('Cartesia API Key is missing.');
      return false;
    }

    const contextId = randomUUID();
    const ws = new WebSocket('wss://api.cartesia.ai/tts/websocket', {
      headers: {
        'Cartesia-Version': this.cartesiaVersion,
        'X-API-Key': this.cartesiaApiKey
      }
    });

    let totalBytesSent = 0;
    let pcmRemainder = Buffer.alloc(0);
    const startedAt = Date.now();

    console.log(`[Cartesia TTS] Streaming with ${this.cartesiaModelId}, voice: ${this.cartesiaVoiceId}, language: ${this.cartesiaLanguage}`);

    return new Promise((resolve) => {
      let settled = false;

      const finish = (ok) => {
        if (settled) return;
        settled = true;

        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }

        if (ok && totalBytesSent > 0 && this.isSpeaking && !this.isClosed) {
          const elapsedMs = Date.now() - startedAt;
          const playbackMs = Math.ceil((totalBytesSent / 8000) * 1000);
          const remainingMs = Math.max(0, playbackMs - elapsedMs) + 250;

          setTimeout(() => {
            const secondsSent = (totalBytesSent / 8000).toFixed(2);
            this.markAgentIdle(`[Cartesia TTS] Streamed audio to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
            resolve(true);
          }, remainingMs);
          return;
        }

        resolve(ok);
      };

      const timeout = setTimeout(() => {
        console.error('[Cartesia TTS] Timed out waiting for audio.');
        finish(false);
      }, 30000);

      ws.on('open', () => {
        ws.send(JSON.stringify({
          model_id: this.cartesiaModelId,
          transcript: cleanText,
          voice: {
            mode: 'id',
            id: this.cartesiaVoiceId
          },
          language: this.cartesiaLanguage,
          context_id: contextId,
          output_format: {
            container: 'raw',
            encoding: 'pcm_s16le',
            sample_rate: 8000
          },
          generation_config: {
            speed: this.cartesiaSpeed,
            volume: this.cartesiaVolume,
            emotion: this.cartesiaEmotion
          },
          continue: false
        }));
      });

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());

          if (message.type === 'error' || message.status_code >= 400) {
            console.error('[Cartesia TTS] API error:', message);
            clearTimeout(timeout);
            finish(false);
            return;
          }

          if (message.type === 'chunk' && message.data) {
            const pcmChunk = Buffer.from(message.data, 'base64');
            const combined = pcmRemainder.length > 0 ? Buffer.concat([pcmRemainder, pcmChunk]) : pcmChunk;
            const safeLength = combined.length - (combined.length % 2);
            const playablePcm = combined.subarray(0, safeLength);
            pcmRemainder = combined.subarray(safeLength);

            if (playablePcm.length > 0 && this.isSpeaking && !this.isClosed) {
              const mulawAudio = this.pcm16BufferToMulawBuffer(playablePcm);
              totalBytesSent += mulawAudio.length;
              this.sendAudioToPlivo(mulawAudio);
            }
          }

          if (message.type === 'done' || message.done === true) {
            clearTimeout(timeout);
            finish(totalBytesSent > 0);
          }
        } catch (err) {
          console.error('[Cartesia TTS] Error parsing WebSocket message:', err);
          clearTimeout(timeout);
          finish(false);
        }
      });

      ws.on('error', (err) => {
        console.error('[Cartesia TTS] WebSocket error:', err);
        clearTimeout(timeout);
        finish(false);
      });

      ws.on('close', () => {
        clearTimeout(timeout);
        if (!settled) finish(totalBytesSent > 0);
      });
    });
  }

  extractPcm16FromWav(wavBuffer) {
    if (wavBuffer.toString('ascii', 0, 4) !== 'RIFF' || wavBuffer.toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error('Invalid WAV response from Cartesia.');
    }

    let offset = 12;
    let format = null;
    let dataStart = -1;
    let dataSize = 0;

    while (offset + 8 <= wavBuffer.length) {
      const chunkId = wavBuffer.toString('ascii', offset, offset + 4);
      const chunkSize = wavBuffer.readUInt32LE(offset + 4);
      const chunkDataStart = offset + 8;
      const chunkDataEnd = Math.min(chunkDataStart + chunkSize, wavBuffer.length);

      if (chunkId === 'fmt ') {
        if (chunkDataEnd - chunkDataStart < 16) {
          throw new Error('Invalid WAV fmt chunk from Cartesia.');
        }
        format = {
          audioFormat: wavBuffer.readUInt16LE(chunkDataStart),
          channels: wavBuffer.readUInt16LE(chunkDataStart + 2),
          sampleRate: wavBuffer.readUInt32LE(chunkDataStart + 4),
          bitsPerSample: wavBuffer.readUInt16LE(chunkDataStart + 14)
        };
      } else if (chunkId === 'data') {
        dataStart = chunkDataStart;
        dataSize = chunkDataEnd - chunkDataStart;
        break;
      }

      offset = chunkDataStart + chunkSize + (chunkSize % 2);
    }

    if (!format || dataStart < 0) {
      throw new Error('WAV response missing fmt or data chunk.');
    }

    if (format.audioFormat !== 1 || format.bitsPerSample !== 16) {
      throw new Error(`Unsupported WAV format: format=${format.audioFormat}, bits=${format.bitsPerSample}`);
    }

    const bytesPerFrame = format.channels * 2;
    const safeDataSize = Math.floor(dataSize / bytesPerFrame) * bytesPerFrame;
    const sampleCount = Math.floor(safeDataSize / bytesPerFrame);
    const samples = new Int16Array(sampleCount);

    for (let i = 0; i < sampleCount; i++) {
      let mixed = 0;
      for (let ch = 0; ch < format.channels; ch++) {
        const sampleOffset = dataStart + ((i * format.channels + ch) * 2);
        mixed += wavBuffer.readInt16LE(sampleOffset);
      }
      samples[i] = Math.max(-32768, Math.min(32767, Math.round(mixed / format.channels)));
    }

    return {
      samples,
      sampleRate: format.sampleRate
    };
  }

  resamplePcm16(samples, sourceRate, targetRate) {
    if (sourceRate === targetRate) return samples;

    const targetLength = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
    const resampled = new Int16Array(targetLength);
    const ratio = sourceRate / targetRate;

    for (let i = 0; i < targetLength; i++) {
      const sourceIndex = i * ratio;
      const leftIndex = Math.floor(sourceIndex);
      const rightIndex = Math.min(samples.length - 1, leftIndex + 1);
      const fraction = sourceIndex - leftIndex;
      const left = samples[leftIndex] || 0;
      const right = samples[rightIndex] || 0;
      resampled[i] = Math.round(left + ((right - left) * fraction));
    }

    return resampled;
  }

  pcm16SampleToMulaw(sample) {
    const BIAS = 0x84;
    const CLIP = 32635;
    let sign = 0;
    let magnitude = sample;

    if (magnitude < 0) {
      magnitude = -magnitude;
      sign = 0x80;
    }

    magnitude = Math.min(CLIP, magnitude) + BIAS;
    let exponent = 7;
    for (let mask = 0x4000; (magnitude & mask) === 0 && exponent > 0; mask >>= 1) {
      exponent--;
    }

    const mantissa = (magnitude >> (exponent + 3)) & 0x0f;
    return (~(sign | (exponent << 4) | mantissa)) & 0xff;
  }

  pcm16SamplesToMulawBuffer(samples) {
    const mulawBuffer = Buffer.alloc(samples.length);
    for (let i = 0; i < samples.length; i++) {
      mulawBuffer[i] = this.pcm16SampleToMulaw(samples[i]);
    }
    return mulawBuffer;
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
   * Previous ElevenLabs TTS implementation kept commented for rollback/reference.
   *
  async synthesizeWithElevenLabs(cleanText) {
    if (!this.elevenLabsApiKey) {
      console.error('[ElevenLabs TTS] Missing API key in environment variables.');
      this.statusCallbacks.onError('ElevenLabs API Key is missing.');
      return false;
    }

    if (this.elevenLabsStreaming) {
      const streamed = await this.synthesizeWithElevenLabsStreaming(cleanText);
      if (streamed) return true;
    }

    return this.synthesizeWithElevenLabsRest(cleanText);
  }

  async synthesizeWithElevenLabsStreaming(cleanText) {
    try {
      console.log(`[ElevenLabs TTS] Streaming with ${this.elevenLabsModelId}, voice: ${this.elevenLabsVoiceId}`);
      const url = new URL(`https://api.elevenlabs.io/v1/text-to-speech/${this.elevenLabsVoiceId}/stream`);
      url.searchParams.set('output_format', this.elevenLabsOutputFormat);
      url.searchParams.set('optimize_streaming_latency', '3');

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': this.elevenLabsApiKey
        },
        body: JSON.stringify({
          text: cleanText,
          model_id: this.elevenLabsModelId,
          voice_settings: {
            stability: this.elevenLabsStability,
            similarity_boost: this.elevenLabsSimilarityBoost
          }
        })
      });

      if (!response.ok || !response.body) {
        const errorBody = await response.text().catch(() => '');
        throw new Error(`ElevenLabs streaming HTTP error: ${response.status} ${errorBody}`);
      }

      const reader = response.body.getReader();
      let totalBytesSent = 0;
      const firstAudioAt = Date.now();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value || value.length === 0) continue;
        if (this.isClosed || !this.isSpeaking) return false;

        const audioChunk = Buffer.from(value);
        totalBytesSent += audioChunk.length;
        this.sendAudioToPlivo(audioChunk);
      }

      if (totalBytesSent === 0) {
        throw new Error('ElevenLabs streaming returned no audio.');
      }

      const elapsedMs = Date.now() - firstAudioAt;
      const estimatedPlaybackMs = Math.ceil((totalBytesSent / 8000) * 1000);
      const remainingMs = Math.max(0, estimatedPlaybackMs - elapsedMs) + 250;

      await new Promise((resolve) => setTimeout(resolve, remainingMs));

      const secondsSent = (totalBytesSent / 8000).toFixed(2);
      this.markAgentIdle(`[ElevenLabs TTS] Streamed audio to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
      return true;
    } catch (err) {
      console.error('[ElevenLabs TTS] Streaming error:', err);
      return false;
    }
  }

  async synthesizeWithElevenLabsRest(cleanText) {
    try {
      console.log(`[ElevenLabs TTS] REST synthesizing with ${this.elevenLabsModelId}, voice: ${this.elevenLabsVoiceId}`);
      const url = new URL(`https://api.elevenlabs.io/v1/text-to-speech/${this.elevenLabsVoiceId}`);
      url.searchParams.set('output_format', this.elevenLabsOutputFormat);

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': this.elevenLabsApiKey
        },
        body: JSON.stringify({
          text: cleanText,
          model_id: this.elevenLabsModelId,
          voice_settings: {
            stability: this.elevenLabsStability,
            similarity_boost: this.elevenLabsSimilarityBoost
          }
        })
      });

      if (!response.ok) {
        const errorBody = await response.text().catch(() => '');
        throw new Error(`ElevenLabs REST HTTP error: ${response.status} ${errorBody}`);
      }

      const audioBuffer = Buffer.from(await response.arrayBuffer());
      if (audioBuffer.length === 0) {
        throw new Error('ElevenLabs REST returned no audio.');
      }

      this.outboundAudioBuffer = Buffer.concat([this.outboundAudioBuffer, audioBuffer]);

      if (!this.playbackInterval && this.isSpeaking) {
        this.startOutboundPlaybackLoop();
      }

      return true;
    } catch (err) {
      console.error('[ElevenLabs TTS] REST error:', err);
      return false;
    }
  }
  */

  normalizeTranscript(text) {
    return text
      .replace(/\s+/g, ' ')
      .trim();
  }

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

  /**
   * Drop tiny STT fragments without blocking common short caller intents.
   */
  isActionableUtterance(text) {
    const normalized = text.trim().toLowerCase();
    const words = normalized.split(/\s+/).filter(Boolean);
    const tamilCharCount = (normalized.match(/[\u0B80-\u0BFF]/g) || []).length;
    const shortTamilIntents = new Set([
      '\u0BAE\u0BCD', // ம்
      '\u0BAE\u0BCD\u0BAE\u0BCD', // ம்ம்
      '\u0B86', // ஆ
      '\u0B86\u0BAE\u0BCD', // ஆம்
      '\u0B86\u0BAE\u0BBE', // ஆமா
      '\u0B9A\u0BB0\u0BBF', // சரி
      '\u0B87\u0BB2\u0BCD\u0BB2\u0BC8', // இல்லை
      '\u0B93\u0B95\u0BC7' // ஓகே
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

  /**
   * Process a completed user utterance: sends context + history to OpenAI, then sends the response to TTS.
   */
  async handleUserUtterance(transcript) {
    if (this.isClosed) return;

    console.log(`[Agent] Processing utterance: "${transcript}"`);
    
    // Mark states
    this.isSpeaking = true;
    this.outboundAudioBuffer = Buffer.alloc(0); // clear existing buffer
    this.outboundBytesSent = 0;
    
    // Add user message to history
    this.history.push({ role: 'user', content: transcript });

    this.statusCallbacks.onStateChange('active');
    this.clearSilenceTimer();

    const { systemPrompt, knowledgeBase } = this.loadAgentFiles();

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
      console.log(`[OpenAI] Querying ${this.isResponsesApi ? 'responses endpoint' : this.isAzureOpenAI ? 'Azure deployment' : 'model'}: ${this.openaiModelId}`);
      const headers = {
        'Content-Type': 'application/json',
        ...(this.isAzureOpenAI ? { 'api-key': openaiKey } : { 'Authorization': `Bearer ${openaiKey}` })
      };
      const body = this.isResponsesApi
        ? {
          input: messages,
          stream: true,
          max_output_tokens: 32,
          temperature: 0.35
        }
        : {
          messages: messages,
          stream: true,
          max_tokens: 32,
          temperature: 0.35
        };

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
        buffer = lines.pop(); // save incomplete line

        for (const line of lines) {
          const cleanedLine = line.trim();
          if (cleanedLine === '' || cleanedLine === 'data: [DONE]') continue;
          if (cleanedLine.startsWith('data: ')) {
            try {
              const data = JSON.parse(cleanedLine.substring(6));
              const token = this.extractOpenAIStreamToken(data);
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
        completeAiResponseText = 'Sorry, sariyaa kekkala. Innum oru thadava sollunga.';
      }

      console.log(`[OpenAI] AI Complete Response: "${completeAiResponseText}"`);
      
      // Add final assistant message to conversation history
      if (completeAiResponseText.trim().length > 0) {
        this.history.push({ role: 'assistant', content: completeAiResponseText });
        this.statusCallbacks.onAiTranscript('', true); // mark finished
      }

      if (this.isSpeaking) {
        await this.synthesizeSpeech(completeAiResponseText);
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('[OpenAI] Stream aborted by user interruption.');
      } else {
        console.error('[OpenAI] Fetch completed with error:', err);
        const fallbackText = 'Sorry, technical issue. Team contact pannuvanga.';
        this.history.push({ role: 'assistant', content: fallbackText });
        this.statusCallbacks.onAiTranscript(fallbackText, true);
        if (this.isSpeaking) await this.synthesizeSpeech(fallbackText);
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
      this.playbackDoneTimeout = null;
      this.markAgentIdle(`[Agent] Audio sent to Plivo. Estimated playback ${secondsSent}s. Agent idle.`);
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
    this.clearSilenceTimer();

    // 1. Mark states immediately to halt active streams
    this.isSpeaking = false;
    this.stopOutboundPlaybackLoop();
    this.outboundAudioBuffer = Buffer.alloc(0);

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
    this.clearSilenceTimer();

    if (this.currentLlmController) {
      this.currentLlmController.abort();
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



