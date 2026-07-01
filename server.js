import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { VoiceAgent } from './agent.js';

const PORT = process.env.PORT || 3000;
const app = express();

app.use(express.json());
app.use(express.static('public'));

// ── Phase 5: Concurrent call support ─────────────────────────────────────────
// Map<agentId (UUID), VoiceAgent> — supports unlimited simultaneous calls.
// Previously: let activeAgent = null (single-call only).
const activeAgents = new Map();

// Dashboard UI clients
const browserClients = new Set();

// Call start timestamps for duration tracking (keyed by agentId)
const callStartTimes = new Map();

// Global call state for dashboard display (last call wins for single-dashboard simplicity)
let callState = {
  status: 'idle',
  duration: 0,
  outcome: 'No calls yet',
  plivoNumber: process.env.PLIVO_NUMBER || '+1234567890',
  activeCalls: 0
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Broadcast a JSON message to all connected browser dashboard clients.
 */
function broadcastToBrowsers(data) {
  const payload = JSON.stringify(data);
  for (const client of browserClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Express Routes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * POST /webhook — Executed by Plivo when a caller dials our number.
 * Returns XML instruction opening a secure WebSocket audio stream.
 */
app.post('/webhook', (req, res) => {
  console.log('[Plivo Webhook] Received call notification.');

  const host = req.headers.host;
  const protocol = req.secure || req.headers['x-forwarded-proto'] === 'https' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${host}/stream`;

  console.log(`[Plivo Webhook] Dynamic Stream Destination: ${wsUrl}`);

  res.set('Content-Type', 'text/xml');
  res.send(`
    <Response>
      <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">${wsUrl}</Stream>
    </Response>
  `.trim());

  callState.status = 'ringing';
  callState.outcome = 'Call connecting...';
  broadcastToBrowsers({ event: 'status-update', state: callState });
});

/**
 * GET /api/config — Fetch system prompt and knowledge base text.
 */
app.get('/api/config', (req, res) => {
  try {
    const promptPath = path.resolve('./system_prompt.txt');
    const kbPath = path.resolve('./knowledge_base.txt');
    const system_prompt = fs.existsSync(promptPath) ? fs.readFileSync(promptPath, 'utf8') : '';
    const knowledge_base = fs.existsSync(kbPath) ? fs.readFileSync(kbPath, 'utf8') : '';
    res.json({ system_prompt, knowledge_base });
  } catch (error) {
    res.status(500).json({ error: 'Failed to read config files.' });
  }
});

/**
 * POST /api/config — Save system prompt and knowledge base updates.
 * Phase 2: Also hot-reloads config on ALL active agents without restart.
 */
app.post('/api/config', (req, res) => {
  try {
    const { system_prompt, knowledge_base } = req.body;
    fs.writeFileSync(path.resolve('./system_prompt.txt'), system_prompt || '', 'utf8');
    fs.writeFileSync(path.resolve('./knowledge_base.txt'), knowledge_base || '', 'utf8');
    console.log('[Config] Prompt and Knowledge Base updated successfully from UI.');

    // Phase 2: Hot-reload config on every active call agent
    let reloaded = 0;
    for (const agent of activeAgents.values()) {
      agent.reloadConfig();
      reloaded++;
    }
    if (reloaded > 0) {
      console.log(`[Config] Hot-reloaded config on ${reloaded} active call agent(s).`);
    }

    res.json({ success: true, activeAgentsReloaded: reloaded });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save config files.' });
  }
});

/**
 * GET /health — Health check endpoint for monitoring and load balancers.
 * Phase 5: Returns live call count and server uptime.
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    activeCalls: activeAgents.size,
    uptimeSeconds: Math.round(process.uptime()),
    timestamp: new Date().toISOString()
  });
});

/**
 * GET /api/calls — List all active call agent IDs.
 * Phase 5: Useful for monitoring concurrent calls.
 */
app.get('/api/calls', (req, res) => {
  const calls = [];
  for (const [agentId, agent] of activeAgents.entries()) {
    const startTime = callStartTimes.get(agentId);
    calls.push({
      agentId,
      streamId: agent.streamId,
      durationSeconds: startTime ? Math.round((Date.now() - startTime) / 1000) : 0,
      isSpeaking: agent.isSpeaking
    });
  }
  res.json({ activeCalls: calls.length, calls });
});

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket Servers
// ─────────────────────────────────────────────────────────────────────────────

const server = createServer(app);
const wssStream = new WebSocketServer({ noServer: true });
const wssStatus = new WebSocketServer({ noServer: true });

server.on('upgrade', (request, socket, head) => {
  const { pathname } = new URL(request.url, `http://${request.headers.host}`);
  if (pathname === '/stream') {
    wssStream.handleUpgrade(request, socket, head, (ws) => {
      wssStream.emit('connection', ws, request);
    });
  } else if (pathname === '/status') {
    wssStatus.handleUpgrade(request, socket, head, (ws) => {
      wssStatus.emit('connection', ws, request);
    });
  } else {
    socket.destroy();
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard WebSocket (/status)
// ─────────────────────────────────────────────────────────────────────────────

wssStatus.on('connection', (ws) => {
  console.log('[Dashboard] New Web UI client connected.');
  browserClients.add(ws);

  // Send current state on connect
  ws.send(JSON.stringify({ event: 'status-update', state: callState }));

  ws.on('close', () => {
    browserClients.delete(ws);
    console.log('[Dashboard] Web UI client disconnected.');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Plivo Call Stream WebSocket (/stream)
// Phase 5: Each call gets a unique agentId and is stored in activeAgents Map
// ─────────────────────────────────────────────────────────────────────────────

wssStream.on('connection', (ws, req) => {
  // Phase 5: Assign a unique ID to this call
  const agentId = randomUUID();
  console.log(`[Plivo Stream] New caller connected. Agent ID: ${agentId}`);

  // Heartbeat ping to keep connection alive through gateways
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.ping();
  }, 10000);

  // Create the Voice Agent pipeline for this call
  const agent = new VoiceAgent(ws, {
    onStateChange: (state) => {
      if (callState.status !== 'ended') {
        broadcastToBrowsers({ event: 'agent-speaking', isSpeaking: agent.isSpeaking, agentId });
      }
    },
    onUserTranscript: (text, isFinal) => {
      broadcastToBrowsers({ event: 'user-transcript', text, isFinal, agentId });
    },
    onAiTranscript: (text, isFinal) => {
      broadcastToBrowsers({ event: 'ai-transcript', text, isFinal, agentId });
    },
    onInterruption: () => {
      broadcastToBrowsers({ event: 'interruption', agentId });
    },
    onAiTiming: ({ llmMs, ttsMs, totalMs }) => {
      // Broadcast exact response latency breakdown to dashboard
      broadcastToBrowsers({ event: 'ai-timing', llmMs, ttsMs, totalMs, agentId });
      console.log(`[Timing Broadcast] ⚡ Total: ${totalMs}ms | LLM: ${llmMs}ms | TTS: ${ttsMs}ms`);
    },
    onTtsAudio: (audioState) => {
      broadcastToBrowsers({ event: 'tts-audio', ...audioState, agentId });
    },
    onError: (errMsg) => {
      console.error(`[Agent Error] [${agentId}]`, errMsg);
      broadcastToBrowsers({ event: 'error', message: errMsg, agentId });
    }
  });

  // Phase 5: Register in concurrent agents Map
  activeAgents.set(agentId, agent);
  callStartTimes.set(agentId, Date.now());

  callState.status = 'ringing';
  callState.duration = 0;
  callState.outcome = 'Call established';
  callState.activeCalls = activeAgents.size;
  broadcastToBrowsers({ event: 'status-update', state: callState });

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data);

      switch (message.event) {
        case 'start':
          console.log(`[Plivo Stream] [${agentId}] Call stream started. Plivo StreamID: ${message.start?.streamId}`);
          agent.setStreamId(message.start?.streamId);

          callState.status = 'active';
          callState.activeCalls = activeAgents.size;
          broadcastToBrowsers({ event: 'status-update', state: callState });

          agent.start();
          break;

        case 'media':
          // Catch and assign streamId from media packets if start packet was delayed
          if (!agent.streamId && message.streamId) {
            agent.setStreamId(message.streamId);
          }
          if (message.media?.payload) {
            agent.handleInboundAudio(message.media.payload);
          }
          break;

        case 'stop':
          console.log(`[Plivo Stream] [${agentId}] Call stop event received.`);
          handleCallTeardown(agentId, 'Ended normally');
          break;
      }
    } catch (err) {
      console.error(`[Plivo Stream] [${agentId}] Error processing incoming payload:`, err);
    }
  });

  ws.on('close', (code, reason) => {
    console.log(`[Plivo Stream] [${agentId}] WebSocket closed. Code: ${code}, Reason: ${reason}`);
    handleCallTeardown(agentId, 'Call disconnected');
  });

  ws.on('error', (err) => {
    console.error(`[Plivo Stream] [${agentId}] Connection error:`, err);
    handleCallTeardown(agentId, 'Error occurred');
  });

  /**
   * Phase 5: Per-call teardown.
   * Removes from Map and broadcasts updated state.
   */
  function handleCallTeardown(id, outcomeText) {
    clearInterval(pingInterval);

    const agentToClose = activeAgents.get(id);
    if (agentToClose) {
      agentToClose.close();
      activeAgents.delete(id);
    }

    const startTime = callStartTimes.get(id);
    const callDuration = startTime ? Math.round((Date.now() - startTime) / 1000) : 0;
    callStartTimes.delete(id);

    callState.status = activeAgents.size > 0 ? 'active' : 'ended';
    callState.duration = callDuration;
    callState.outcome = `${outcomeText} (Duration: ${callDuration}s)`;
    callState.activeCalls = activeAgents.size;

    broadcastToBrowsers({ event: 'status-update', state: callState });
    console.log(`[Call ended] [${id}] ${callState.outcome}. Active calls remaining: ${activeAgents.size}`);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Start server
// ─────────────────────────────────────────────────────────────────────────────

server.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(` Voice Agent Backend running on port ${PORT}`);
  console.log(` Webhook URL: http://localhost:${PORT}/webhook`);
  console.log(` Dashboard URL: http://localhost:${PORT}/index.html`);
  console.log(` Health Check: http://localhost:${PORT}/health`);
  console.log(` Active Calls: http://localhost:${PORT}/api/calls`);
  console.log(`==================================================`);
});
