import 'dotenv/config';
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import fs from 'fs';
import path from 'path';
import { VoiceAgent } from './agent.js';

const PORT = process.env.PORT || 3000;
const app = express();

app.use(express.json());
app.use(express.static('public'));

// Global Call State tracking
let callState = {
  status: 'idle', // idle, ringing, active, ended
  duration: 0,
  outcome: 'No calls yet',
  plivoNumber: process.env.PLIVO_NUMBER || '+1234567890'
};

let activeAgent = null;
let callStartTime = null;
let callTimerInterval = null;

// Sets of connected WebSocket browser clients for dashboard real-time updates
const browserClients = new Set();

/**
 * Helper to broadcast JSON messages to all active browser dashboards
 */
function broadcastToBrowsers(data) {
  const payload = JSON.stringify(data);
  for (const client of browserClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

// -------------------------------------------------------------
// Express Routes
// -------------------------------------------------------------

/**
 * POST /webhook - Executed by Plivo when the caller dials our number.
 * Returns XML instruction opening a secure WebSocket audio stream to our server.
 */
app.post('/webhook', (req, res) => {
  console.log('[Plivo Webhook] Received call notification.');
  
  // Resolve host dynamically so ngrok or deployment URLs work out-of-the-box
  const host = req.headers.host;
  const protocol = req.secure || req.headers['x-forwarded-proto'] === 'https' ? 'wss' : 'ws';
  const wsUrl = `${protocol}://${host}/stream`;

  console.log(`[Plivo Webhook] Dynamic Stream Destination: ${wsUrl}`);

  // Return standard Plivo Voice Stream XML
  res.set('Content-Type', 'text/xml');
  res.send(`
    <Response>
      <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">${wsUrl}</Stream>
    </Response>
  `.trim());

  // Set status to ringing
  callState.status = 'ringing';
  callState.outcome = 'Call connecting...';
  broadcastToBrowsers({ event: 'status-update', state: callState });
});

/**
 * GET /api/config - Fetch prompt and knowledge base text
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
 * POST /api/config - Save system prompt and knowledge base updates
 */
app.post('/api/config', (req, res) => {
  try {
    const { system_prompt, knowledge_base } = req.body;
    
    fs.writeFileSync(path.resolve('./system_prompt.txt'), system_prompt || '', 'utf8');
    fs.writeFileSync(path.resolve('./knowledge_base.txt'), knowledge_base || '', 'utf8');
    
    console.log('[Config] Prompt and Knowledge Base updated successfully from UI.');
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save config files.' });
  }
});

// -------------------------------------------------------------
// WebSocket Servers Hook
// -------------------------------------------------------------
const server = createServer(app);

const wssStream = new WebSocketServer({ noServer: true });
const wssStatus = new WebSocketServer({ noServer: true });

// Handle WebSocket upgrades for separate routes
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

// -------------------------------------------------------------
// Dashboard WebSocket Handler (/status)
// -------------------------------------------------------------
wssStatus.on('connection', (ws) => {
  console.log('[Dashboard] New Web UI client connected.');
  browserClients.add(ws);

  // Send current state immediately on connection
  ws.send(JSON.stringify({ event: 'status-update', state: callState }));

  ws.on('close', () => {
    browserClients.delete(ws);
    console.log('[Dashboard] Web UI client disconnected.');
  });
});

// -------------------------------------------------------------
// Plivo Call Stream WebSocket Handler (/stream)
// -------------------------------------------------------------
wssStream.on('connection', (ws, req) => {
  console.log('[Plivo Stream] Caller connected over WebSocket.');

  // Clean up any lingering active agent instance
  if (activeAgent) {
    activeAgent.close();
    activeAgent = null;
  }

  // Setup heartbeat ping-pong interval to preserve active calls on gateways
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 10000);

  // Create the Voice Agent logic pipeline
  const agent = new VoiceAgent(ws, {
    onStateChange: (state) => {
      // E.g. when agent starts speaking or goes back to active listening
      if (callState.status !== 'ended') {
        broadcastToBrowsers({ event: 'agent-speaking', isSpeaking: agent.isSpeaking });
      }
    },
    onUserTranscript: (text, isFinal) => {
      broadcastToBrowsers({ event: 'user-transcript', text, isFinal });
    },
    onAiTranscript: (text, isFinal) => {
      broadcastToBrowsers({ event: 'ai-transcript', text, isFinal });
    },
    onInterruption: () => {
      broadcastToBrowsers({ event: 'interruption' });
    },
    onError: (errMsg) => {
      console.error('[Agent Error]', errMsg);
      broadcastToBrowsers({ event: 'error', message: errMsg });
    }
  });

  activeAgent = agent;
  callStartTime = Date.now();
  callState.status = 'ringing';
  callState.duration = 0;
  callState.outcome = 'Call established';
  broadcastToBrowsers({ event: 'status-update', state: callState });

  // Start call timer tracking
  if (callTimerInterval) clearInterval(callTimerInterval);
  callTimerInterval = setInterval(() => {
    if (callStartTime) {
      callState.duration = Math.round((Date.now() - callStartTime) / 1000);
    }
  }, 1000);

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data);

      switch (message.event) {
        case 'start':
          console.log('[Plivo Stream] Call Stream started. ID:', message.start?.streamId);
          agent.setStreamId(message.start?.streamId);
          
          callState.status = 'active';
          broadcastToBrowsers({ event: 'status-update', state: callState });

          // Establish deepgram and cartesia streams
          agent.start();
          break;

        case 'media':
          // Catch and assign stream ID from media packets if start packet was delayed
          if (!agent.streamId && message.streamId) {
            agent.setStreamId(message.streamId);
          }
          // Feed the inbound G.711 mu-law audio chunk directly to Deepgram
          if (message.media?.payload) {
            agent.handleInboundAudio(message.media.payload);
          }
          break;

        case 'stop':
          console.log('[Plivo Stream] Call stop event received.');
          handleCallTeardown('Ended normally');
          break;
      }
    } catch (err) {
      console.error('[Plivo Stream] Error processing incoming payload:', err);
    }
  });

  // Handle errors and normal disconnection teardowns
  ws.on('close', (code, reason) => {
    console.log(`[Plivo Stream] Telephony WebSocket closed. Code: ${code}, Reason: ${reason}`);
    handleCallTeardown('Call disconnected');
  });

  ws.on('error', (err) => {
    console.error('[Plivo Stream] Connection error:', err);
    handleCallTeardown('Error occurred');
  });

  function handleCallTeardown(outcomeText) {
    clearInterval(pingInterval);
    if (callTimerInterval) {
      clearInterval(callTimerInterval);
      callTimerInterval = null;
    }

    if (agent === activeAgent) {
      agent.close();
      activeAgent = null;
    }

    if (callState.status !== 'ended') {
      const callDuration = callStartTime ? Math.round((Date.now() - callStartTime) / 1000) : 0;
      callState.status = 'ended';
      callState.duration = callDuration;
      callState.outcome = `${outcomeText} (Duration: ${callDuration}s)`;

      broadcastToBrowsers({ event: 'status-update', state: callState });
      console.log(`[Call ended] ${callState.outcome}`);
    }
  }
});

// Start the server
server.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(` Voice Agent Backend running locally on port ${PORT}`);
  console.log(` Webhook URL: http://localhost:${PORT}/webhook`);
  console.log(` Dashboard URL: http://localhost:${PORT}/index.html`);
  console.log(`==================================================`);
});
