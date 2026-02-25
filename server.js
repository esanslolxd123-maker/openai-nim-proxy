// server.js - OpenAI to NVIDIA NIM API Proxy (hardened)

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();

// Render uses PORT, and often binds to 10000 internally.
// Your existing logs show port 10000, so we keep this.
const PORT = process.env.PORT || 10000;

// ===== Config =====
const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// Payload limits (fix 413). Adjust in Render env if needed.
const BODY_LIMIT = process.env.BODY_LIMIT || "50mb";

// Force streaming off if your client is flaky with SSE (JanitorAI/mobile).
// Set FORCE_NO_STREAM=true in Render env to force non-stream responses.
const FORCE_NO_STREAM = (process.env.FORCE_NO_STREAM || "false").toLowerCase() === "true";

// Reasoning / thinking toggles
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Axios timeouts (avoid hanging forever when response “takes 10 minutes”)
const AXIOS_TIMEOUT_MS = Number(process.env.AXIOS_TIMEOUT_MS || 600000); // 10 min default

// Model mapping: "OpenAI-ish name" -> "NIM model id"
const MODEL_MAPPING = {
  "gpt-3.5-turbo": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
  "gpt-4": "qwen/qwen3-coder-480b-a35b-instruct",
  "gpt-4-turbo": "moonshotai/kimi-k2-instruct-0905",
  "gpt-4o": "moonshotai/kimi-k2-instruct-0905",
  "claude-3-opus": "openai/gpt-oss-120b",
  "claude-3-sonnet": "openai/gpt-oss-20b",
  "gemini-pro": "qwen/qwen3-next-80b-a3b-thinking",
};

// ===== Middleware =====

// CORS (handles OPTIONS automatically)
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    exposedHeaders: ["*"],
  })
);

// Accept big JSON bodies (fix 413)
app.use(
  express.json({
    limit: BODY_LIMIT,
    type: ["application/json", "application/*+json"],
  })
);

// Some clients might send urlencoded; allow large too (safe)
app.use(express.urlencoded({ extended: true, limit: BODY_LIMIT }));

// ===== Helpers =====
function pickFallbackModel(requestedModel) {
  const modelLower = String(requestedModel || "").toLowerCase();
  if (modelLower.includes("405b") || modelLower.includes("gpt-4") || modelLower.includes("opus")) {
    return "meta/llama-3.1-405b-instruct";
  }
  if (modelLower.includes("70b") || modelLower.includes("sonnet") || modelLower.includes("gemini")) {
    return "meta/llama-3.1-70b-instruct";
  }
  return "meta/llama-3.1-8b-instruct";
}

function toBool(v) {
  return v === true || v === "true" || v === 1 || v === "1";
}

// ===== Routes =====

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    service: "OpenAI to NVIDIA NIM Proxy",
    body_limit: BODY_LIMIT,
    force_no_stream: FORCE_NO_STREAM,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_base: NIM_API_BASE,
    nim_api_key_present: Boolean(NIM_API_KEY),
  });
});

app.get("/v1/models", (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((id) => ({
    id,
    object: "model",
    created: Date.now(),
    owned_by: "nvidia-nim-proxy",
  }));

  res.json({ object: "list", data: models });
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: "Server misconfigured: NIM_API_KEY is missing (set it in Render environment variables).",
          type: "server_error",
          code: 500,
        },
      });
    }

    const body = req.body || {};
    const model = body.model;
    const messages = body.messages;

    if (!model || typeof model !== "string") {
      return res.status(400).json({
        error: { message: "Missing or invalid 'model' in request body", type: "invalid_request_error", code: 400 },
      });
    }

    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: "Missing or invalid 'messages' array in request body", type: "invalid_request_error", code: 400 },
      });
    }

    const temperature = Number.isFinite(+body.temperature) ? +body.temperature : 0.6;
    const max_tokens = Number.isFinite(+body.max_tokens) ? parseInt(body.max_tokens, 10) : 2048;

    // streaming: allow client to request stream, but optionally force it off
    let stream = toBool(body.stream);
    if (FORCE_NO_STREAM) stream = false;

    // Choose NIM model
    let nimModel = MODEL_MAPPING[model] || pickFallbackModel(model);

    // Build NIM request (avoid sending undefined fields)
    const nimRequest = {
      model: nimModel,
      messages,
      temperature,
      max_tokens,
      stream,
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    // Make request to NVIDIA NIM
    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        "Content-Type": "application/json",
      },
      responseType: stream ? "stream" : "json",
      timeout: AXIOS_TIMEOUT_MS,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      validateStatus: (status) => status >= 200 && status < 600, // let us forward errors cleanly
    });

    // Forward upstream error statuses as JSON when possible
    if (upstream.status >= 400) {
      // If it's JSON, forward it; if it's not, wrap it
      const data = upstream.data;
      if (typeof data === "object") {
        return res.status(upstream.status).json(data);
      }
      return res.status(upstream.status).json({
        error: {
          message: String(data || `Upstream error ${upstream.status}`),
          type: "upstream_error",
          code: upstream.status,
        },
      });
    }

    // STREAMING PATH
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");

      let buffer = "";
      let reasoningStarted = false;

      upstream.data.on("data", (chunk) => {
        buffer += chunk.toString("utf8");
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          if (line.includes("[DONE]")) {
            res.write("data: [DONE]\n\n");
            continue;
          }

          // Try parsing JSON events; if not parseable, pass through line
          try {
            const data = JSON.parse(line.slice(6));

            // Optional reasoning merge
            if (data?.choices?.[0]?.delta) {
              const delta = data.choices[0].delta;
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combined = "";
                if (reasoning && !reasoningStarted) {
                  combined = "<think>\n" + reasoning;
                  reasoningStarted = true;
                } else if (reasoning) {
                  combined = reasoning;
                }

                if (content && reasoningStarted) {
                  combined += "</think>\n\n" + content;
                  reasoningStarted = false;
                } else if (content) {
                  combined += content;
                }

                if (combined) delta.content = combined;
              }

              // Always delete reasoning_content so OpenAI-style clients don’t choke
              delete delta.reasoning_content;
              if (!delta.content) delta.content = "";
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + "\n");
          }
        }
      });

      upstream.data.on("end", () => res.end());
      upstream.data.on("error", (err) => {
        console.error("Stream error:", err?.message || err);
        res.end();
      });

      return;
    }

    // NON-STREAM PATH: Convert NIM response to OpenAI-style response
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model, // keep client-requested name
      choices: (upstream.data.choices || []).map((choice, idx) => {
        let fullContent = choice?.message?.content || "";

        if (SHOW_REASONING && choice?.message?.reasoning_content) {
          fullContent = `<think>\n${choice.message.reasoning_content}\n</think>\n\n` + fullContent;
        }

        return {
          index: Number.isFinite(choice.index) ? choice.index : idx,
          message: {
            role: choice?.message?.role || "assistant",
            content: fullContent,
          },
          finish_reason: choice.finish_reason || "stop",
        };
      }),
      usage: upstream.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error("Proxy error:", error?.message || error);

    if (error?.response) {
      console.error("Upstream status:", error.response.status);
      try {
        console.error("Upstream data:", JSON.stringify(error.response.data).slice(0, 2000));
      } catch {
        console.error("Upstream data: <non-json>");
      }
    } else {
      console.error("No upstream response (network/DNS/timeout)");
    }

    res.status(error?.response?.status || 500).json({
      error: {
        message: error?.message || "Internal server error",
        type: "invalid_request_error",
        code: error?.response?.status || 500,
      },
    });
  }
});

// Catch-all for unsupported endpoints
app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: "invalid_request_error",
      code: 404,
    },
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Body limit: ${BODY_LIMIT}`);
  console.log(`Force no stream: ${FORCE_NO_STREAM ? "ENABLED" : "DISABLED"}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? "ENABLED" : "DISABLED"}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? "ENABLED" : "DISABLED"}`);
  console.log(`NIM_API_BASE: ${NIM_API_BASE}`);
  console.log(`NIM_API_KEY present: ${Boolean(NIM_API_KEY)}`);
});
