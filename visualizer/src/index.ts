import { serve } from "bun";
import index from "./index.html";
import { readParquet } from 'parquet-wasm/node';
import { resolve } from 'path';

// ── Research investigation job management ─────────────────────────────────────

interface InvestigationJob {
  logs:        string[];
  status:      'running' | 'done' | 'error';
  stage:       'ingesting' | 'ready' | 'researching' | 'complete';
  voidId:      number;
  voidName:    string;
  subscribers: Set<(msg: string, type: string) => void>;
  sessionDir?: string;   // set once the research orchestrator announces it
}

// Mini jobs for single-paper codegen triggered from the UI
interface CodegenJob {
  logs:   string[];
  status: 'running' | 'done' | 'error';
  doi:    string;
}
const codegenJobs = new Map<string, CodegenJob>();

const activeJobs = new Map<string, InvestigationJob>();

function makeJobId(): string {
  return `job_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

// ── Shared subprocess pipe helper ─────────────────────────────────────────────
// Drains stdout + stderr fully THEN fires the exit callback so the last error
// lines are guaranteed to reach subscribers before the terminal event is sent.

function makePusher(job: InvestigationJob) {
  return (prefix: string, line: string) => {
    if (!line.trim()) return;
    const msg = `${prefix}${line}`;
    job.logs.push(msg);
    if (!job.sessionDir) {
      const m = line.match(/Session directory:\s*(.+)/);
      if (m) job.sessionDir = m[1].trim();
    }
    for (const sub of job.subscribers) {
      try { sub(msg, 'log'); } catch { job.subscribers.delete(sub); }
    }
  };
}

async function drainStream(
  stream: ReadableStream<Uint8Array>,
  prefix: string,
  push: (prefix: string, line: string) => void,
) {
  const dec = new TextDecoder();
  let buf = '';
  try {
    for await (const chunk of stream) {
      buf += dec.decode(chunk, { stream: true });
      const parts = buf.split('\n');
      buf = parts.pop() ?? '';
      for (const line of parts) push(prefix, line);
    }
  } catch { /* stream closed early */ }
  if (buf.trim()) push(prefix, buf);
}

function notify(
  job: InvestigationJob,
  msg: string,
  type: 'log' | 'done' | 'error',
) {
  job.logs.push(msg);
  for (const sub of job.subscribers) {
    try { sub(msg, type); } catch { job.subscribers.delete(sub); }
  }
  if (type !== 'log') job.subscribers.clear();
}

async function spawnInvestigation(
  jobId: string,
  voidId: number,
  voidName: string,
  papers: Array<{ doi: string; title: string }>,
): Promise<void> {
  const rootDir  = resolve(import.meta.dir, '../..');
  const gx10Url  = process.env.LOCAL_LLM_URL ?? 'http://100.123.34.54:11434';
  const scriptPy = resolve(rootDir, 'run_investigation.py');

  const job: InvestigationJob = {
    logs:        [],
    status:      'running',
    stage:       'ingesting',
    voidId,
    voidName,
    subscribers: new Set(),
  };
  activeJobs.set(jobId, job);

  const push = makePusher(job);

  const proc = Bun.spawn({
    cmd: [
      'python3', scriptPy,
      '--void-id',   String(voidId),
      '--void-name', voidName,
      '--papers',    JSON.stringify(papers),
      '--gx10-url',  gx10Url,
    ],
    stdout: 'pipe',
    stderr: 'pipe',
    cwd: rootDir,
    env: {
      ...process.env,
      LLM_BACKEND:        'gx10',
      LOCAL_LLM_URL:      gx10Url,
      LOCAL_LLM_MODEL:    'gemma4:latest',
      OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY ?? 'not_used',
      PYTHONUNBUFFERED:   '1',
    },
  });

  // Drain both streams and wait for process exit — in that order so all output
  // is captured before we fire the terminal event.
  const stdoutDone = drainStream(proc.stdout, '',       push);
  const stderrDone = drainStream(proc.stderr, '[ERR] ', push);

  ;(async () => {
    const code = await proc.exited;
    await Promise.allSettled([stdoutDone, stderrDone]); // flush remaining bytes

    // This is always the ingest process — advance to ready, keep SSE open
    job.stage  = 'ready';
    job.status = 'running';
    const msg = code === 0
      ? '=== INGEST COMPLETE — click Research to start the agent swarm ==='
      : `=== INGEST ERROR (exit ${code}) — see logs above for details ===`;
    notify(job, msg, 'log');          // keep subscribers alive for research phase
  })();
}

// ── SSE helper ────────────────────────────────────────────────────────────────

function makeSSEStream(job: InvestigationJob, voidName: string): ReadableStream {
  const enc = new TextEncoder();
  const sse  = (data: object) => enc.encode(`data: ${JSON.stringify(data)}\n\n`);

  return new ReadableStream({
    start(ctrl) {
      // Send job metadata
      ctrl.enqueue(sse({ type: 'info', voidName, voidId: job.voidId }));

      // Replay buffered logs
      for (const msg of job.logs) {
        ctrl.enqueue(sse({ type: 'log', message: msg }));
      }

      // Already finished?
      if (job.status !== 'running') {
        ctrl.enqueue(sse({ type: job.status }));
        try { ctrl.close(); } catch { }
        return;
      }

      // Subscribe to live updates
      const sub = (msg: string, type: string) => {
        try {
          ctrl.enqueue(sse(type === 'log' ? { type: 'log', message: msg } : { type }));
          if (type !== 'log') {
            job.subscribers.delete(sub);
            try { ctrl.close(); } catch { }
          }
        } catch {
          job.subscribers.delete(sub);
        }
      };
      job.subscribers.add(sub);
    },
    cancel() { /* client disconnected; subscriber cleaned up on next write */ },
  });
}

// ── HTTP server ───────────────────────────────────────────────────────────────

const server = serve({
  port: Number(process.env.PORT ?? 3000),
  routes: {

    "/public/umap_cv.parquet": async () => {
      const bytes = await Bun.file(resolve("public/umap_cv.parquet")).arrayBuffer();
      const table = readParquet(new Uint8Array(bytes));
      const arrowBytes = table.intoIPCStream();
      return new Response(arrowBytes, {
        headers: { "Content-Type": "application/vnd.apache.arrow.stream" },
      });
    },

    "/public/cluster_labels_cv.json": async () => {
      const file = Bun.file(resolve("public/cluster_labels_cv.json"));
      return new Response(file, { headers: { "Content-Type": "application/json" } });
    },

    "/public/voids_ranked_cv.json": async () => {
      const file = Bun.file(resolve("public/voids_ranked_cv.json"));
      return new Response(file, { headers: { "Content-Type": "application/json" } });
    },

    "/public/papers/:filename": async req => {
      const filename = decodeURIComponent(req.params.filename);
      if (filename.includes("/") || filename.includes("..") || !filename.endsWith(".pdf")) {
        return new Response("Invalid paper path", { status: 400 });
      }
      const file = Bun.file(resolve("public/papers", filename));
      if (!(await file.exists())) {
        return new Response("Paper not found", { status: 404 });
      }
      return new Response(file, { headers: { "Content-Type": "application/pdf" } });
    },

    // ── Investigation API ───────────────────────────────────────────────────

    "/api/investigate": {
      async POST(req) {
        let body: { voidId?: number; voidName?: string; papers?: unknown[] };
        try { body = await req.json(); }
        catch { return Response.json({ error: 'Invalid JSON' }, { status: 400 }); }

        const { voidId, voidName, papers } = body;
        if (voidId === undefined || !voidName || !Array.isArray(papers)) {
          return Response.json({ error: 'Missing voidId, voidName, or papers' }, { status: 400 });
        }

        const jobId = makeJobId();
        console.log(`[investigate] Starting job ${jobId} for void ${voidId}: "${voidName}"`);

        // Fire and don't await — investigation runs in background
        spawnInvestigation(
          jobId, voidId, voidName,
          papers as Array<{ doi: string; title: string }>,
        ).catch(err => console.error(`[investigate] spawn error:`, err));

        return Response.json({ jobId });
      },
    },

    "/api/investigate/:jobId/stream": async req => {
      const { jobId } = req.params;
      const job = activeJobs.get(jobId);
      if (!job) return new Response('Job not found', { status: 404 });

      const stream = makeSSEStream(job, job.voidName);
      return new Response(stream, {
        headers: {
          'Content-Type':  'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection':    'keep-alive',
        },
      });
    },

    // Job stage/status (polled by the UI to show the Research button)
    "/api/investigate/:jobId/status": async req => {
      const job = activeJobs.get(req.params.jobId);
      if (!job) return new Response('Not found', { status: 404 });
      return Response.json({ stage: job.stage, status: job.status });
    },

    // Start the research phase (called when user clicks "Research")
    "/api/investigate/:jobId/start-research": {
      async POST(req) {
        const job = activeJobs.get(req.params.jobId);
        if (!job) return Response.json({ error: 'job not found' }, { status: 404 });
        if (job.stage === 'researching') return Response.json({ error: 'already researching' }, { status: 409 });
        if (job.stage === 'ingesting')   return Response.json({ error: 'still ingesting' }, { status: 409 });

        job.stage  = 'researching';
        job.status = 'running';

        const rootDir = resolve(import.meta.dir, '../..');
        const push    = makePusher(job);

        // ── MNIST swarm: run shell script + tail the structured log file ──────
        if (job.voidName === 'Transformer-Augmented Vision Adaptation Gap') {
          const swarmDir = resolve(rootDir, 'research_problems/mnist_fcnn');
          const logFile  = resolve(swarmDir, 'logs/research_swarm_live.log');

          const proc = Bun.spawn({
            cmd: ['bash', 'run_research_swarm.sh'],
            stdout: 'pipe', stderr: 'pipe', cwd: swarmDir,
            env: { ...process.env, PYTHONUNBUFFERED: '1' },
          });

          const stdoutDone = drainStream(proc.stdout, '',     push);
          const stderrDone = drainStream(proc.stderr, '',     push);

          // Tail the structured log file (contains DEBUG TOKEN lines not on stderr)
          const stopTail = { stop: false };
          ;(async () => {
            let offset = -1; // -1 = wait for file to appear, then snapshot size
            while (!stopTail.stop) {
              try {
                const f = Bun.file(logFile);
                if (await f.exists()) {
                  const buf  = await f.arrayBuffer();
                  const text = new TextDecoder().decode(buf);
                  if (offset === -1) {
                    // First read after file appears: start from current end so we
                    // only stream content written by THIS run.
                    offset = text.length;
                  } else if (text.length > offset) {
                    const newText = text.slice(offset);
                    offset = text.length;
                    for (const line of newText.split('\n')) {
                      if (line.trim()) push('[SWARM] ', line);
                    }
                  }
                }
              } catch { /* file not yet writable */ }
              await Bun.sleep(400);
            }
          })();

          ;(async () => {
            const code = await proc.exited;
            await Promise.allSettled([stdoutDone, stderrDone]);
            stopTail.stop = true;

            // One final log-file drain to catch any lines buffered before exit
            await Bun.sleep(600);
            try {
              const f = Bun.file(logFile);
              if (await f.exists()) {
                const buf  = await f.arrayBuffer();
                const text = new TextDecoder().decode(buf);
                if (text.length > (offset ?? 0)) {
                  const newText = text.slice(offset ?? 0);
                  for (const line of newText.split('\n')) {
                    if (line.trim()) push('[SWARM] ', line);
                  }
                }
              }
            } catch { /* ignore */ }

            job.status = code === 0 ? 'done' : 'error';
            job.stage  = code === 0 ? 'complete' : 'ready';
            const final = code === 0
              ? '=== RESEARCH COMPLETE ==='
              : `=== RESEARCH ERROR (exit ${code}) — scroll up for details ===`;
            notify(job, final, job.status);
          })();

          return Response.json({ ok: true });
        }

        // ── Default path: generic research swarm python script ────────────────
        const gx10Url  = process.env.LOCAL_LLM_URL ?? 'http://100.123.34.54:11434';
        const scriptPy = resolve(rootDir, 'run_research_swarm.py');

        const proc = Bun.spawn({
          cmd: [
            'python3', scriptPy,
            '--void-id',   String(job.voidId),
            '--void-name', job.voidName,
            '--gx10-url',  gx10Url,
          ],
          stdout: 'pipe', stderr: 'pipe', cwd: rootDir,
          env: {
            ...process.env,
            LLM_BACKEND:        'gx10',
            LOCAL_LLM_URL:      gx10Url,
            LOCAL_LLM_MODEL:    'gemma4:latest',
            OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY ?? 'not_used',
            PYTHONUNBUFFERED:   '1',
          },
        });

        const stdoutDone = drainStream(proc.stdout, '',       push);
        const stderrDone = drainStream(proc.stderr, '[ERR] ', push);

        ;(async () => {
          const code = await proc.exited;
          await Promise.allSettled([stdoutDone, stderrDone]);

          job.status = code === 0 ? 'done' : 'error';
          job.stage  = code === 0 ? 'complete' : 'ready';
          const final = code === 0
            ? '=== RESEARCH COMPLETE ==='
            : `=== RESEARCH ERROR (exit ${code}) — scroll up for details ===`;
          notify(job, final, job.status);
        })();

        return Response.json({ ok: true });
      },
    },

    // Research events from the session's events.jsonl (polled by DeepResearchTab)
    "/api/investigate/:jobId/research-events": async req => {
      const job = activeJobs.get(req.params.jobId);
      if (!job) return new Response('Job not found', { status: 404 });
      if (!job.sessionDir) return Response.json([]);

      const eventsFile = Bun.file(`${job.sessionDir}/events.jsonl`);
      if (!(await eventsFile.exists())) return Response.json([]);

      const text = await eventsFile.text();
      const events = text.trim().split('\n').filter(Boolean).map(l => {
        try { return JSON.parse(l); } catch { return null; }
      }).filter(Boolean);
      return Response.json(events);
    },

    // Current metrics from the research problem's metrics file
    "/api/investigate/:jobId/latest-metrics": async req => {
      const job = activeJobs.get(req.params.jobId);
      if (!job) return new Response('Job not found', { status: 404 });
      if (!job.sessionDir) return Response.json(null);

      // Try imagenet problem dir first, then investigations dir
      const rootDir = resolve(import.meta.dir, '../..');
      const candidates = [
        `${rootDir}/research_problems/mnist_fcnn/logs/latest_metrics.json`,
        `${rootDir}/research_problems/imagenet_cnn/logs/latest_metrics.json`,
        `${rootDir}/outputs/investigations/void_${job.voidId}/logs/latest_metrics.json`,
      ];
      for (const p of candidates) {
        const f = Bun.file(p);
        if (await f.exists()) {
          return new Response(f, { headers: { 'Content-Type': 'application/json' } });
        }
      }
      return Response.json(null);
    },

    // Single-paper codegen triggered from the Deep Research tab
    "/api/paper2code": {
      async POST(req) {
        let body: { doi?: string; title?: string };
        try { body = await req.json(); } catch { return Response.json({ error: 'bad json' }, { status: 400 }); }
        const { doi, title } = body;
        if (!doi || !title) return Response.json({ error: 'doi and title required' }, { status: 400 });

        const jobId = makeJobId();
        const cgJob: CodegenJob = { logs: [], status: 'running', doi };
        codegenJobs.set(jobId, cgJob);

        const rootDir = resolve(import.meta.dir, '../..');
        const gx10Url = process.env.LOCAL_LLM_URL ?? 'http://100.123.34.54:11434';
        const proc = Bun.spawn({
          cmd: [
            'python3', resolve(rootDir, 'run_paper2code_single.py'),
            '--doi', doi, '--title', title, '--gx10-url', gx10Url,
          ],
          stdout: 'pipe', stderr: 'pipe', cwd: rootDir,
          env: { ...process.env, LLM_BACKEND: 'gx10', LOCAL_LLM_URL: gx10Url, LOCAL_LLM_MODEL: 'gemma4:latest', PYTHONUNBUFFERED: '1' },
        });

        (async () => {
          const dec = new TextDecoder(); let buf = '';
          for await (const chunk of proc.stdout) {
            buf += dec.decode(chunk, { stream: true });
            const parts = buf.split('\n'); buf = parts.pop() ?? '';
            for (const l of parts) if (l.trim()) cgJob.logs.push(l);
          }
          if (buf.trim()) cgJob.logs.push(buf);
        })().catch(() => {});
        (async () => {
          const dec = new TextDecoder(); let buf = '';
          for await (const chunk of proc.stderr) {
            buf += dec.decode(chunk, { stream: true });
            const parts = buf.split('\n'); buf = parts.pop() ?? '';
            for (const l of parts) if (l.trim()) cgJob.logs.push(`[ERR] ${l}`);
          }
        })().catch(() => {});
        proc.exited.then(code => { cgJob.status = code === 0 ? 'done' : 'error'; });

        return Response.json({ jobId });
      },
    },

    // Poll codegen job status
    "/api/paper2code/:jobId": async req => {
      const cj = codegenJobs.get(req.params.jobId);
      if (!cj) return new Response('Not found', { status: 404 });
      return Response.json({ status: cj.status, logs: cj.logs.slice(-20) });
    },

    "/api/investigate/jobs": async () => {
      const jobs = Array.from(activeJobs.entries()).map(([id, j]) => ({
        jobId:    id,
        voidId:   j.voidId,
        voidName: j.voidName,
        status:   j.status,
        logLines: j.logs.length,
      }));
      return Response.json(jobs);
    },

    // ── Fallthrough SPA ─────────────────────────────────────────────────────
    "/*": index,
  },

  development: process.env.NODE_ENV !== "production" && {
    hmr:     true,
    console: true,
  },
});

console.log(`🚀 Server running at ${server.url}`);
