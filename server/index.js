const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 3000;

// In-memory store for job status and results (keyed by filename)
const jobs = new Map();

// Restore jobs from existing .json result files on startup
function restoreJobs() {
  if (!fs.existsSync(uploadDir)) return;
  const files = fs.readdirSync(uploadDir).filter((f) => f.endsWith('.json'));
  for (const jsonFile of files) {
    const videoFile = jsonFile.replace(/\.json$/, '');
    const videoPath = path.join(uploadDir, videoFile);
    if (!fs.existsSync(videoPath)) continue;
    try {
      const raw = fs.readFileSync(path.join(uploadDir, jsonFile), 'utf-8');
      const data = JSON.parse(raw);
      const highlights = Array.isArray(data.highlights) ? data.highlights : [];
      jobs.set(videoFile, {
        id: videoFile,
        status: 'done',
        highlights,
      });
    } catch {
      // skip corrupt files
    }
  }
}

// Path to the ML script
const ML_SCRIPT = path.resolve(__dirname, '..', 'ml', 'highlight_detector.py');
const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';

// ML performance tuning (set via env for local dev, 0 = full quality for prod)
function parseNonNegativeIntEnv(value, fallback) {
  const parsed = Number.parseInt(value ?? '', 10);
  if (Number.isNaN(parsed) || parsed < 0) return String(fallback);
  return String(parsed);
}
const ML_SKIP_FRAMES = parseNonNegativeIntEnv(process.env.ML_SKIP_FRAMES, 0);
const ML_MAX_WIDTH = parseNonNegativeIntEnv(process.env.ML_MAX_WIDTH, 0);
const WHISPER_MODEL = process.env.WHISPER_MODEL || 'tiny';
const ML_NO_AUDIO = process.env.ML_NO_AUDIO === '1';
const ML_NO_COLOR = process.env.ML_NO_COLOR === '1';

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

restoreJobs();

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + '-' + file.originalname);
  }
});

const fileFilter = function (req, file, cb) {
  if (file.mimetype.startsWith('video/')) {
    cb(null, true);
  } else {
    cb(new multer.MulterError('LIMIT_UNEXPECTED_FILE', 'video'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 500 * 1024 * 1024 } // 500 MB
});

app.use(express.json());

// Serve web UI
app.use(express.static(path.join(__dirname, 'public')));

// Upload endpoint with multer error handling
app.post('/upload', (req, res, next) => {
  upload.single('video')(req, res, (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(413).json({ error: 'File too large. Maximum size is 500 MB.' });
        }
        if (err.code === 'LIMIT_UNEXPECTED_FILE') {
          return res.status(415).json({ error: 'Only video files are allowed.' });
        }
        return res.status(400).json({ error: err.message });
      }
      return next(err);
    }
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filename = req.file.filename;
    const videoPath = path.join(uploadDir, filename);
    const resultPath = path.join(uploadDir, filename + '.json');

    // Set initial status
    jobs.set(filename, { id: filename, status: 'pending', highlights: [] });

    // Spawn ML pipeline asynchronously
    runMLPipeline(filename, videoPath, resultPath);

    return res.status(201).json({ message: 'Upload successful', filename });
  });
});

/**
 * Spawn the Python ML pipeline for a video file.
 * Updates the jobs Map with status and results.
 */
function runMLPipeline(filename, videoPath, resultPath) {
  jobs.set(filename, { id: filename, status: 'processing', highlights: [], progress: 0, stage: 'starting', startedAt: Date.now() });

  const args = [
    ML_SCRIPT,
    '--video', videoPath,
    '--format', 'json',
    '--output', resultPath,
    '--skip-frames', ML_SKIP_FRAMES,
    '--max-width', ML_MAX_WIDTH,
    '--whisper-model', WHISPER_MODEL,
  ];
  if (ML_NO_AUDIO) args.push('--no-audio');
  if (ML_NO_COLOR) args.push('--no-color');

  const child = spawn(PYTHON_BIN, args, { stdio: 'pipe' });

  let stderrBuf = '';
  child.stderr.on('data', (data) => {
    stderrBuf += data.toString();
    // Parse progress JSON lines from stderr
    const lines = stderrBuf.split('\n');
    stderrBuf = lines.pop(); // keep incomplete last line in buffer
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('{')) {
        try {
          const msg = JSON.parse(trimmed);
          if (typeof msg.progress === 'number') {
            const job = jobs.get(filename);
            if (job && job.status === 'processing') {
              job.progress = msg.progress;
              job.stage = msg.stage || '';
              // Calculate estimated remaining time
              if (msg.progress > 0 && job.startedAt) {
                const elapsed = (Date.now() - job.startedAt) / 1000;
                job.eta_seconds = Math.round(elapsed / msg.progress * (100 - msg.progress));
              } else {
                job.eta_seconds = null;
              }
            }
          }
        } catch {
          // not a progress line, ignore
        }
      }
    }
  });

  child.on('close', (code) => {
    if (code !== 0) {
      jobs.set(filename, {
        id: filename,
        status: 'error',
        error: stderrBuf.trim() || `ML process exited with code ${code}`,
        highlights: [],
      });
      return;
    }

    try {
      const raw = fs.readFileSync(resultPath, 'utf-8');
      const data = JSON.parse(raw);
      jobs.set(filename, {
        id: filename,
        status: 'done',
        highlights: data.highlights || [],
      });
    } catch (e) {
      jobs.set(filename, {
        id: filename,
        status: 'error',
        error: 'Failed to parse ML results',
        highlights: [],
      });
    }
  });

  child.on('error', (err) => {
    jobs.set(filename, {
      id: filename,
      status: 'error',
      error: err.message,
      highlights: [],
    });
  });
}

// Expose for testing
app._jobs = jobs;
app._runMLPipeline = runMLPipeline;
app._restoreJobs = restoreJobs;

// List uploaded videos with processing status
app.get('/uploads', (req, res) => {
  const files = fs.readdirSync(uploadDir).filter((f) => f !== '.gitkeep' && !f.endsWith('.json'));
  const uploads = files.map((filename) => {
    const job = jobs.get(filename);
    return {
      filename,
      status: job ? job.status : 'unknown',
      progress: job ? job.progress : undefined,
      stage: job ? job.stage : undefined,
      eta_seconds: job ? job.eta_seconds : undefined,
    };
  });
  res.json({ uploads });
});

// Get highlight results for a video
app.get('/results/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    return res.status(404).json({ error: 'No results found for this id.' });
  }
  return res.json(job);
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Stream video file with Range support for seeking
app.get('/video/:filename', (req, res) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(uploadDir, filename);

  if (!fs.existsSync(filePath) || filename.endsWith('.json')) {
    return res.status(404).json({ error: 'Video not found' });
  }

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;
  const range = req.headers.range;

  // Determine MIME type from file extension
  const extMap = { '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo', '.webm': 'video/webm', '.mkv': 'video/x-matroska' };
  const contentType = extMap[path.extname(filename).toLowerCase()] || 'application/octet-stream';

  if (range) {
    const rangeMatch = /^bytes=(\d*)-(\d*)$/.exec(range);

    if (!rangeMatch || !rangeMatch[1]) {
      return res.status(416).json({ message: 'Requested Range Not Satisfiable' });
    }

    const start = Number(rangeMatch[1]);
    let end = rangeMatch[2] ? Number(rangeMatch[2]) : fileSize - 1;

    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return res.status(416).json({ message: 'Requested Range Not Satisfiable' });
    }

    if (end > fileSize - 1) {
      end = fileSize - 1;
    }

    if (start < 0 || start > end || start >= fileSize) {
      return res.status(416).json({ message: 'Requested Range Not Satisfiable' });
    }

    const chunkSize = end - start + 1;

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': contentType,
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': contentType,
    });
    fs.createReadStream(filePath).pipe(res);
  }
});

/**
 * Returns true if the named job is currently processing.
 */
function isProcessing(filename) {
  const job = jobs.get(filename);
  return job ? job.status === 'processing' : false;
}

// Delete an uploaded video and its results
app.delete('/uploads/:filename', (req, res) => {
  const filename = path.basename(req.params.filename);
  const videoPath = path.join(uploadDir, filename);
  const resultPath = path.join(uploadDir, filename + '.json');

  if (isProcessing(filename)) {
    return res.status(409).json({ error: 'Job is currently processing' });
  }

  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: 'File not found' });
  }

  fs.unlinkSync(videoPath);

  if (fs.existsSync(resultPath)) {
    fs.unlinkSync(resultPath);
  }

  jobs.delete(filename);

  return res.status(200).json({ message: 'Deleted' });
});

// Reprocess an uploaded video through the ML pipeline
app.post('/reprocess/:filename', (req, res) => {
  const filename = path.basename(req.params.filename);
  const videoPath = path.join(uploadDir, filename);
  const resultPath = path.join(uploadDir, filename + '.json');

  if (isProcessing(filename)) {
    return res.status(409).json({ error: 'Job is currently processing' });
  }

  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: 'File not found' });
  }

  if (fs.existsSync(resultPath)) {
    fs.unlinkSync(resultPath);
  }

  runMLPipeline(filename, videoPath, resultPath);

  return res.status(202).json({ message: 'Reprocessing started' });
});

// Error-handling middleware
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

if (require.main === module) {
  const server = app.listen(port, () => {
    console.log(`Server läuft auf http://localhost:${port}`);
  });
  // Increase timeouts for large video uploads (10 min)
  server.requestTimeout = 600000;
  server.headersTimeout = 120000;
  server.timeout = 600000;
}

module.exports = app;
