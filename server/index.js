const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { spawn } = require('child_process');
const archiver = require('archiver');
const rateLimit = require('express-rate-limit');

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
const TRACKER_SCRIPT = path.resolve(__dirname, '..', 'ml', 'object_tracker.py');
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

// Create clips cache directory
const clipsDir = path.join(uploadDir, '.clips');
if (!fs.existsSync(clipsDir)) {
  fs.mkdirSync(clipsDir);
}

// Create tracking results cache directory
const tracksDir = path.join(uploadDir, '.tracks');
if (!fs.existsSync(tracksDir)) {
  fs.mkdirSync(tracksDir);
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
app._clipsDir = clipsDir;
app._tracksDir = tracksDir;
app._deleteClipsForVideo = deleteClipsForVideo;
app._deleteTracksForVideo = deleteTracksForVideo;
app._getTrackCachePath = getTrackCachePath;

// Track an object in a video using OpenCV CSRT
app.post('/track', async (req, res) => {
  const { filename, time, bbox } = req.body;

  if (!filename || typeof filename !== 'string' || !filename.trim()) {
    return res.status(400).json({ error: 'filename is required' });
  }
  if (typeof time !== 'number' || !Number.isFinite(time) || time < 0) {
    return res.status(400).json({ error: 'time must be a non-negative number' });
  }
  if (
    !bbox ||
    typeof bbox.x !== 'number' ||
    typeof bbox.y !== 'number' ||
    typeof bbox.w !== 'number' ||
    typeof bbox.h !== 'number'
  ) {
    return res.status(400).json({ error: 'bbox must have numeric x, y, w, h' });
  }
  if (bbox.w <= 0 || bbox.h <= 0) {
    return res.status(400).json({ error: 'bbox w and h must be positive' });
  }

  const safeFilename = path.basename(filename);
  const videoPath = path.join(uploadDir, safeFilename);

  if (!fs.existsSync(videoPath) || safeFilename.endsWith('.json')) {
    return res.status(404).json({ error: 'Video not found' });
  }

  const cachePath = getTrackCachePath(safeFilename, time, bbox);

  if (fs.existsSync(cachePath)) {
    try {
      const cached = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
      return res.json({ positions: cached });
    } catch {
      // Corrupt cache — fall through to re-run
    }
  }

  const args = [
    TRACKER_SCRIPT,
    videoPath,
    String(time),
    String(Math.round(bbox.x)),
    String(Math.round(bbox.y)),
    String(Math.round(bbox.w)),
    String(Math.round(bbox.h)),
    '--duration', '30',
    '--fps', '5',
  ];

  return new Promise((resolve) => {
    const child = spawn(PYTHON_BIN, args, { stdio: 'pipe' });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });

    child.on('close', (code) => {
      if (code !== 0) {
        resolve(res.status(500).json({ error: `Tracker failed: ${stderr.trim()}` }));
        return;
      }
      try {
        const positions = JSON.parse(stdout.trim());
        fs.writeFileSync(cachePath, JSON.stringify(positions));
        resolve(res.json({ positions }));
      } catch {
        resolve(res.status(500).json({ error: 'Failed to parse tracker output' }));
      }
    });

    child.on('error', (err) => {
      resolve(res.status(500).json({ error: `Tracker spawn error: ${err.message}` }));
    });
  });
});

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

  deleteClipsForVideo(filename);
  deleteTracksForVideo(filename);
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

/**
 * Parse and clamp clip window parameters from query string.
 * Returns { beforeSec, afterSec } clamped to [0, 30].
 */
function parseClipParams(query) {
  const clamp = (val, min, max) => Math.min(max, Math.max(min, val));
  const before = clamp(parseFloat(query.before ?? '5'), 0, 30);
  const after = clamp(parseFloat(query.after ?? '5'), 0, 30);
  return {
    beforeSec: Number.isFinite(before) ? before : 5,
    afterSec: Number.isFinite(after) ? after : 5,
  };
}

// Rate limiter for clip extraction endpoints (ffmpeg is CPU-intensive)
const clipRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many clip requests, please try again later.' },
});

/**
 * Delete all cached clips for a given video filename.
 */
function deleteClipsForVideo(filename) {
  if (!fs.existsSync(clipsDir)) return;
  const prefix = filename + '_t';
  const entries = fs.readdirSync(clipsDir);
  for (const entry of entries) {
    if (entry.startsWith(prefix)) {
      try {
        fs.unlinkSync(path.join(clipsDir, entry));
      } catch {
        // ignore errors during cleanup
      }
    }
  }
}

/**
 * Extract a clip from a video using ffmpeg.
 * Returns a promise that resolves with the output clip path, or rejects on error.
 */
function extractClip(videoPath, outputPath, start, duration) {
  return new Promise((resolve, reject) => {
    const args = [
      '-y',
      '-ss', String(start),
      '-i', videoPath,
      '-t', String(duration),
      '-c:v', 'libx264',
      '-c:a', 'aac',
      '-movflags', '+faststart',
      outputPath,
    ];

    const ffmpeg = spawn('ffmpeg', args, { stdio: 'pipe' });
    let stderrBuf = '';
    ffmpeg.stderr.on('data', (d) => { stderrBuf += d.toString(); });

    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}: ${stderrBuf.trim()}`));
      } else {
        resolve(outputPath);
      }
    });

    ffmpeg.on('error', (err) => {
      reject(new Error(`ffmpeg spawn error: ${err.message}`));
    });
  });
}

/**
 * Delete all cached tracking results for a given video filename.
 */
function deleteTracksForVideo(filename) {
  if (!fs.existsSync(tracksDir)) return;
  const prefix = filename + '_';
  const entries = fs.readdirSync(tracksDir);
  for (const entry of entries) {
    if (entry.startsWith(prefix)) {
      try {
        fs.unlinkSync(path.join(tracksDir, entry));
      } catch {
        // ignore errors during cleanup
      }
    }
  }
}

/**
 * Compute the cache file path for a track request.
 */
function getTrackCachePath(filename, time, bbox) {
  const hash = crypto
    .createHash('md5')
    .update(`${time}:${bbox.x}:${bbox.y}:${bbox.w}:${bbox.h}`)
    .digest('hex');
  return path.join(tracksDir, `${filename}_${hash}.json`);
}

// Extract and serve a highlight clip
app.get('/clip/:filename', clipRateLimit, async (req, res) => {
  const filename = path.basename(req.params.filename);
  const videoPath = path.join(uploadDir, filename);

  if (!fs.existsSync(videoPath) || filename.endsWith('.json')) {
    return res.status(404).json({ error: 'Video not found' });
  }

  const tRaw = parseFloat(req.query.t);
  if (!Number.isFinite(tRaw) || tRaw < 0) {
    return res.status(400).json({ error: 'Query param "t" must be a non-negative number (timestamp in seconds).' });
  }

  const { beforeSec, afterSec } = parseClipParams(req.query);

  const start = Math.max(0, tRaw - beforeSec);
  const duration = beforeSec + afterSec;

  const cacheKey = `${filename}_t${tRaw}_b${beforeSec}_a${afterSec}.mp4`;
  const cachePath = path.join(clipsDir, cacheKey);

  if (fs.existsSync(cachePath)) {
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="${cacheKey}"`);
    return fs.createReadStream(cachePath).pipe(res);
  }

  try {
    await extractClip(videoPath, cachePath, start, duration);
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="${cacheKey}"`);
    return fs.createReadStream(cachePath).pipe(res);
  } catch (err) {
    // Remove partial output if any
    if (fs.existsSync(cachePath)) {
      try { fs.unlinkSync(cachePath); } catch { /* ignore */ }
    }
    return res.status(500).json({ error: 'Failed to extract clip: ' + err.message });
  }
});

// Export all highlight clips for a video as a zip archive
app.get('/clips/zip/:filename', clipRateLimit, async (req, res) => {
  const filename = path.basename(req.params.filename);
  const videoPath = path.join(uploadDir, filename);

  if (!fs.existsSync(videoPath) || filename.endsWith('.json')) {
    return res.status(404).json({ error: 'Video not found' });
  }

  const job = jobs.get(filename);
  if (!job || !Array.isArray(job.highlights) || job.highlights.length === 0) {
    return res.status(404).json({ error: 'No highlights found for this video.' });
  }

  const { beforeSec, afterSec } = parseClipParams(req.query);

  const zipName = filename.replace(/\.[^.]+$/, '') + '_highlights.zip';
  res.setHeader('Content-Type', 'application/zip');
  res.setHeader('Content-Disposition', `attachment; filename="${zipName}"`);

  const archive = archiver('zip', { zlib: { level: 0 } });
  archive.on('error', (err) => {
    // Headers may already be sent; just destroy the response
    res.destroy(err);
  });
  archive.pipe(res);

  const highlights = job.highlights;
  for (const h of highlights) {
    const ts = Number.isFinite(Number(h.timestamp)) ? Number(h.timestamp) : 0;
    const start = Math.max(0, ts - beforeSec);
    const duration = beforeSec + afterSec;
    const cacheKey = `${filename}_t${ts}_b${beforeSec}_a${afterSec}.mp4`;
    const cachePath = path.join(clipsDir, cacheKey);

    if (!fs.existsSync(cachePath)) {
      try {
        await extractClip(videoPath, cachePath, start, duration);
      } catch (err) {
        console.error(`Failed to extract clip for ${filename} at t=${ts}: ${err.message}`);
        // Skip clips that fail to extract
        continue;
      }
    }
    archive.file(cachePath, { name: cacheKey });
  }

  await archive.finalize();
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
