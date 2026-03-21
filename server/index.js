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
      jobs.set(videoFile, {
        id: videoFile,
        status: 'done',
        highlights: data.highlights || [],
      });
    } catch {
      // skip corrupt files
    }
  }
}

// Path to the ML script
const ML_SCRIPT = path.resolve(__dirname, '..', 'ml', 'highlight_detector.py');
const PYTHON_BIN = process.env.PYTHON_BIN || 'python3';

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
  jobs.set(filename, { id: filename, status: 'processing', highlights: [] });

  const args = [
    ML_SCRIPT,
    '--video', videoPath,
    '--format', 'json',
    '--output', resultPath,
  ];

  const child = spawn(PYTHON_BIN, args, { stdio: 'pipe' });

  let stderr = '';
  child.stderr.on('data', (data) => { stderr += data.toString(); });

  child.on('close', (code) => {
    if (code !== 0) {
      jobs.set(filename, {
        id: filename,
        status: 'error',
        error: stderr.trim() || `ML process exited with code ${code}`,
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

// List uploaded videos with processing status
app.get('/uploads', (req, res) => {
  const files = fs.readdirSync(uploadDir).filter((f) => f !== '.gitkeep' && !f.endsWith('.json'));
  const uploads = files.map((filename) => {
    const job = jobs.get(filename);
    return {
      filename,
      status: job ? job.status : 'unknown',
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

  if (range) {
    const parts = range.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
    const chunkSize = end - start + 1;

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunkSize,
      'Content-Type': 'video/mp4',
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': fileSize,
      'Content-Type': 'video/mp4',
    });
    fs.createReadStream(filePath).pipe(res);
  }
});

// Error-handling middleware
app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: 'Internal server error' });
});

if (require.main === module) {
  app.listen(port, () => {
    console.log(`Server läuft auf http://localhost:${port}`);
  });
}

module.exports = app;
