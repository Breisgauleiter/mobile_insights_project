const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 3000;

// In-memory store for highlight results (keyed by filename)
const results = new Map();

// Create uploads directory if it doesn't exist
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

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

    // Store dummy highlight results for now
    results.set(req.file.filename, {
      id: req.file.filename,
      status: 'processed',
      highlights: [
        { timestamp: 1.23, label: 'Kill', confidence: 0.95 },
        { timestamp: 3.45, label: 'Objective', confidence: 0.87 },
      ],
    });

    return res.status(201).json({ message: 'Upload successful', filename: req.file.filename });
  });
});

// List uploaded videos
app.get('/uploads', (req, res) => {
  const files = fs.readdirSync(uploadDir).filter((f) => f !== '.gitkeep');
  const uploads = files.map((filename) => ({
    filename,
    hasResults: results.has(filename),
  }));
  res.json({ uploads });
});

// Get highlight results for a video
app.get('/results/:id', (req, res) => {
  const data = results.get(req.params.id);
  if (!data) {
    return res.status(404).json({ error: 'No results found for this id.' });
  }
  return res.json(data);
});

// Health check
app.get('/', (req, res) => {
  res.json({ message: 'Mobile Insights Server ist online' });
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
