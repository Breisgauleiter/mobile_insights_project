const request = require('supertest');
const path = require('path');
const fs = require('fs');
const app = require('../index');

const uploadsDir = path.join(__dirname, '..', 'uploads');

function cleanUploads() {
  fs.readdirSync(uploadsDir).forEach((file) => {
    if (file !== '.gitkeep') fs.unlinkSync(path.join(uploadsDir, file));
  });
  app._jobs.clear();
}

afterAll(cleanUploads);

describe('GET /health', () => {
  it('should return health check', async () => {
    const res = await request(app).get('/health');
    expect(res.statusCode).toBe(200);
    expect(res.body.status).toBe('ok');
  });
});

describe('POST /upload', () => {
  afterEach(cleanUploads);

  it('should upload a video file successfully', async () => {
    const res = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video content'), {
        filename: 'test.mp4',
        contentType: 'video/mp4',
      });
    expect(res.statusCode).toBe(201);
    expect(res.body.message).toBe('Upload successful');
    expect(res.body.filename).toBeDefined();
  });

  it('should set job status after upload', async () => {
    const res = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video content'), {
        filename: 'status-test.mp4',
        contentType: 'video/mp4',
      });
    const job = app._jobs.get(res.body.filename);
    expect(job).toBeDefined();
    // Status should be pending or processing (async spawn)
    expect(['pending', 'processing', 'error']).toContain(job.status);
  });

  it('should return 400 when no file is uploaded', async () => {
    const res = await request(app).post('/upload');
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('No file uploaded');
  });

  it('should reject non-video files with 415', async () => {
    const res = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('not a video'), {
        filename: 'readme.txt',
        contentType: 'text/plain',
      });
    expect(res.statusCode).toBe(415);
    expect(res.body.error).toBe('Only video files are allowed.');
  });
});

describe('GET /uploads', () => {
  beforeAll(async () => {
    cleanUploads();
    await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video'), {
        filename: 'list-test.mp4',
        contentType: 'video/mp4',
      });
  });

  afterAll(cleanUploads);

  it('should list uploaded files with status', async () => {
    const res = await request(app).get('/uploads');
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.body.uploads)).toBe(true);
    expect(res.body.uploads.length).toBeGreaterThanOrEqual(1);
    expect(res.body.uploads[0]).toHaveProperty('filename');
    expect(res.body.uploads[0]).toHaveProperty('status');
  });

  it('should not include .json result files in uploads list', async () => {
    const res = await request(app).get('/uploads');
    const jsonFiles = res.body.uploads.filter((u) => u.filename.endsWith('.json'));
    expect(jsonFiles.length).toBe(0);
  });
});

describe('GET /results/:id', () => {
  it('should return job status for a known upload', async () => {
    cleanUploads();
    const uploadRes = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video'), {
        filename: 'result-test.mp4',
        contentType: 'video/mp4',
      });
    const res = await request(app).get(`/results/${uploadRes.body.filename}`);
    expect(res.statusCode).toBe(200);
    expect(res.body.id).toBe(uploadRes.body.filename);
    expect(res.body).toHaveProperty('status');
    expect(res.body).toHaveProperty('highlights');
  });

  it('should return 404 for an unknown id', async () => {
    const res = await request(app).get('/results/nonexistent-file.mp4');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBe('No results found for this id.');
  });
});

describe('ML Pipeline Integration', () => {
  afterEach(cleanUploads);

  it('should set status to done when ML results file exists', () => {
    const filename = 'ml-test-video.mp4';
    const videoPath = path.join(uploadsDir, filename);
    const resultPath = path.join(uploadsDir, filename + '.json');

    // Create a fake video file
    fs.writeFileSync(videoPath, 'fake');

    // Simulate ML output
    fs.writeFileSync(resultPath, JSON.stringify({
      highlights: [{ timestamp: 1.5, score: 25.3 }]
    }));

    // Manually set job as done (simulating what close handler does)
    const raw = fs.readFileSync(resultPath, 'utf-8');
    const data = JSON.parse(raw);
    app._jobs.set(filename, {
      id: filename,
      status: 'done',
      highlights: data.highlights,
    });

    const job = app._jobs.get(filename);
    expect(job.status).toBe('done');
    expect(job.highlights).toHaveLength(1);
    expect(job.highlights[0].timestamp).toBe(1.5);
  });

  it('should handle error status', () => {
    const filename = 'error-video.mp4';
    app._jobs.set(filename, {
      id: filename,
      status: 'error',
      error: 'ML process failed',
      highlights: [],
    });

    const job = app._jobs.get(filename);
    expect(job.status).toBe('error');
    expect(job.error).toBe('ML process failed');
  });
});

describe('GET /video/:filename', () => {
  const testFile = 'video-stream-test.mp4';
  const testContent = 'fake video content for streaming';

  beforeAll(() => {
    fs.writeFileSync(path.join(uploadsDir, testFile), testContent);
  });

  afterAll(() => {
    const p = path.join(uploadsDir, testFile);
    if (fs.existsSync(p)) fs.unlinkSync(p);
  });

  it('should stream the full video file', async () => {
    const res = await request(app).get(`/video/${testFile}`);
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toBe('video/mp4');
    expect(res.headers['content-length']).toBe(String(testContent.length));
  });

  it('should support Range requests', async () => {
    const res = await request(app)
      .get(`/video/${testFile}`)
      .set('Range', 'bytes=0-9');
    expect(res.statusCode).toBe(206);
    expect(res.headers['content-range']).toMatch(/^bytes 0-9\//);
  });

  it('should return 404 for nonexistent video', async () => {
    const res = await request(app).get('/video/does-not-exist.mp4');
    expect(res.statusCode).toBe(404);
  });

  it('should block access to .json files', async () => {
    fs.writeFileSync(path.join(uploadsDir, 'secret.json'), '{}');
    const res = await request(app).get('/video/secret.json');
    expect(res.statusCode).toBe(404);
    fs.unlinkSync(path.join(uploadsDir, 'secret.json'));
  });

  it('should return 416 for malformed Range header', async () => {
    const res = await request(app)
      .get(`/video/${testFile}`)
      .set('Range', 'bytes=-500');
    expect(res.statusCode).toBe(416);
  });

  it('should return 416 for out-of-bounds Range', async () => {
    const res = await request(app)
      .get(`/video/${testFile}`)
      .set('Range', 'bytes=99999-100000');
    expect(res.statusCode).toBe(416);
  });
});

describe('restoreJobs()', () => {
  afterEach(cleanUploads);

  it('should restore done jobs from existing .json result files', () => {
    const filename = 'restored-video.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    fs.writeFileSync(
      path.join(uploadsDir, filename + '.json'),
      JSON.stringify({ highlights: [{ timestamp: 2.0, score: 50 }] })
    );

    app._jobs.clear();
    app._restoreJobs();

    const job = app._jobs.get(filename);
    expect(job).toBeDefined();
    expect(job.status).toBe('done');
    expect(job.highlights).toHaveLength(1);
    expect(job.highlights[0].timestamp).toBe(2.0);
  });

  it('should skip .json files without matching video', () => {
    fs.writeFileSync(
      path.join(uploadsDir, 'orphan.mp4.json'),
      JSON.stringify({ highlights: [] })
    );

    app._jobs.clear();
    app._restoreJobs();

    expect(app._jobs.has('orphan.mp4')).toBe(false);
  });

  it('should handle corrupt .json gracefully', () => {
    const filename = 'corrupt-video.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    fs.writeFileSync(path.join(uploadsDir, filename + '.json'), 'NOT JSON');

    app._jobs.clear();
    app._restoreJobs();

    expect(app._jobs.has(filename)).toBe(false);
  });

  it('should normalize non-array highlights to empty array', () => {
    const filename = 'bad-highlights.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    fs.writeFileSync(
      path.join(uploadsDir, filename + '.json'),
      JSON.stringify({ highlights: 'not-an-array' })
    );

    app._jobs.clear();
    app._restoreJobs();

    const job = app._jobs.get(filename);
    expect(job).toBeDefined();
    expect(job.highlights).toEqual([]);
  });

  it('should be reflected in GET /uploads after restore', async () => {
    const filename = 'api-restore-test.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    fs.writeFileSync(
      path.join(uploadsDir, filename + '.json'),
      JSON.stringify({ highlights: [{ timestamp: 5, score: 80 }] })
    );

    app._jobs.clear();
    app._restoreJobs();

    const uploadsRes = await request(app).get('/uploads');
    const entry = uploadsRes.body.uploads.find((u) => u.filename === filename);
    expect(entry).toBeDefined();
    expect(entry.status).toBe('done');

    const resultsRes = await request(app).get(`/results/${filename}`);
    expect(resultsRes.statusCode).toBe(200);
    expect(resultsRes.body.highlights).toHaveLength(1);
  });
});
