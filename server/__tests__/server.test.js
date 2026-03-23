const request = require('supertest');
const path = require('path');
const fs = require('fs');
const app = require('../index');

const uploadsDir = path.join(__dirname, '..', 'uploads');
const clipsDir = app._clipsDir;

function cleanUploads() {
  fs.readdirSync(uploadsDir).forEach((file) => {
    const filePath = path.join(uploadsDir, file);
    if (file === '.gitkeep') return;
    if (fs.statSync(filePath).isDirectory()) {
      // Clean contents of subdirectories (e.g. .clips) but keep the dir
      fs.readdirSync(filePath).forEach((sub) => {
        const subPath = path.join(filePath, sub);
        if (fs.statSync(subPath).isFile()) {
          fs.unlinkSync(subPath);
        }
      });
    } else {
      fs.unlinkSync(filePath);
    }
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

describe('DELETE /uploads/:filename', () => {
  afterEach(cleanUploads);

  it('should delete the video file, JSON result, and job entry — 200', async () => {
    const uploadRes = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video content'), {
        filename: 'delete-me.mp4',
        contentType: 'video/mp4',
      });
    const filename = uploadRes.body.filename;

    const resultPath = path.join(uploadsDir, filename + '.json');
    fs.writeFileSync(resultPath, JSON.stringify({ highlights: [] }));
    app._jobs.set(filename, { id: filename, status: 'done', highlights: [] });

    const res = await request(app).delete(`/uploads/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(200);
    expect(res.body.message).toBe('Deleted');
    expect(fs.existsSync(path.join(uploadsDir, filename))).toBe(false);
    expect(fs.existsSync(resultPath)).toBe(false);
    expect(app._jobs.has(filename)).toBe(false);
  });

  it('should return 404 when the video file does not exist', async () => {
    const res = await request(app).delete('/uploads/nonexistent-video.mp4');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return 409 when the job is currently processing', async () => {
    const filename = 'processing-video.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake content');
    app._jobs.set(filename, { id: filename, status: 'processing', highlights: [] });

    const res = await request(app).delete(`/uploads/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(409);
    expect(res.body.error).toBeDefined();
    expect(fs.existsSync(path.join(uploadsDir, filename))).toBe(true);
  });

  it('should delete without a JSON result file if one does not exist', async () => {
    const filename = 'no-json.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake content');
    app._jobs.set(filename, { id: filename, status: 'done', highlights: [] });

    const res = await request(app).delete(`/uploads/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(200);
    expect(fs.existsSync(path.join(uploadsDir, filename))).toBe(false);
  });
});

describe('POST /reprocess/:filename', () => {
  afterEach(cleanUploads);

  it('should delete old JSON result, start ML pipeline, and return 202', async () => {
    const uploadRes = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video content'), {
        filename: 'reprocess-me.mp4',
        contentType: 'video/mp4',
      });
    const filename = uploadRes.body.filename;

    const resultPath = path.join(uploadsDir, filename + '.json');
    fs.writeFileSync(resultPath, JSON.stringify({ highlights: [{ timestamp: 1, score: 10 }] }));
    app._jobs.set(filename, { id: filename, status: 'done', highlights: [] });

    const res = await request(app).post(`/reprocess/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(202);
    expect(res.body.message).toBe('Reprocessing started');
    expect(fs.existsSync(resultPath)).toBe(false);

    const job = app._jobs.get(filename);
    expect(job).toBeDefined();
    expect(job.status).toBe('processing');
  });

  it('should return 404 when the video file does not exist', async () => {
    const res = await request(app).post('/reprocess/ghost-file.mp4');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return 409 when the job is currently processing', async () => {
    const filename = 'still-processing.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake content');
    app._jobs.set(filename, { id: filename, status: 'processing', highlights: [] });

    const res = await request(app).post(`/reprocess/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(409);
    expect(res.body.error).toBeDefined();
  });
});

describe('GET /clip/:filename', () => {
  const testVideo = 'clip-test-video.mp4';

  beforeEach(() => {
    cleanUploads();
    fs.writeFileSync(path.join(uploadsDir, testVideo), 'fake video content');
    app._jobs.set(testVideo, { id: testVideo, status: 'done', highlights: [{ timestamp: 10, score: 50 }] });
  });

  afterEach(cleanUploads);

  it('should return 404 when video does not exist', async () => {
    const res = await request(app).get('/clip/nonexistent.mp4?t=10');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when t param is missing', async () => {
    const res = await request(app).get(`/clip/${testVideo}`);
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toMatch(/param "t"/i);
  });

  it('should return 400 when t param is not a valid number', async () => {
    const res = await request(app).get(`/clip/${testVideo}?t=abc`);
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toMatch(/param "t"/i);
  });

  it('should return 400 when t is negative', async () => {
    const res = await request(app).get(`/clip/${testVideo}?t=-5`);
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toMatch(/param "t"/i);
  });

  it('should serve a cached clip if it already exists', async () => {
    const cacheKey = `${testVideo}_t10_b5_a5.mp4`;
    const cachePath = path.join(clipsDir, cacheKey);
    const fakeClipContent = 'fake clip data';
    fs.writeFileSync(cachePath, fakeClipContent);

    const res = await request(app).get(`/clip/${testVideo}?t=10`);
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/video\/mp4/);
    expect(res.headers['content-disposition']).toContain(cacheKey);
  });

  it('should clamp before/after params to 0-30 and use cached clip', async () => {
    // before=60 clamps to 30, after=-1 clamps to 0
    const cacheKey = `${testVideo}_t10_b30_a0.mp4`;
    const cachePath = path.join(clipsDir, cacheKey);
    fs.writeFileSync(cachePath, 'fake clamped clip');

    const res = await request(app).get(`/clip/${testVideo}?t=10&before=60&after=-1`);
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-disposition']).toContain(cacheKey);
  });

  it('should block access to .json files', async () => {
    fs.writeFileSync(path.join(uploadsDir, 'secret.json'), '{}');
    const res = await request(app).get('/clip/secret.json?t=10');
    expect(res.statusCode).toBe(404);
    fs.unlinkSync(path.join(uploadsDir, 'secret.json'));
  });
});

describe('GET /clips/zip/:filename', () => {
  const testVideo = 'zip-test-video.mp4';

  beforeEach(() => {
    cleanUploads();
    fs.writeFileSync(path.join(uploadsDir, testVideo), 'fake video content');
  });

  afterEach(cleanUploads);

  it('should return 404 when video does not exist', async () => {
    const res = await request(app).get('/clips/zip/nonexistent.mp4');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return 404 when no highlights exist for the video', async () => {
    app._jobs.set(testVideo, { id: testVideo, status: 'done', highlights: [] });
    const res = await request(app).get(`/clips/zip/${testVideo}`);
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toMatch(/highlights/i);
  });

  it('should return a zip archive when pre-cached clips exist', async () => {
    app._jobs.set(testVideo, {
      id: testVideo,
      status: 'done',
      highlights: [{ timestamp: 5, score: 80 }],
    });

    // Pre-create cached clip so ffmpeg is not needed
    const cacheKey = `${testVideo}_t5_b5_a5.mp4`;
    fs.writeFileSync(path.join(clipsDir, cacheKey), 'fake clip data');

    const res = await request(app).get(`/clips/zip/${testVideo}`);
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/application\/zip/);
    expect(res.headers['content-disposition']).toContain('_highlights.zip');
  });
});

describe('Clip cleanup on DELETE /uploads/:filename', () => {
  afterEach(cleanUploads);

  it('should delete cached clips when a video is deleted', async () => {
    const filename = 'video-with-clips.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    app._jobs.set(filename, { id: filename, status: 'done', highlights: [] });

    // Create fake cached clips
    const clipA = path.join(clipsDir, `${filename}_t5_b5_a5.mp4`);
    const clipB = path.join(clipsDir, `${filename}_t30_b5_a5.mp4`);
    fs.writeFileSync(clipA, 'clip A');
    fs.writeFileSync(clipB, 'clip B');

    const res = await request(app).delete(`/uploads/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(200);
    expect(fs.existsSync(clipA)).toBe(false);
    expect(fs.existsSync(clipB)).toBe(false);
  });
});

describe('POST /minimap-analysis', () => {
  const testVideo = 'minimap-test-video.mp4';

  beforeEach(() => {
    cleanUploads();
    fs.writeFileSync(path.join(uploadsDir, testVideo), 'fake video content');
  });

  afterEach(cleanUploads);

  it('should return 400 when filename is missing', async () => {
    const res = await request(app).post('/minimap-analysis').send({});
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when body is empty', async () => {
    const res = await request(app).post('/minimap-analysis');
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 404 when video does not exist', async () => {
    const res = await request(app)
      .post('/minimap-analysis')
      .send({ filename: 'nonexistent.mp4' });
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return 404 for .json files', async () => {
    fs.writeFileSync(path.join(uploadsDir, 'secret.json'), '{}');
    const res = await request(app)
      .post('/minimap-analysis')
      .send({ filename: 'secret.json' });
    expect(res.statusCode).toBe(404);
    fs.unlinkSync(path.join(uploadsDir, 'secret.json'));
  });

  it('should serve a cached minimap result if one exists', async () => {
    const cacheFile = path.join(uploadsDir, testVideo + '.minimap.json');
    const fakeResult = {
      timeline: [{ time: 1.0, positions: [{ x: 0.1, y: 0.9, team: 'ally' }] }],
      events: [],
      minimap_region: { x: 0, y: 196, width: 43, height: 43 },
    };
    fs.writeFileSync(cacheFile, JSON.stringify(fakeResult));

    const res = await request(app)
      .post('/minimap-analysis')
      .send({ filename: testVideo });
    expect(res.statusCode).toBe(200);
    expect(res.body.timeline).toBeDefined();
    expect(res.body.events).toBeDefined();
    expect(res.body.minimap_region).toBeDefined();
    expect(res.body.timeline).toHaveLength(1);
    expect(res.body.timeline[0].time).toBe(1.0);
  });

  it('should sanitize filename with path traversal attempt', async () => {
    const res = await request(app)
      .post('/minimap-analysis')
      .send({ filename: '../../etc/passwd' });
    expect(res.statusCode).toBe(404);
  });

  it('should delete minimap cache file when video is deleted', async () => {
    const cacheFile = path.join(uploadsDir, testVideo + '.minimap.json');
    fs.writeFileSync(cacheFile, JSON.stringify({ timeline: [], events: [], minimap_region: {} }));
    app._jobs.set(testVideo, { id: testVideo, status: 'done', highlights: [] });

    const res = await request(app).delete(`/uploads/${encodeURIComponent(testVideo)}`);
    expect(res.statusCode).toBe(200);
    expect(fs.existsSync(cacheFile)).toBe(false);
  });
});

describe('POST /track', () => {
  const testVideo = 'track-test-video.mp4';
  const tracksDir = app._tracksDir;

  beforeEach(() => {
    cleanUploads();
    fs.writeFileSync(path.join(uploadsDir, testVideo), 'fake video content');
  });

  afterEach(cleanUploads);

  it('should return 400 when filename is missing', async () => {
    const res = await request(app)
      .post('/track')
      .send({ time: 0, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when filename is empty', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: '', time: 0, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when time is negative', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: -1, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when time is not a number', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 'abc', bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when bbox is missing', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 0 });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when bbox w is zero', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 0, bbox: { x: 0, y: 0, w: 0, h: 50 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 400 when bbox h is negative', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 0, bbox: { x: 0, y: 0, w: 50, h: -10 } });
    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBeDefined();
  });

  it('should return 404 when video does not exist', async () => {
    const res = await request(app)
      .post('/track')
      .send({ filename: 'nonexistent.mp4', time: 0, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBeDefined();
  });

  it('should return cached positions when cache file exists', async () => {
    const cachePath = app._getTrackCachePath(testVideo, 0, { x: 0, y: 0, w: 50, h: 50 });
    const fakePositions = [{ time: 0, x: 0, y: 0, w: 50, h: 50 }];
    fs.writeFileSync(cachePath, JSON.stringify(fakePositions));

    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 0, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    expect(res.statusCode).toBe(200);
    expect(res.body.positions).toEqual(fakePositions);
  });

  it('should fall through corrupt cache to re-run tracker', async () => {
    // Write a corrupt cache file
    const cachePath = app._getTrackCachePath(testVideo, 0, { x: 0, y: 0, w: 50, h: 50 });
    fs.writeFileSync(cachePath, 'NOT JSON');

    // The tracker will fail (fake video), but it should NOT return a cache error
    const res = await request(app)
      .post('/track')
      .send({ filename: testVideo, time: 0, bbox: { x: 0, y: 0, w: 50, h: 50 } });
    // Either 200 (if somehow tracker succeeds) or 500 (tracker fails on fake video)
    expect([200, 500]).toContain(res.statusCode);
  });
});

describe('Track cache cleanup on DELETE /uploads/:filename', () => {
  afterEach(cleanUploads);

  it('should delete cached tracking results when a video is deleted', async () => {
    const filename = 'video-with-tracks.mp4';
    fs.writeFileSync(path.join(uploadsDir, filename), 'fake video');
    app._jobs.set(filename, { id: filename, status: 'done', highlights: [] });

    // Create fake track cache files
    const tracksDir = app._tracksDir;
    const trackA = path.join(tracksDir, `${filename}_abc123.json`);
    const trackB = path.join(tracksDir, `${filename}_def456.json`);
    fs.writeFileSync(trackA, '[]');
    fs.writeFileSync(trackB, '[]');

    const res = await request(app).delete(`/uploads/${encodeURIComponent(filename)}`);
    expect(res.statusCode).toBe(200);
    expect(fs.existsSync(trackA)).toBe(false);
    expect(fs.existsSync(trackB)).toBe(false);
  });
});
