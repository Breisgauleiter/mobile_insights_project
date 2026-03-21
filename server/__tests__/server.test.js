const request = require('supertest');
const path = require('path');
const fs = require('fs');
const app = require('../index');

const uploadsDir = path.join(__dirname, '..', 'uploads');

function cleanUploads() {
  fs.readdirSync(uploadsDir).forEach((file) => {
    if (file !== '.gitkeep') fs.unlinkSync(path.join(uploadsDir, file));
  });
}

afterAll(cleanUploads);

describe('GET /', () => {
  it('should return health check message', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(200);
    expect(res.body.message).toBe('Mobile Insights Server ist online');
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

  it('should list uploaded files', async () => {
    const res = await request(app).get('/uploads');
    expect(res.statusCode).toBe(200);
    expect(Array.isArray(res.body.uploads)).toBe(true);
    expect(res.body.uploads.length).toBeGreaterThanOrEqual(1);
    expect(res.body.uploads[0]).toHaveProperty('filename');
    expect(res.body.uploads[0]).toHaveProperty('hasResults');
  });
});

describe('GET /results/:id', () => {
  let uploadedFilename;

  beforeAll(async () => {
    const res = await request(app)
      .post('/upload')
      .attach('video', Buffer.from('fake video'), {
        filename: 'result-test.mp4',
        contentType: 'video/mp4',
      });
    uploadedFilename = res.body.filename;
  });

  afterAll(cleanUploads);

  it('should return results for a valid id', async () => {
    const res = await request(app).get(`/results/${uploadedFilename}`);
    expect(res.statusCode).toBe(200);
    expect(res.body.id).toBe(uploadedFilename);
    expect(res.body.status).toBe('processed');
    expect(Array.isArray(res.body.highlights)).toBe(true);
    expect(res.body.highlights.length).toBe(2);
  });

  it('should return 404 for an unknown id', async () => {
    const res = await request(app).get('/results/nonexistent-file.mp4');
    expect(res.statusCode).toBe(404);
    expect(res.body.error).toBe('No results found for this id.');
  });
});
