const request = require('supertest');
const app = require('../index');

describe('GET /', () => {
  it('should return health check message', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(200);
    expect(res.body.message).toBe('Mobile Insights Server ist online');
  });
});

describe('POST /upload', () => {
  it('should return 400 when no file is uploaded', async () => {
    const res = await request(app).post('/upload');
    expect(res.statusCode).toBe(400);
    expect(res.body.message).toBe('No file uploaded');
  });
});
