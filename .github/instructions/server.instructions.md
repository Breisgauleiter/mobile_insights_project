---
applyTo: "server/**"
---

# Server Instructions (Node.js / Express)

## Stack
- Node.js with CommonJS (`require` / `module.exports`)
- Express 4.x as HTTP framework
- Multer for multipart file uploads
- Jest for testing

## Conventions
- Use `const` by default, `let` only when reassignment is needed
- Error responses: `{ message: "..." }` with appropriate HTTP status codes
- Always validate incoming request data before processing
- Use `path.join(__dirname, ...)` for file paths
- Environment variables via `process.env` with sensible defaults
- Keep endpoint handlers small — extract business logic into separate modules under `server/lib/`

## File Structure
```
server/
  index.js          – App entry point, route registration
  lib/              – Business logic modules
  routes/           – Express route files (when splitting)
  __tests__/        – Jest test files
  uploads/          – Uploaded files (gitignored)
  package.json
```

## Testing
- Test files: `server/__tests__/*.test.js`
- Run: `cd server && npm test`
- Mock file uploads with supertest + multer

## Security
- Never trust user-uploaded filenames — always sanitize
- Limit upload file size via multer options
- No secrets in code — use environment variables
