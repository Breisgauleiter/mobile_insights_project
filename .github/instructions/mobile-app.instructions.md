---
applyTo: "mobile_app/**"
---

# Mobile App Instructions (React Native)

## Stack
- React Native with TypeScript
- Expo or bare workflow (TBD)
- Native modules: MediaProjection (Android), ReplayKit (iOS)

## Conventions
- TypeScript strict mode
- Functional components with hooks
- State management: React Context or Zustand (TBD)
- Style: StyleSheet.create (no inline styles)
- Navigation: React Navigation

## File Structure (planned)
```
mobile_app/
  src/
    components/     – Reusable UI components
    screens/        – Screen components
    services/       – API calls, recording service
    hooks/          – Custom hooks
    types/          – TypeScript types
  App.tsx           – Entry point
  package.json
  tsconfig.json
```

## Key Features (planned)
- Screen recording via native APIs
- Video upload to server POST `/upload`
- Display highlight results from server
