# Media Submission Session Design Specification

## 1. Feature Summary
This document defines the design specifications for the media submission feature.

Feature scope:
- Upload media files (image-only session or video-only session)
- Require per-file metadata completion before processing
- Route processing to static or dynamic flow based on media type
- Persist a submission session as a folder plus JSON manifest

## 2. Goals
- Keep each upload batch isolated as a session.
- Ensure deterministic naming and traceability.
- Prevent mixed-media submissions (images and videos together).
- Serialize metadata and file paths for downstream processing.

## 3. Session Rules
- A session contains exactly one media category:
  - `static` for images
  - `dynamic` for videos
- Mixed media is invalid:
  - Invalid if the same batch contains both image and video
  - Invalid if session is already image-locked and a video is added (and vice versa)

## 4. UI/UX Specifications
### 4.1 Main Submission Page
- User can stage up to 10 files.
- Allowed types:
  - Images: PNG, JPG, JPEG, WEBP
  - Videos: MP4, MOV, WEBM
- Invalid or unsupported files show an error toast.
- Process button behavior:
  - Greyed out (disabled) until all submitted media items have complete metadata
  - Green (enabled) when all metadata indicators are complete

### 4.2 Metadata Entry per Media Item
- Each media tile shows a top-right status indicator:
  - Red exclamation when metadata is incomplete
  - Green check when metadata is complete
- Clicking the indicator opens a metadata dialog with:
  - Camera dropdown (from JSON source)
  - Auto-populated latitude/longitude from selected camera
  - Date dropdown
  - Time dropdown

## 5. Processing Page Routing
When the user clicks Process:
- Route to `processed-static` if session category is `static`
- Route to `processed-dynamic` if session category is `dynamic`

## 6. Persistence Design
> Note: Folder creation and file writes require backend support (or desktop runtime). Browser-only frontend cannot reliably write arbitrary folders on disk.

### 6.1 Session Folder Naming
Create a folder under category root:
- `static-submission/session-YYYYMMDD-HHmmss`
- `dynamic-submission/session-YYYYMMDD-HHmmss`

Timestamp format constraints:
- Use filesystem-safe format (avoid `:`)
- Recommended: `YYYYMMDD-HHmmss`

### 6.2 Stored Artifacts per Session
Inside each session folder:
- Uploaded media files copied/saved into session folder
- Session manifest JSON saved as:
  - `session-YYYYMMDD-HHmmss.json`

## 7. Data Model
`Media` object fields:
- `unprocessedMediaPath: string`
- `processedMediaPath: string`
- `date: epochTime`
- `latitudeCoords: float`
- `longitudeCoords: float`
- `flagged: boolean` (required; defaults to `false`, set to `true` when trash is detected)

Constructor:
- `Media(unprocessedMediaPath, date, latitudeCoords, longitudeCoords)`

Methods:
- Getters: latitude, longitude, date, unprocessed, processed
- Setter: processed

## 8. Session Manifest Specification
Recommended manifest shape:

```json
{
  "sessionId": "session-20260316-153045",
  "mediaType": "static",
  "createdAtEpoch": 1773684645,
  "items": [
    {
      "unprocessedMediaPath": "static-submission/session-20260316-153045/image1.jpg",
      "processedMediaPath": "",
      "date": 1773684600,
      "latitudeCoords": 37.7614,
      "longitudeCoords": -122.1826,
      "flagged": false
    }
  ]
}
```

Required manifest properties:
- `sessionId`
- `mediaType`
- `createdAtEpoch`
- `items[]`

Required item properties:
- `unprocessedMediaPath`
- `processedMediaPath` (required; initialize to `""` until processing completes)
- `date`
- `latitudeCoords`
- `longitudeCoords`
- `flagged` (required; initialize to `false` before detection, update to `true` on trash detection)

Optional item properties:
- `mediaId`
- `status`
- `errors[]`

`mediaId` guidance:
- Not strictly required if `unprocessedMediaPath` is guaranteed unique within a session.
- Recommended when items may be renamed, re-ordered, retried, or referenced by asynchronous jobs.

## 9. Backend API Contract (Recommended)
### 9.1 Create Session + Upload
- `POST /api/sessions`
- Request:
  - mediaType
  - files (multipart)
  - metadata per file
- Response:
  - sessionId
  - sessionFolderPath
  - manifestPath

### 9.2 Trigger Processing
- `POST /api/sessions/{sessionId}/process`
- Response:
  - accepted flag
  - processing job id

### 9.3 Get Session State
- `GET /api/sessions/{sessionId}`
- Response:
  - processing status
  - per-item state
  - processed paths (when available)

## 10. Validation and Error Handling
Validation rules:
- Reject mixed-media batches.
- Reject files that do not match allowed mime types.
- Reject submit/process if required metadata is incomplete.
- Enforce max files per session.

Error handling:
- Show user-facing toast for invalid operations.
- Persist backend validation errors with actionable messages.
- Log session creation and write failures with sessionId.

## 11. Security and Reliability Requirements
- Sanitize uploaded filenames.
- Do not trust client-provided paths.
- Validate mime type and extension server-side.
- Use atomic write for manifest file.
- Store files outside public web root unless explicitly required.

## 12. Acceptance Criteria
- User cannot create or process mixed media sessions.
- Every uploaded file has metadata status indicator.
- Process button only enables when all metadata is complete.
- Correct page routing occurs by media type.
- Session folder and manifest are generated with required naming and fields.
- Manifest deserializes into `Media` objects without data loss.

## 13. Open Questions
- Should session folders be retained permanently or expire?
- Should processing overwrite existing `processedMediaPath` values?
- Do we need timezone normalization for epoch conversion?
- Should camera metadata be snapshotted into manifest for audit reproducibility?
