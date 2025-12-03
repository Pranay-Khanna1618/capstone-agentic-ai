# API Quickstart
NovaCRM exposes REST endpoints under `/api/v1/` and requires an API key in the header.

Example request:
```
GET /api/v1/accounts/{id}
Headers: Authorization: Bearer <API_KEY>
```

Rate Limits per plan:
- Free: 10 requests/minute
- Pro: 100 requests/minute
- Enterprise: 1000 requests/minute

Exceeding the rate limit returns HTTP 429 with a `Retry-After` header.