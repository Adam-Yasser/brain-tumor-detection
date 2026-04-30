# Brain Tumor Detection API

REST API that wraps the trained brain-tumor classifier so the .NET backend (or any HTTP client) can run predictions without a Python runtime.

- **Base URL (local dev):** `http://localhost:8000`
- **Interactive docs (Swagger UI):** `http://localhost:8000/docs`
- **OpenAPI JSON:** `http://localhost:8000/openapi.json`

The .NET team can point a code generator at `/openapi.json` (e.g. NSwag, Kiota) to produce a typed C# client automatically.

## Running the service

```bash
# from the project root
python scripts/run_api.py
```

The model loads once at startup; every request reuses it.

## Endpoints

### `GET /health`

Liveness probe. Returns `200 OK` once the model is loaded.

**Response:**
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

Classify a single brain MRI image.

- **Request:** `multipart/form-data` with a single field named `file` containing the image (JPEG, PNG, BMP, or TIFF; max 10 MB).
- **Response:** `200 OK` with the prediction.

**Response body:**
```json
{
  "predicted_class": "glioma",
  "confidence": 0.9542,
  "probabilities": {
    "glioma":     0.9542,
    "meningioma": 0.0301,
    "notumor":    0.0098,
    "pituitary":  0.0059
  }
}
```

**Classes:** `glioma`, `meningioma`, `notumor`, `pituitary`.

**Error responses:**

| Status | Meaning                                   |
| ------ | ----------------------------------------- |
| 400    | Empty file or unreadable image            |
| 413    | Image exceeds 10 MB                       |
| 415    | Content type is not an allowed image type |
| 503    | Model is not yet loaded                   |

## C# integration example

```csharp
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;

public sealed class BrainTumorPrediction
{
    [JsonPropertyName("predicted_class")]
    public string PredictedClass { get; set; } = "";

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("probabilities")]
    public Dictionary<string, double> Probabilities { get; set; } = new();
}

public sealed class BrainTumorClient
{
    private readonly HttpClient _http;

    public BrainTumorClient(HttpClient http) => _http = http;

    public async Task<BrainTumorPrediction> PredictAsync(
        Stream imageStream,
        string fileName,
        string contentType = "image/jpeg",
        CancellationToken ct = default)
    {
        using var form = new MultipartFormDataContent();
        var fileContent = new StreamContent(imageStream);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue(contentType);
        form.Add(fileContent, name: "file", fileName: fileName);

        using var response = await _http.PostAsync("/predict", form, ct);
        response.EnsureSuccessStatusCode();

        return (await response.Content.ReadFromJsonAsync<BrainTumorPrediction>(cancellationToken: ct))!;
    }
}
```

Register it in `Program.cs`:

```csharp
builder.Services.AddHttpClient<BrainTumorClient>(c =>
{
    c.BaseAddress = new Uri(builder.Configuration["BrainTumorApi:BaseUrl"]
                          ?? "http://localhost:8000");
    c.Timeout = TimeSpan.FromSeconds(30);
});
```

`appsettings.json`:

```json
{
  "BrainTumorApi": {
    "BaseUrl": "http://localhost:8000"
  }
}
```

## curl example (smoke test)

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/mri.jpg;type=image/jpeg"
```

## Deployment notes for the team

- Run the API and the .NET backend as **two separate services**. The .NET app talks to the API over HTTP — no Python required on the .NET side.
- For production, put both behind the same reverse proxy (or container network) and set `BrainTumorApi:BaseUrl` to the internal address (e.g. `http://ml-api:8000`).
- CORS is currently set to `*` for easy local integration; tighten it before going to production.
- The model file (`outputs/models/best_model.pth`) must be present at the path given by the `MODEL_PATH` env var (default `outputs/models/best_model.pth`).
