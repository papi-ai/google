# Google Gemini

Google Gemini provider for PapiAI.

## Installation

```bash
composer require papi-ai/google
```

## Usage

```php
use PapiAI\Core\Agent;
use PapiAI\Google\GoogleProvider;

$provider = new GoogleProvider(
    apiKey: $_ENV['GOOGLE_API_KEY'],
    defaultModel: GoogleProvider::MODEL_3_0_PRO,
);

$agent = new Agent(
    provider: $provider,
    model: 'gemini-3.0-pro',
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Models

### Chat Models

```php
GoogleProvider::MODEL_3_1_PRO       // gemini-3.1-pro (newest)
GoogleProvider::MODEL_3_0_PRO       // gemini-3.0-pro
GoogleProvider::MODEL_3_FLASH       // gemini-3-flash
GoogleProvider::MODEL_2_5_PRO       // gemini-2.5-pro
GoogleProvider::MODEL_2_5_FLASH     // gemini-2.5-flash
GoogleProvider::MODEL_2_0_FLASH     // gemini-2.0-flash
GoogleProvider::MODEL_1_5_PRO       // gemini-1.5-pro
GoogleProvider::MODEL_1_5_FLASH     // gemini-1.5-flash
```

### Image Generation (Imagen)

```php
GoogleProvider::IMAGEN_4            // imagen-4.0-generate-001
GoogleProvider::IMAGEN_4_ULTRA      // imagen-4.0-ultra-generate-001
```

## Image Generation

Generate images using Google's Imagen model:

```php
use PapiAI\Google\GoogleProvider;

$provider = new GoogleProvider($_ENV['GOOGLE_API_KEY']);

// Generate image and get base64 data
$result = $provider->generateImage(
    prompt: 'A professional product photo of headphones',
    options: [
        'model' => GoogleProvider::IMAGEN_4,
        'aspectRatio' => '1:1',      // 1:1, 16:9, 9:16, 4:3, 3:4
        'numberOfImages' => 1,
    ]
);

// Access generated image
$imageData = base64_decode($result['images'][0]['data']);
file_put_contents('output.png', $imageData);

// Or save directly to file
$provider->generateImageToFile(
    prompt: 'A modern minimalist workspace',
    outputPath: '/path/to/image.png',
);
```

## Capabilities

| Capability | Supported |
|---|---|
| Chat | Yes |
| Streaming | Yes |
| Tool calling | Yes |
| Vision | Yes |
| Structured output | Yes |
| Embeddings | Yes |
| Image generation | Yes |
| Image editing | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.14
