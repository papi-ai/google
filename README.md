# PapiAI Google Provider

[![Tests](https://github.com/papi-ai/google/workflows/Tests/badge.svg)](https://github.com/papi-ai/google/actions?query=workflow%3ATests)

Google Gemini provider for [PapiAI](https://github.com/papi-ai/papi-core) - A simple but powerful PHP library for building AI agents.

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

## Available Models

### Gemini (Text/Chat)

```php
GoogleProvider::MODEL_3_1_PRO   // 'gemini-3.1-pro' (newest)
GoogleProvider::MODEL_3_0_PRO   // 'gemini-3.0-pro' (default)
GoogleProvider::MODEL_2_0_FLASH // 'gemini-2.0-flash-exp' (fast)
GoogleProvider::MODEL_1_5_PRO   // 'gemini-1.5-pro'
GoogleProvider::MODEL_1_5_FLASH // 'gemini-1.5-flash' (cost-effective)
```

### Imagen (Image Generation)

```php
GoogleProvider::IMAGEN_3      // 'imagen-3.0-generate-001' (best quality)
GoogleProvider::IMAGEN_3_FAST // 'imagen-3.0-fast-generate-001' (faster)
```

## Features

- Tool/function calling
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support
- Image generation (Imagen 3)

## Image Generation

Generate images using Google's Imagen 3 model:

```php
use PapiAI\Google\GoogleProvider;

$provider = new GoogleProvider($_ENV['GOOGLE_API_KEY']);

// Generate image and get base64 data
$result = $provider->generateImage(
    prompt: 'A professional product photo of headphones on a white background',
    options: [
        'model' => GoogleProvider::IMAGEN_3,
        'aspectRatio' => '1:1',      // 1:1, 16:9, 9:16, 4:3, 3:4
        'numberOfImages' => 1,
        'negativePrompt' => 'blurry, low quality',
    ]
);

// Access generated image
$imageData = base64_decode($result['images'][0]['data']);
file_put_contents('output.png', $imageData);

// Or save directly to file
$provider->generateImageToFile(
    prompt: 'A modern minimalist workspace',
    outputPath: '/path/to/image.png'
);
```

## License

MIT
