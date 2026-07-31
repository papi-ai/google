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
    defaultModel: GoogleProvider::MODEL_3_6_FLASH,
);

$agent = new Agent(
    provider: $provider,
    model: GoogleProvider::MODEL_3_6_FLASH,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Models

### Chat Models

```php
GoogleProvider::MODEL_3_6_FLASH      // gemini-3.6-flash (default)
GoogleProvider::MODEL_3_5_FLASH      // gemini-3.5-flash
GoogleProvider::MODEL_3_5_FLASH_LITE // gemini-3.5-flash-lite
GoogleProvider::MODEL_3_1_PRO        // gemini-3.1-pro-preview
GoogleProvider::MODEL_3_FLASH        // gemini-3-flash-preview
GoogleProvider::MODEL_2_5_PRO        // gemini-2.5-pro
GoogleProvider::MODEL_2_5_FLASH      // gemini-2.5-flash
GoogleProvider::MODEL_2_5_FLASH_LITE // gemini-2.5-flash-lite
GoogleProvider::MODEL_2_0_FLASH      // gemini-2.0-flash
```

### Image Models

```php
GoogleProvider::MODEL_3_1_FLASH_IMAGE      // gemini-3.1-flash-image (default)
GoogleProvider::MODEL_3_1_FLASH_LITE_IMAGE // gemini-3.1-flash-lite-image
GoogleProvider::MODEL_3_PRO_IMAGE          // gemini-3-pro-image
GoogleProvider::MODEL_2_5_FLASH_IMAGE      // gemini-2.5-flash-image
```

The `IMAGEN_*` constants remain for backwards compatibility and are all `@deprecated`. Imagen
shuts down on 17 August 2026 and its separate `:predict` endpoint goes with it.

## Image Generation

Image generation and editing both go through `generateContent`, asking for an image modality
back. There is no separate endpoint any more.

```php
use PapiAI\Google\GoogleProvider;

$provider = new GoogleProvider($_ENV['GOOGLE_API_KEY']);

$result = $provider->generateImage(
    prompt: 'A professional product photo of headphones',
    options: [
        'model' => GoogleProvider::MODEL_3_1_FLASH_IMAGE,
        'aspectRatio' => '1:1',  // 1:1, 16:9, 9:16, 4:3, 3:4, and more on 3.1 Flash Image
        'imageSize' => '2K',     // 1K, 2K or 4K
    ]
);

$imageData = base64_decode($result['images'][0]['data']);
file_put_contents('output.png', $imageData);

// Or save directly to file
$provider->generateImageToFile(
    prompt: 'A modern minimalist workspace',
    outputPath: '/path/to/image.png',
);
```

Both options are optional. Omit them and the model chooses, rather than being pushed into a
square by a default the caller never asked for.

One image comes back per request. `numberOfImages` above 1 throws a `ProviderException` rather
than silently returning a single image, because the Gemini image models have no equivalent of
Imagen's `sampleCount`.

## Image Editing

```php
$result = $provider->editImage(
    imageUrl: 'https://example.com/photo.jpg',
    prompt: 'Make the sky dramatic and overcast',
);

file_put_contents('edited.png', base64_decode($result['images'][0]['data']));
echo $result['text'];
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
| Video generation | Yes |
| Forced tool choice | Yes, including a named tool |
| Reasoning effort | Yes |

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.15
