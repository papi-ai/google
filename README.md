# PapiAI Google Provider

[![CI](https://github.com/papi-ai/google/workflows/CI/badge.svg)](https://github.com/papi-ai/google/actions?query=workflow%3ACI) [![Latest Version](https://img.shields.io/packagist/v/papi-ai/google.svg)](https://packagist.org/packages/papi-ai/google) [![Total Downloads](https://img.shields.io/packagist/dt/papi-ai/google.svg)](https://packagist.org/packages/papi-ai/google) [![PHP Version](https://img.shields.io/packagist/php-v/papi-ai/google.svg)](https://packagist.org/packages/papi-ai/google) [![License](https://img.shields.io/packagist/l/papi-ai/google.svg)](https://packagist.org/packages/papi-ai/google)

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

## Available Models

### Gemini (text and chat)

```php
GoogleProvider::MODEL_3_6_FLASH      // 'gemini-3.6-flash' (default)
GoogleProvider::MODEL_3_5_FLASH      // 'gemini-3.5-flash'
GoogleProvider::MODEL_3_5_FLASH_LITE // 'gemini-3.5-flash-lite'
GoogleProvider::MODEL_3_1_PRO        // 'gemini-3.1-pro-preview'
GoogleProvider::MODEL_3_FLASH        // 'gemini-3-flash-preview'
GoogleProvider::MODEL_2_5_PRO        // 'gemini-2.5-pro'
GoogleProvider::MODEL_2_5_FLASH      // 'gemini-2.5-flash'
```

### Image generation and editing

```php
GoogleProvider::MODEL_3_1_FLASH_IMAGE      // 'gemini-3.1-flash-image' (default)
GoogleProvider::MODEL_3_1_FLASH_LITE_IMAGE // 'gemini-3.1-flash-lite-image'
GoogleProvider::MODEL_3_PRO_IMAGE          // 'gemini-3-pro-image'
GoogleProvider::MODEL_2_5_FLASH_IMAGE      // 'gemini-2.5-flash-image'
```

The Imagen constants are still here but every one of them is `@deprecated`: the Imagen line
shuts down on **17 August 2026**, and its `:predict` endpoint has no successor. Use the Gemini
image models above.

### Video generation

```php
GoogleProvider::MODEL_VEO_3_1      // 'veo-3.1-generate-preview' (default)
GoogleProvider::MODEL_VEO_3_1_LITE // 'veo-3.1-lite-generate-preview'
```

## Features

- Tool/function calling, including forced tool choice
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support
- Reasoning effort, mapped to Gemini's thinking levels and budgets
- Image generation and editing
- Video generation and text embeddings

## Image Generation

```php
use PapiAI\Google\GoogleProvider;

$provider = new GoogleProvider($_ENV['GOOGLE_API_KEY']);

$result = $provider->generateImage(
    prompt: 'A professional product photo of headphones on a white background',
    options: [
        'model' => GoogleProvider::MODEL_3_1_FLASH_IMAGE,
        'aspectRatio' => '1:1',  // 1:1, 16:9, 9:16, 4:3, 3:4, and more on 3.1 Flash Image
        'imageSize' => '2K',     // 1K, 2K or 4K ('0.5K' on 3.1 Flash Image)
    ]
);

$imageData = base64_decode($result['images'][0]['data']);
file_put_contents('output.png', $imageData);

// Or save straight to disk
$provider->generateImageToFile(
    prompt: 'A modern minimalist workspace',
    outputPath: '/path/to/image.png'
);
```

Leave `aspectRatio` and `imageSize` out and the model picks its own, rather than being forced
into a square.

These models return **one image per request**. Asking for `numberOfImages` greater than one
throws a `ProviderException` instead of quietly handing back a single image; call
`generateImage()` once per image you need.

## Image Editing

```php
$result = $provider->editImage(
    imageUrl: 'https://example.com/photo.jpg',
    prompt: 'Make the sky dramatic and overcast',
);

file_put_contents('edited.png', base64_decode($result['images'][0]['data']));
echo $result['text']; // any commentary the model returned alongside the image
```

## License

MIT
