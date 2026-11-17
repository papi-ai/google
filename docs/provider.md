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
    defaultModel: GoogleProvider::MODEL_3_8_FLASH,
);

$agent = new Agent(
    provider: $provider,
    model: GoogleProvider::MODEL_3_8_FLASH,
    instructions: 'You are a helpful assistant.',
);

$response = $agent->run('Hello!');
echo $response->text;
```

## Models

### Chat Models

```php
GoogleProvider::MODEL_4_HYDROGEN     // gemini-4-hydrogen
GoogleProvider::MODEL_4_CARBON       // gemini-4-carbon
GoogleProvider::MODEL_4_ARGON        // gemini-4-argon
GoogleProvider::MODEL_3_8_FLASH      // gemini-3.8-flash (default)
GoogleProvider::MODEL_3_6_FLASH      // gemini-3.6-flash
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
GoogleProvider::MODEL_NANO_BANANA_2_1      // gemini-nano-banana-2.1
GoogleProvider::MODEL_3_1_FLASH_LITE_IMAGE // gemini-3.1-flash-lite-image
GoogleProvider::MODEL_3_PRO_IMAGE          // gemini-3-pro-image
GoogleProvider::MODEL_2_5_FLASH_IMAGE      // gemini-2.5-flash-image
```

The `IMAGEN_*` constants remain for backwards compatibility and are all `@deprecated`. Imagen
shuts down on 17 August 2026 and its separate `:predict` endpoint goes with it.

### Embedding Models

```php
GoogleProvider::MODEL_EMBEDDING_2   // gemini-embedding-2 (default, 3072 dimensions)
GoogleProvider::MODEL_EMBEDDING_001 // gemini-embedding-001
```

Pass `dimensions` (768, 1536 or 3072) to truncate the vector.

## Image Generation

Image generation and editing go through the Interactions API, asking for an image back through
`response_format`.

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

$image = $result['images'][0];

// These models return JPEG by default, so take the extension from the response
// rather than assuming PNG.
$extension = str_replace('image/', '', $image['mimeType']);
file_put_contents("output.{$extension}", base64_decode($image['data']));

// Or save directly to file, choosing your own path
$provider->generateImageToFile(
    prompt: 'A modern minimalist workspace',
    outputPath: '/path/to/image.jpg',
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

file_put_contents('edited.jpg', base64_decode($result['images'][0]['data']));
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

Thinking counts against `maxTokens`. Current Gemini models think before they answer, and even a one-line question
can use a couple of hundred tokens at the default effort, so a small ceiling can run out before the
answer: with `outputSchema`, what comes back is then cut-off prose rather than JSON.
`$response->isTruncated()` reports it; raise `maxTokens` or lower `effort`.

## Creating It by Name

An application that picks its provider from configuration can build this one without naming
its class:

```php
use PapiAI\Core\Factory\ComposerDiscovery;
use PapiAI\Core\Factory\ProviderRegistry;

$registry = new ProviderRegistry(new ComposerDiscovery());

$provider = $registry->create('google', [
    'api_key' => $_ENV['GOOGLE_API_KEY'],
]);
```

It takes `api_key` (required), `model`, `max_tokens` and `effort`. Anything it is not given falls
back to the constructor's defaults, and a missing or unknown setting throws `ConfigurationException`
before any request is made.

## Requirements

- PHP 8.2+
- `ext-curl`
- `papi-ai/papi-core` ^0.15
