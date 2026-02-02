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

```php
GoogleProvider::MODEL_3_1_PRO   // 'gemini-3.1-pro' (newest)
GoogleProvider::MODEL_3_0_PRO   // 'gemini-3.0-pro' (default)
GoogleProvider::MODEL_2_0_FLASH // 'gemini-2.0-flash-exp' (fast)
GoogleProvider::MODEL_1_5_PRO   // 'gemini-1.5-pro'
GoogleProvider::MODEL_1_5_FLASH // 'gemini-1.5-flash' (cost-effective)
```

## Features

- Tool/function calling
- Vision/multimodal support
- Structured output (JSON mode)
- Streaming support

## License

MIT
