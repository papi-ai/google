<?php

/*
 * This file is part of PapiAI,
 * A simple but powerful PHP library for building AI agents.
 *
 * (c) Marcello Duarte <marcello.duarte@gmail.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

declare(strict_types=1);

use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ImageProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Google\GoogleProvider;

/**
 * Test subclass that stubs HTTP methods for unit testing.
 */
class TestableGoogleProvider extends GoogleProvider
{
    public string $lastUrl = '';
    public array $lastPayload = [];
    public array $fakeResponse = [];
    public array $fakeEmbeddingResponse = [];
    public array $fakeStreamEvents = [];
    public string $fakeImageData = 'fake-image-binary-data';

    protected function request(string $url, array $payload): array
    {
        $this->lastUrl = $url;
        $this->lastPayload = $payload;

        return $this->fakeResponse;
    }

    protected function streamRequest(string $url, array $payload): Generator
    {
        $this->lastUrl = $url;
        $this->lastPayload = $payload;

        foreach ($this->fakeStreamEvents as $event) {
            yield $event;
        }
    }

    protected function embeddingRequest(array $payload, string $url): array
    {
        $this->lastUrl = $url;
        $this->lastPayload = $payload;

        return $this->fakeEmbeddingResponse;
    }

    protected function fetchImage(string $url): string|false
    {
        return $this->fakeImageData;
    }

    public function callThrowForStatusCode(int $httpCode, ?array $data): never
    {
        $this->throwForStatusCode($httpCode, $data);
    }
}

describe('GoogleProvider', function () {
    beforeEach(function () {
        $this->provider = new TestableGoogleProvider('test-api-key');
    });

    describe('construction', function () {
        it('implements ProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('implements ImageProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(ImageProviderInterface::class);
        });

        it('implements EmbeddingProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(EmbeddingProviderInterface::class);
        });

        it('returns google as name', function () {
            expect($this->provider->getName())->toBe('google');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            expect($this->provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            expect($this->provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            expect($this->provider->supportsStructuredOutput())->toBeTrue();
        });

        it('supports image generation', function () {
            expect($this->provider->supportsImageGeneration())->toBeTrue();
        });

        it('supports image editing', function () {
            expect($this->provider->supportsImageEditing())->toBeTrue();
        });
    });

    describe('embed', function () {
        it('embeds a single string input', function () {
            $this->provider->fakeEmbeddingResponse = [
                'embedding' => [
                    'values' => [0.1, 0.2, 0.3],
                ],
            ];

            $response = $this->provider->embed('Hello world');

            expect($response)->toBeInstanceOf(EmbeddingResponse::class);
            expect($response->embeddings)->toBe([[0.1, 0.2, 0.3]]);
            expect($response->model)->toBe('gemini-embedding-2');
            expect($this->provider->lastUrl)->toContain(':embedContent');
            expect($this->provider->lastUrl)->toContain('gemini-embedding-2');
            expect($this->provider->lastPayload['model'])->toBe('models/gemini-embedding-2');
            expect($this->provider->lastPayload)->not->toHaveKey('outputDimensionality');
            expect($this->provider->lastPayload['content']['parts'][0]['text'])->toBe('Hello world');
        });

        it('embeds an array of inputs using batch endpoint', function () {
            $this->provider->fakeEmbeddingResponse = [
                'embeddings' => [
                    ['values' => [0.1, 0.2, 0.3]],
                    ['values' => [0.4, 0.5, 0.6]],
                ],
            ];

            $response = $this->provider->embed(['Hello', 'World']);

            expect($response)->toBeInstanceOf(EmbeddingResponse::class);
            expect($response->embeddings)->toBe([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
            expect($response->count())->toBe(2);
            expect($this->provider->lastUrl)->toContain(':batchEmbedContents');
            expect($this->provider->lastPayload['requests'])->toHaveCount(2);
            expect($this->provider->lastPayload['requests'][0]['content']['parts'][0]['text'])->toBe('Hello');
            expect($this->provider->lastPayload['requests'][1]['content']['parts'][0]['text'])->toBe('World');
        });

        it('uses custom model', function () {
            $this->provider->fakeEmbeddingResponse = [
                'embedding' => [
                    'values' => [0.1, 0.2],
                ],
            ];

            $response = $this->provider->embed('Test', ['model' => 'text-embedding-005']);

            expect($response->model)->toBe('text-embedding-005');
            expect($this->provider->lastUrl)->toContain('text-embedding-005');
            expect($this->provider->lastPayload['model'])->toBe('models/text-embedding-005');
        });

        it('passes dimensions as outputDimensionality, on each request of a batch too', function () {
            $this->provider->fakeEmbeddingResponse = ['embedding' => ['values' => [0.1]]];
            $this->provider->embed('Test', ['dimensions' => 768]);
            expect($this->provider->lastPayload['outputDimensionality'])->toBe(768);

            $this->provider->fakeEmbeddingResponse = ['embeddings' => [['values' => [0.1]], ['values' => [0.2]]]];
            $this->provider->embed(['a', 'b'], ['dimensions' => 768]);
            expect($this->provider->lastPayload['requests'][1]['outputDimensionality'])->toBe(768);
        });

        it('parses response with correct structure', function () {
            $this->provider->fakeEmbeddingResponse = [
                'embedding' => [
                    'values' => [0.1, 0.2, 0.3, 0.4],
                ],
            ];

            $response = $this->provider->embed('Test');

            expect($response->first())->toBe([0.1, 0.2, 0.3, 0.4]);
            expect($response->dimensions())->toBe(4);
            expect($response->count())->toBe(1);
        });
    });

    describe('error mapping', function () {
        it('throws AuthenticationException for 401', function () {
            expect(fn () => $this->provider->callThrowForStatusCode(401, ['error' => ['message' => 'Invalid API key']]))
                ->toThrow(PapiAI\Core\Exception\AuthenticationException::class);
        });

        it('throws RateLimitException for 429', function () {
            expect(fn () => $this->provider->callThrowForStatusCode(429, ['error' => ['message' => 'Rate limit exceeded']]))
                ->toThrow(PapiAI\Core\Exception\RateLimitException::class);
        });

        it('throws ProviderException for 500', function () {
            expect(fn () => $this->provider->callThrowForStatusCode(500, ['error' => ['message' => 'Internal server error']]))
                ->toThrow(PapiAI\Core\Exception\ProviderException::class);
        });

        it('throws ProviderException when data is null', function () {
            expect(fn () => $this->provider->callThrowForStatusCode(500, null))
                ->toThrow(PapiAI\Core\Exception\ProviderException::class);
        });
    });

    describe('generateImageToFile', function () {
        it('saves generated image to file', function () {
            $this->provider->fakeResponse = [
                'status' => 'completed',
                'steps' => [['type' => 'model_output', 'content' => [
                    ['type' => 'image', 'mime_type' => 'image/png', 'data' => base64_encode('fake-png-data')],
                ]]],
                'usage' => [],
            ];

            $tmpFile = tempnam(sys_get_temp_dir(), 'papi_test_');

            try {
                $path = $this->provider->generateImageToFile('A cat', $tmpFile);
                expect($path)->toBe($tmpFile);
                expect(file_get_contents($tmpFile))->toBe('fake-png-data');
            } finally {
                @unlink($tmpFile);
            }
        });

        it('throws when no images generated', function () {
            $this->provider->fakeResponse = ['status' => 'completed', 'steps' => [], 'usage' => []];

            expect(fn () => $this->provider->generateImageToFile('A cat', '/tmp/test.png'))
                ->toThrow(RuntimeException::class, 'No images generated');
        });
    });
});
