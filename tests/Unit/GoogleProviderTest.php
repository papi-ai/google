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
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
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

    describe('chat', function () {
        it('sends messages and returns a Response', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'Hello back!']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => [
                    'promptTokenCount' => 10,
                    'candidatesTokenCount' => 5,
                ],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response)->toBeInstanceOf(Response::class);
            expect($response->text)->toBe('Hello back!');
        });

        it('includes system message as systemInstruction in payload', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'OK']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::system('Be helpful'),
                Message::user('Hello'),
            ]);

            expect($this->provider->lastPayload['systemInstruction'])->toBe([
                'parts' => [['text' => 'Be helpful']],
            ]);
            expect($this->provider->lastPayload['contents'])->toHaveCount(1);
            expect($this->provider->lastPayload['contents'][0]['role'])->toBe('user');
        });

        it('uses default model and max tokens', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'OK']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([Message::user('Hello')]);

            expect($this->provider->lastPayload['generationConfig']['maxOutputTokens'])->toBe(8192);
            expect($this->provider->lastUrl)->toContain('gemini-3-pro-preview');
        });

        it('overrides model and options from parameters', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'OK']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([Message::user('Hello')], [
                'model' => 'gemini-1.5-pro',
                'maxTokens' => 2048,
                'temperature' => 0.5,
                'stopSequences' => ['END'],
            ]);

            expect($this->provider->lastUrl)->toContain('gemini-1.5-pro');
            expect($this->provider->lastPayload['generationConfig']['maxOutputTokens'])->toBe(2048);
            expect($this->provider->lastPayload['generationConfig']['temperature'])->toBe(0.5);
            expect($this->provider->lastPayload['generationConfig']['stopSequences'])->toBe(['END']);
        });

        it('includes tools in payload', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'OK']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $tools = [
                [
                    'name' => 'get_weather',
                    'description' => 'Get weather',
                    'input_schema' => ['type' => 'object', 'properties' => []],
                ],
            ];

            $this->provider->chat([Message::user('Hello')], ['tools' => $tools]);

            expect($this->provider->lastPayload['tools'])->toBe([
                [
                    'functionDeclarations' => [
                        [
                            'name' => 'get_weather',
                            'description' => 'Get weather',
                            'parameters' => ['type' => 'object', 'properties' => []],
                        ],
                    ],
                ],
            ]);
        });

        it('converts tool result messages', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'The weather is sunny']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::user('What is the weather?'),
                Message::assistant('Let me check', [
                    new ToolCall('get_weather', 'get_weather', ['city' => 'London']),
                ]),
                Message::toolResult('get_weather', ['temp' => 20]),
            ]);

            $contents = $this->provider->lastPayload['contents'];
            expect($contents)->toHaveCount(3);

            // Tool result message
            $toolMsg = $contents[2];
            expect($toolMsg['role'])->toBe('user');
            expect($toolMsg['parts'][0]['functionResponse']['name'])->toBe('get_weather');
            expect($toolMsg['parts'][0]['functionResponse']['response']['result'])->toBe('{"temp":20}');
        });

        it('converts assistant messages with tool calls', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'Done']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::user('Hello'),
                Message::assistant('Let me help', [
                    new ToolCall('search', 'search', ['q' => 'test']),
                ]),
                Message::toolResult('search', 'result'),
            ]);

            $contents = $this->provider->lastPayload['contents'];
            $assistantMsg = $contents[1];
            expect($assistantMsg['role'])->toBe('model');
            expect($assistantMsg['parts'][0]['text'])->toBe('Let me help');
            expect($assistantMsg['parts'][1]['functionCall']['name'])->toBe('search');
            expect($assistantMsg['parts'][1]['functionCall']['args'])->toEqual((object) ['q' => 'test']);
        });

        it('converts multimodal messages with url images', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'I see a cat']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 100, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'https://example.com/cat.jpg'),
            ]);

            $contents = $this->provider->lastPayload['contents'];
            $parts = $contents[0]['parts'];
            expect($parts[0]['text'])->toBe('What is this?');
            expect($parts[1]['fileData']['fileUri'])->toBe('https://example.com/cat.jpg');
            expect($parts[1]['fileData']['mimeType'])->toBe('image/jpeg');
        });

        it('converts multimodal messages with base64 images', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'I see a cat']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 100, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::userWithImage('What is this?', 'base64data', 'image/png'),
            ]);

            $contents = $this->provider->lastPayload['contents'];
            $parts = $contents[0]['parts'];
            expect($parts[1]['inlineData']['mimeType'])->toBe('image/png');
            expect($parts[1]['inlineData']['data'])->toBe('base64data');
        });

        it('handles response with tool calls', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => [
                            'parts' => [
                                ['text' => 'Let me check'],
                                [
                                    'functionCall' => [
                                        'name' => 'get_weather',
                                        'args' => ['city' => 'London'],
                                    ],
                                ],
                            ],
                        ],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 20],
            ];

            $response = $this->provider->chat([Message::user('Weather?')]);

            expect($response->hasToolCalls())->toBeTrue();
            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'London']);
        });

        it('includes usage metadata in response', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'OK']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => [
                    'promptTokenCount' => 42,
                    'candidatesTokenCount' => 15,
                ],
            ];

            $response = $this->provider->chat([Message::user('Hello')]);

            expect($response->usage)->toBe([
                'input_tokens' => 42,
                'output_tokens' => 15,
            ]);
        });

        it('handles structured output options', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => '{"name":"test"}']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $schema = ['type' => 'object', 'properties' => ['name' => ['type' => 'string']]];
            $this->provider->chat([Message::user('Hello')], ['outputSchema' => $schema]);

            expect($this->provider->lastPayload['generationConfig']['responseMimeType'])->toBe('application/json');
            expect($this->provider->lastPayload['generationConfig']['responseSchema'])->toBe($schema);
        });

        it('preserves thoughtSignature from tool calls', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => [
                            'parts' => [
                                [
                                    'functionCall' => [
                                        'name' => 'search',
                                        'args' => ['q' => 'test'],
                                    ],
                                    'thoughtSignature' => 'sig_abc123',
                                ],
                            ],
                        ],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            // First call - get tool call with thoughtSignature
            $response = $this->provider->chat([Message::user('Search')]);

            // Second call - send back tool result + assistant message with tool call
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => ['parts' => [['text' => 'Done']]],
                        'finishReason' => 'STOP',
                    ],
                ],
                'usageMetadata' => ['promptTokenCount' => 10, 'candidatesTokenCount' => 5],
            ];

            $this->provider->chat([
                Message::user('Search'),
                Message::assistant('', $response->toolCalls),
                Message::toolResult('search', 'result'),
            ]);

            $contents = $this->provider->lastPayload['contents'];
            $assistantParts = $contents[1]['parts'];
            // The function call part should have the thoughtSignature
            expect($assistantParts[0]['thoughtSignature'])->toBe('sig_abc123');
        });
    });

    describe('stream', function () {
        it('yields StreamChunk for text deltas', function () {
            $this->provider->fakeStreamEvents = [
                ['candidates' => [['content' => ['parts' => [['text' => 'Hello']]]]]],
                ['candidates' => [['content' => ['parts' => [['text' => ' world']]]]]],
            ];

            $chunks = [];
            foreach ($this->provider->stream([Message::user('Hi')]) as $chunk) {
                $chunks[] = $chunk;
            }

            expect($chunks)->toHaveCount(3);
            expect($chunks[0])->toBeInstanceOf(StreamChunk::class);
            expect($chunks[0]->text)->toBe('Hello');
            expect($chunks[1]->text)->toBe(' world');
            expect($chunks[2]->isComplete)->toBeTrue();
        });

        it('uses correct streaming URL', function () {
            $this->provider->fakeStreamEvents = [];

            iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastUrl)->toContain(':streamGenerateContent');
            expect($this->provider->lastUrl)->toContain('alt=sse');
        });

        it('skips events with no text', function () {
            $this->provider->fakeStreamEvents = [
                ['candidates' => [['content' => ['parts' => []]]]],
                ['candidates' => [['content' => ['parts' => [['text' => 'Hi']]]]]],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($chunks)->toHaveCount(2); // text + complete
        });
    });

    describe('generateImage', function () {
        it('sends prompt to imagen predict endpoint', function () {
            $this->provider->fakeResponse = [
                'predictions' => [
                    [
                        'bytesBase64Encoded' => base64_encode('fake-png-data'),
                        'mimeType' => 'image/png',
                    ],
                ],
            ];

            $result = $this->provider->generateImage('A cat');

            expect($this->provider->lastUrl)->toContain(':predict');
            expect($this->provider->lastPayload['instances'][0]['prompt'])->toBe('A cat');
            expect($result['images'])->toHaveCount(1);
            expect($result['images'][0]['mimeType'])->toBe('image/png');
        });

        it('handles generatedImages response format', function () {
            $this->provider->fakeResponse = [
                'generatedImages' => [
                    [
                        'image' => ['imageBytes' => base64_encode('fake-data')],
                    ],
                ],
            ];

            $result = $this->provider->generateImage('A dog');

            expect($result['images'])->toHaveCount(1);
            expect($result['images'][0]['mimeType'])->toBe('image/png');
        });

        it('passes custom options', function () {
            $this->provider->fakeResponse = ['predictions' => []];

            $this->provider->generateImage('A cat', [
                'model' => 'imagen-4.0-ultra-generate-001',
                'numberOfImages' => 2,
                'aspectRatio' => '16:9',
            ]);

            expect($this->provider->lastUrl)->toContain('imagen-4.0-ultra-generate-001');
            expect($this->provider->lastPayload['parameters']['sampleCount'])->toBe(2);
            expect($this->provider->lastPayload['parameters']['aspectRatio'])->toBe('16:9');
        });
    });

    describe('editImage', function () {
        it('sends image data and prompt to generateContent endpoint', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => [
                            'parts' => [
                                [
                                    'inlineData' => [
                                        'mimeType' => 'image/png',
                                        'data' => base64_encode('edited-image'),
                                    ],
                                ],
                                ['text' => 'Here is your edited image'],
                            ],
                        ],
                    ],
                ],
            ];

            $result = $this->provider->editImage('https://example.com/photo.jpg', 'Make it blue');

            expect($this->provider->lastUrl)->toContain(':generateContent');
            expect($result['images'])->toHaveCount(1);
            expect($result['text'])->toBe('Here is your edited image');
        });

        it('skips thought images in response', function () {
            $this->provider->fakeResponse = [
                'candidates' => [
                    [
                        'content' => [
                            'parts' => [
                                [
                                    'thought' => true,
                                    'inlineData' => [
                                        'mimeType' => 'image/png',
                                        'data' => 'thought-image',
                                    ],
                                ],
                                [
                                    'inlineData' => [
                                        'mimeType' => 'image/png',
                                        'data' => 'final-image',
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
            ];

            $result = $this->provider->editImage('https://example.com/photo.png', 'Edit it');

            expect($result['images'])->toHaveCount(1);
            expect($result['images'][0]['data'])->toBe('final-image');
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
            expect($response->model)->toBe('text-embedding-004');
            expect($this->provider->lastUrl)->toContain(':embedContent');
            expect($this->provider->lastUrl)->toContain('text-embedding-004');
            expect($this->provider->lastPayload['model'])->toBe('models/text-embedding-004');
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

    describe('generateImageToFile', function () {
        it('saves generated image to file', function () {
            $this->provider->fakeResponse = [
                'predictions' => [
                    [
                        'bytesBase64Encoded' => base64_encode('fake-png-data'),
                        'mimeType' => 'image/png',
                    ],
                ],
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
            $this->provider->fakeResponse = ['predictions' => []];

            expect(fn () => $this->provider->generateImageToFile('A cat', '/tmp/test.png'))
                ->toThrow(RuntimeException::class, 'No images generated');
        });
    });
});
