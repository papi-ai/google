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

namespace PapiAI\Google;

use Generator;
use PapiAI\Core\Contracts\EmbeddingProviderInterface;
use PapiAI\Core\Contracts\ImageProviderInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use RuntimeException;

/**
 * Google Gemini API Provider.
 *
 * Supports Gemini models including:
 *
 * Gemini 3.x (Latest):
 * - gemini-3.1-pro (newest, best quality)
 * - gemini-3.0-pro (excellent quality)
 *
 * Gemini 2.x:
 * - gemini-2.5-pro (best quality)
 * - gemini-2.5-flash (fast, balanced)
 * - gemini-2.5-flash-lite (fastest, most cost-effective)
 * - gemini-2.0-flash (fast, multimodal)
 * - gemini-2.0-flash-lite (lightweight)
 *
 * Gemini 1.5:
 * - gemini-1.5-pro (proven quality)
 * - gemini-1.5-flash (fast, cost-effective)
 */
class GoogleProvider implements ProviderInterface, ImageProviderInterface, EmbeddingProviderInterface
{
    private const API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

    // Gemini model aliases
    public const MODEL_3_1_PRO = 'gemini-3.1-pro-preview';
    public const MODEL_3_0_PRO = 'gemini-3-pro-preview';
    public const MODEL_3_FLASH = 'gemini-3-flash-preview';
    public const MODEL_3_PRO_IMAGE = 'gemini-3-pro-image-preview';
    public const MODEL_2_5_PRO = 'gemini-2.5-pro';
    public const MODEL_2_5_FLASH = 'gemini-2.5-flash';
    public const MODEL_2_5_FLASH_LITE = 'gemini-2.5-flash-lite';
    public const MODEL_2_0_FLASH = 'gemini-2.0-flash';
    public const MODEL_2_0_FLASH_LITE = 'gemini-2.0-flash-lite';
    public const MODEL_1_5_PRO = 'gemini-1.5-pro';
    public const MODEL_1_5_FLASH = 'gemini-1.5-flash';

    // Imagen model aliases for image generation and editing
    public const IMAGEN_4 = 'imagen-4.0-generate-001';
    public const IMAGEN_4_ULTRA = 'imagen-4.0-ultra-generate-001';
    public const IMAGEN_4_FAST = 'imagen-4.0-fast-generate-001';
    public const IMAGEN_EDIT = 'imagen-3.0-capability-001';

    /** @var array<string, string> tool call ID → thought signature */
    private array $thoughtSignatures = [];

    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_3_0_PRO,
        private readonly int $defaultMaxTokens = 8192,
    ) {
    }

    public function chat(array $messages, array $options = []): Response
    {
        $model = $options['model'] ?? $this->defaultModel;
        $payload = $this->buildPayload($messages, $options);

        $url = self::API_BASE . "/{$model}:generateContent?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        return $this->parseResponse($response, $messages);
    }

    public function stream(array $messages, array $options = []): iterable
    {
        $model = $options['model'] ?? $this->defaultModel;
        $payload = $this->buildPayload($messages, $options);

        $url = self::API_BASE . "/{$model}:streamGenerateContent?key={$this->apiKey}&alt=sse";

        foreach ($this->streamRequest($url, $payload) as $event) {
            $text = $this->extractTextFromCandidate($event);
            if ($text !== '') {
                yield new StreamChunk($text);
            }
        }

        yield new StreamChunk('', isComplete: true);
    }

    public function supportsTool(): bool
    {
        return true;
    }

    public function supportsVision(): bool
    {
        return true;
    }

    public function supportsStructuredOutput(): bool
    {
        return true; // Gemini supports JSON mode
    }

    public function getName(): string
    {
        return 'google';
    }

    public function supportsImageGeneration(): bool
    {
        return true;
    }

    public function supportsImageEditing(): bool
    {
        return true;
    }

    /**
     * Generate an image using Google's Imagen 4 API.
     *
     * @param string $prompt The image generation prompt
     * @param array{
     *     model?: string,
     *     numberOfImages?: int,
     *     aspectRatio?: string,
     *     imageSize?: int,
     * } $options Generation options
     * @return array{images: array<array{mimeType: string, data: string}>}
     */
    public function generateImage(string $prompt, array $options = []): array
    {
        $model = $options['model'] ?? self::IMAGEN_4_FAST;
        $numberOfImages = $options['numberOfImages'] ?? 1;
        $aspectRatio = $options['aspectRatio'] ?? '1:1';

        // Imagen 4 uses the predict endpoint with instances/parameters format
        $payload = [
            'instances' => [
                ['prompt' => $prompt],
            ],
            'parameters' => [
                'sampleCount' => $numberOfImages,
                'aspectRatio' => $aspectRatio,
                'outputOptions' => [
                    'mimeType' => 'image/png',
                ],
            ],
        ];

        $url = self::API_BASE . "/{$model}:predict?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        $images = [];

        // Imagen returns predictions with bytesBase64Encoded
        foreach ($response['predictions'] ?? [] as $prediction) {
            if (isset($prediction['bytesBase64Encoded'])) {
                $images[] = [
                    'mimeType' => $prediction['mimeType'] ?? 'image/png',
                    'data' => $prediction['bytesBase64Encoded'],
                ];
            }
        }

        // Also check generateImages response format
        foreach ($response['generatedImages'] ?? [] as $image) {
            if (isset($image['image']['imageBytes'])) {
                $images[] = [
                    'mimeType' => 'image/png',
                    'data' => $image['image']['imageBytes'],
                ];
            }
        }

        return ['images' => $images];
    }

    /**
     * Edit an existing image using AI.
     *
     * Uses Gemini's image generation models to edit/enhance images.
     * Supports multi-turn editing via thoughtSignature preservation.
     *
     * @param string $imageUrl URL of the source image to edit
     * @param string $prompt Instructions for how to edit the image
     * @param array{
     *     model?: string,
     *     aspectRatio?: string,
     *     imageSize?: int,
     * } $options Edit options
     * @return array{images: array<array{mimeType: string, data: string}>, text: string}
     */
    public function editImage(string $imageUrl, string $prompt, array $options = []): array
    {
        // Fetch the source image with browser-like headers
        $imageData = $this->fetchImage($imageUrl);
        if ($imageData === false) {
            throw new RuntimeException("Failed to fetch image from: {$imageUrl}");
        }

        $base64Image = base64_encode($imageData);
        $mimeType = $this->detectMimeType($imageUrl, $imageData);

        $model = $options['model'] ?? self::MODEL_3_PRO_IMAGE;
        $aspectRatio = $options['aspectRatio'] ?? '1:1';
        $imageSize = $options['imageSize'] ?? '2K';

        $payload = [
            'contents' => [
                [
                    'parts' => [
                        [
                            'inlineData' => [
                                'mimeType' => $mimeType,
                                'data' => $base64Image,
                            ],
                        ],
                        ['text' => $prompt],
                    ],
                ],
            ],
            'generationConfig' => [
                'responseModalities' => ['TEXT', 'IMAGE'],
                'imageConfig' => [
                    'aspectRatio' => $aspectRatio,
                    'imageSize' => $imageSize,
                ],
            ],
        ];

        $url = self::API_BASE . "/{$model}:generateContent?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        return $this->parseImageResponse($response);
    }

    /**
     * Generate embeddings for the given input(s).
     *
     * @param string|array<string> $input One or more texts to embed
     * @param array{
     *     model?: string,
     *     dimensions?: int,
     * } $options Provider-specific options
     */
    public function embed(string|array $input, array $options = []): EmbeddingResponse
    {
        $model = $options['model'] ?? 'text-embedding-004';
        $inputs = is_array($input) ? $input : [$input];

        if (count($inputs) === 1) {
            $url = self::API_BASE . "/{$model}:embedContent?key={$this->apiKey}";
            $payload = [
                'model' => "models/{$model}",
                'content' => [
                    'parts' => [['text' => $inputs[0]]],
                ],
            ];

            $response = $this->embeddingRequest($payload, $url);

            return new EmbeddingResponse(
                embeddings: [$response['embedding']['values']],
                model: $model,
            );
        }

        $url = self::API_BASE . "/{$model}:batchEmbedContents?key={$this->apiKey}";
        $requests = [];
        foreach ($inputs as $text) {
            $requests[] = [
                'model' => "models/{$model}",
                'content' => [
                    'parts' => [['text' => $text]],
                ],
            ];
        }

        $response = $this->embeddingRequest(['requests' => $requests], $url);

        $embeddings = [];
        foreach ($response['embeddings'] as $embedding) {
            $embeddings[] = $embedding['values'];
        }

        return new EmbeddingResponse(
            embeddings: $embeddings,
            model: $model,
        );
    }

    /**
     * Parse an image generation/editing response.
     *
     * Handles Gemini's thought process: parts with "thought": true are
     * intermediate reasoning images. The final output has "thoughtSignature".
     *
     * @return array{images: array<array{mimeType: string, data: string}>, text: string}
     */
    private function parseImageResponse(array $response): array
    {
        $images = [];
        $text = '';

        foreach ($response['candidates'] ?? [] as $candidate) {
            foreach ($candidate['content']['parts'] ?? [] as $part) {
                // Skip intermediate thought images
                if (!empty($part['thought'])) {
                    continue;
                }

                if (isset($part['inlineData'])) {
                    $images[] = [
                        'mimeType' => $part['inlineData']['mimeType'] ?? 'image/png',
                        'data' => $part['inlineData']['data'],
                    ];
                } elseif (isset($part['text'])) {
                    $text .= $part['text'];
                }
            }
        }

        return ['images' => $images, 'text' => $text];
    }

    /**
     * Fetch an image from URL with proper headers.
     */
    protected function fetchImage(string $url): string|false
    {
        $context = stream_context_create([
            'http' => [
                'header' => implode("\r\n", [
                    'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept: image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language: en-US,en;q=0.9',
                ]),
                'timeout' => 30,
            ],
            'ssl' => [
                'verify_peer' => false,
                'verify_peer_name' => false,
            ],
        ]);

        return @file_get_contents($url, false, $context);
    }

    /**
     * Detect MIME type from URL or content.
     */
    private function detectMimeType(string $url, string $data): string
    {
        $extension = strtolower(pathinfo(parse_url($url, PHP_URL_PATH) ?? '', PATHINFO_EXTENSION));

        return match ($extension) {
            'jpg', 'jpeg' => 'image/jpeg',
            'png' => 'image/png',
            'gif' => 'image/gif',
            'webp' => 'image/webp',
            default => 'image/jpeg',
        };
    }

    /**
     * Generate an image and save it to a file.
     *
     * @param string $prompt The image generation prompt
     * @param string $outputPath Path to save the image
     * @param array $options Generation options
     * @return string The path to the saved image
     */
    public function generateImageToFile(string $prompt, string $outputPath, array $options = []): string
    {
        $result = $this->generateImage($prompt, $options);

        if (empty($result['images'])) {
            throw new RuntimeException('No images generated');
        }

        $image = $result['images'][0];
        $data = base64_decode($image['data']);

        if ($data === false) {
            throw new RuntimeException('Failed to decode image data');
        }

        $written = file_put_contents($outputPath, $data);

        if ($written === false) {
            throw new RuntimeException("Failed to write image to: {$outputPath}");
        }

        return $outputPath;
    }

    /**
     * Build the API request payload.
     */
    private function buildPayload(array $messages, array $options): array
    {
        $contents = [];
        $systemInstruction = null;

        foreach ($messages as $message) {
            if ($message instanceof Message) {
                if ($message->isSystem()) {
                    $systemInstruction = $message->getText();
                    continue;
                }

                $contents[] = $this->convertMessage($message);
            }
        }

        $payload = [
            'contents' => $contents,
            'generationConfig' => [
                'maxOutputTokens' => $options['maxTokens'] ?? $this->defaultMaxTokens,
            ],
        ];

        if ($systemInstruction !== null) {
            $payload['systemInstruction'] = [
                'parts' => [['text' => $systemInstruction]],
            ];
        }

        if (isset($options['temperature'])) {
            $payload['generationConfig']['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $payload['generationConfig']['stopSequences'] = $options['stopSequences'];
        }

        // Handle structured output / JSON mode
        if (isset($options['outputSchema'])) {
            $payload['generationConfig']['responseMimeType'] = 'application/json';
            $payload['generationConfig']['responseSchema'] = $options['outputSchema'];
        }

        // Handle tools
        if (isset($options['tools']) && !empty($options['tools'])) {
            $payload['tools'] = [
                ['functionDeclarations' => $this->convertTools($options['tools'])],
            ];
        }

        return $payload;
    }

    /**
     * Convert a Message to Gemini API format.
     */
    private function convertMessage(Message $message): array
    {
        $role = match ($message->role) {
            Role::User, Role::Tool => 'user',
            Role::Assistant => 'model',
            Role::System => 'user',
        };

        $parts = [];

        if ($message->isTool()) {
            // Function response
            $parts[] = [
                'functionResponse' => [
                    'name' => $message->toolCallId, // Gemini uses name, not ID
                    'response' => ['result' => $message->content],
                ],
            ];
        } elseif ($message->hasToolCalls()) {
            // Model message with function calls
            if ($message->getText() !== '') {
                $parts[] = ['text' => $message->getText()];
            }
            foreach ($message->toolCalls as $toolCall) {
                $part = [
                    'functionCall' => [
                        'name' => $toolCall->name,
                        'args' => (object) $toolCall->arguments,
                    ],
                ];
                if (isset($this->thoughtSignatures[$toolCall->id])) {
                    $part['thoughtSignature'] = $this->thoughtSignatures[$toolCall->id];
                }
                $parts[] = $part;
            }
        } elseif (is_array($message->content)) {
            // Multimodal content
            $parts = $this->convertMultimodalContent($message->content);
        } else {
            // Simple text
            $parts[] = ['text' => $message->content];
        }

        return [
            'role' => $role,
            'parts' => $parts,
        ];
    }

    /**
     * Convert multimodal content to Gemini format.
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['text' => $part['text']];
            } elseif ($part['type'] === 'image') {
                $source = $part['source'];
                if ($source['type'] === 'url') {
                    // Gemini supports URL images directly via fileData
                    $parts[] = [
                        'fileData' => [
                            'mimeType' => $source['media_type'] ?? 'image/jpeg',
                            'fileUri' => $source['url'],
                        ],
                    ];
                } else {
                    // Base64 inline data
                    $parts[] = [
                        'inlineData' => [
                            'mimeType' => $source['media_type'],
                            'data' => $source['data'],
                        ],
                    ];
                }
            }
        }

        return $parts;
    }

    /**
     * Convert tools from PapiAI format to Gemini format.
     */
    private function convertTools(array $tools): array
    {
        $declarations = [];

        foreach ($tools as $tool) {
            // Handle both array format (from Tool::toAnthropic) and object
            if (is_array($tool)) {
                $declarations[] = [
                    'name' => $tool['name'],
                    'description' => $tool['description'],
                    'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                ];
            }
        }

        return $declarations;
    }

    /**
     * Parse API response into Response object.
     */
    private function parseResponse(array $response, array $messages): Response
    {
        $candidate = $response['candidates'][0] ?? [];
        $content = $candidate['content'] ?? [];
        $parts = $content['parts'] ?? [];

        $text = '';
        $toolCalls = [];

        foreach ($parts as $part) {
            if (isset($part['text'])) {
                $text .= $part['text'];
            } elseif (isset($part['functionCall'])) {
                $fc = $part['functionCall'];
                $toolCall = new ToolCall(
                    id: $fc['name'], // Gemini uses name as ID
                    name: $fc['name'],
                    arguments: $fc['args'] ?? [],
                );
                if (isset($part['thoughtSignature'])) {
                    $this->thoughtSignatures[$toolCall->id] = $part['thoughtSignature'];
                }
                $toolCalls[] = $toolCall;
            }
        }

        $usage = [];
        if (isset($response['usageMetadata'])) {
            $usage = [
                'input_tokens' => $response['usageMetadata']['promptTokenCount'] ?? 0,
                'output_tokens' => $response['usageMetadata']['candidatesTokenCount'] ?? 0,
            ];
        }

        $stopReason = $candidate['finishReason'] ?? null;

        return new Response(
            text: $text,
            toolCalls: $toolCalls,
            messages: $messages,
            usage: $usage,
            stopReason: $stopReason,
        );
    }

    /**
     * Extract text from a streaming candidate response.
     */
    private function extractTextFromCandidate(array $event): string
    {
        $candidate = $event['candidates'][0] ?? [];
        $parts = $candidate['content']['parts'] ?? [];

        $text = '';
        foreach ($parts as $part) {
            if (isset($part['text'])) {
                $text .= $part['text'];
            }
        }

        return $text;
    }

    /**
     * Make an API request.
     */
    protected function request(string $url, array $payload): array
    {
        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Google API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }

    /**
     * Make an embedding API request.
     */
    protected function embeddingRequest(array $payload, string $url): array
    {
        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        curl_close($ch);

        if ($error !== '') {
            throw new RuntimeException("Google API request failed: {$error}");
        }

        $data = json_decode($response, true);

        if ($httpCode >= 400) {
            $this->throwForStatusCode($httpCode, $data);
        }

        return $data;
    }

    /**
     * Throw the appropriate exception based on HTTP status code.
     *
     * @throws AuthenticationException
     * @throws RateLimitException
     * @throws ProviderException
     */
    protected function throwForStatusCode(int $httpCode, ?array $data): never
    {
        $errorMessage = $data['error']['message'] ?? 'Unknown error';

        if ($httpCode === 401) {
            throw new AuthenticationException(
                $this->getName(),
                $httpCode,
                $data,
            );
        }

        if ($httpCode === 429) {
            throw new RateLimitException(
                $this->getName(),
                statusCode: $httpCode,
                responseBody: $data,
            );
        }

        throw new ProviderException(
            "Google API error ({$httpCode}): {$errorMessage}",
            $this->getName(),
            $httpCode,
            $data,
        );
    }

    /**
     * Make a streaming API request.
     *
     * @return Generator<array>
     */
    protected function streamRequest(string $url, array $payload): Generator
    {
        $ch = curl_init($url);

        $buffer = '';
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);
        curl_close($ch);

        // Parse SSE events
        $lines = explode("\n", $buffer);
        foreach ($lines as $line) {
            $line = trim($line);
            if (str_starts_with($line, 'data: ')) {
                $json = substr($line, 6);
                $event = json_decode($json, true);
                if ($event !== null) {
                    yield $event;
                }
            }
        }
    }
}
