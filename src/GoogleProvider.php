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
use PapiAI\Core\Contracts\ProviderInterface;
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
 * - gemini-2.0-flash-exp (fast, multimodal)
 *
 * Gemini 1.5:
 * - gemini-1.5-pro (proven quality)
 * - gemini-1.5-flash (fast, cost-effective)
 */
final class GoogleProvider implements ProviderInterface
{
    private const API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models';

    // Gemini model aliases
    public const MODEL_3_1_PRO = 'gemini-3.1-pro';
    public const MODEL_3_0_PRO = 'gemini-3.0-pro';
    public const MODEL_2_0_FLASH = 'gemini-2.0-flash-exp';
    public const MODEL_1_5_PRO = 'gemini-1.5-pro';
    public const MODEL_1_5_FLASH = 'gemini-1.5-flash';

    // Imagen model aliases for image generation
    public const IMAGEN_3 = 'imagen-3.0-generate-001';
    public const IMAGEN_3_FAST = 'imagen-3.0-fast-generate-001';

    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_3_0_PRO,
        private readonly int $defaultMaxTokens = 8192,
    ) {}

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

    /**
     * Generate an image using Google's Imagen API.
     *
     * @param string $prompt The image generation prompt
     * @param array{
     *     model?: string,
     *     numberOfImages?: int,
     *     aspectRatio?: string,
     *     negativePrompt?: string,
     *     personGeneration?: string,
     * } $options Generation options
     * @return array{
     *     images: array<array{
     *         mimeType: string,
     *         data: string,
     *     }>,
     * }
     */
    public function generateImage(string $prompt, array $options = []): array
    {
        $model = $options['model'] ?? self::IMAGEN_3;
        $numberOfImages = $options['numberOfImages'] ?? 1;
        $aspectRatio = $options['aspectRatio'] ?? '1:1';

        $payload = [
            'instances' => [
                ['prompt' => $prompt],
            ],
            'parameters' => [
                'sampleCount' => $numberOfImages,
                'aspectRatio' => $aspectRatio,
            ],
        ];

        if (isset($options['negativePrompt'])) {
            $payload['parameters']['negativePrompt'] = $options['negativePrompt'];
        }

        // Allow/disallow person generation (default: allow_adult)
        $payload['parameters']['personGeneration'] = $options['personGeneration'] ?? 'allow_adult';

        $url = self::API_BASE . "/{$model}:predict?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        $images = [];
        foreach ($response['predictions'] ?? [] as $prediction) {
            if (isset($prediction['bytesBase64Encoded'])) {
                $images[] = [
                    'mimeType' => $prediction['mimeType'] ?? 'image/png',
                    'data' => $prediction['bytesBase64Encoded'],
                ];
            }
        }

        return ['images' => $images];
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
                $parts[] = [
                    'functionCall' => [
                        'name' => $toolCall->name,
                        'args' => $toolCall->arguments,
                    ],
                ];
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
                $toolCalls[] = new ToolCall(
                    id: $fc['name'], // Gemini uses name as ID
                    name: $fc['name'],
                    arguments: $fc['args'] ?? [],
                );
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
    private function request(string $url, array $payload): array
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
            $errorMessage = $data['error']['message'] ?? 'Unknown error';
            throw new RuntimeException("Google API error ({$httpCode}): {$errorMessage}");
        }

        return $data;
    }

    /**
     * Make a streaming API request.
     *
     * @return Generator<array>
     */
    private function streamRequest(string $url, array $payload): Generator
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
