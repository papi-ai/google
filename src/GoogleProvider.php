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
 * Google Gemini API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with Google's Generative Language
 * API, handling format conversion in both directions. Supports chat completions, streaming,
 * tool calling with thought signatures, vision (multimodal), structured JSON output, image
 * generation/editing via Imagen, and text embeddings.
 *
 * Authentication is via API key passed as a query parameter. All HTTP is done with ext-curl
 * directly, with no HTTP abstraction layer.
 *
 * Supported model families:
 *
 * Gemini 3.x (Latest):
 *   - gemini-3.1-pro, gemini-3.0-pro, gemini-3-flash, gemini-3-pro-image
 *
 * Gemini 2.x:
 *   - gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
 *   - gemini-2.0-flash, gemini-2.0-flash-lite
 *
 * Gemini 1.5:
 *   - gemini-1.5-pro, gemini-1.5-flash
 *
 * Imagen (image generation):
 *   - imagen-4.0-generate-001, imagen-4.0-ultra-generate-001
 *   - imagen-4.0-fast-generate-001, imagen-3.0-capability-001
 *
 * @see https://ai.google.dev/gemini-api/docs
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

    /** @var array<string, string> tool call ID to thought signature mapping for multi-turn tool use */
    private array $thoughtSignatures = [];

    /**
     * Create a new Google Gemini provider instance.
     *
     * @param string $apiKey         Google AI API key for authentication
     * @param string $defaultModel   Gemini model to use when not specified in options
     * @param int    $defaultMaxTokens Maximum output tokens when not specified in options
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_3_0_PRO,
        private readonly int $defaultMaxTokens = 8192,
    ) {
    }

    /**
     * Send a chat completion request to the Gemini API.
     *
     * Converts PapiAI Messages to Gemini's content format, sends the request,
     * and parses the response back into a core Response object. Supports tools,
     * vision, structured output, and custom generation parameters.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return Response Parsed response containing text, tool calls, usage, and stop reason
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function chat(array $messages, array $options = []): Response
    {
        $model = $options['model'] ?? $this->defaultModel;
        $payload = $this->buildPayload($messages, $options);

        $url = self::API_BASE . "/{$model}:generateContent?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        return $this->parseResponse($response, $messages);
    }

    /**
     * Stream a chat completion from the Gemini API using server-sent events.
     *
     * Yields StreamChunk objects as partial responses arrive. The final chunk
     * has isComplete=true. Only text content is streamed; tool calls are not
     * supported in streaming mode.
     *
     * @param array<Message> $messages Conversation history as PapiAI Message objects
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options (model, tools, maxTokens, temperature, etc.)
     *
     * @return iterable<StreamChunk> Stream of text chunks, ending with a completion marker
     *
     * @throws RuntimeException When the cURL request fails
     */
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

    /**
     * Whether this provider supports function/tool calling.
     *
     * Gemini supports tool calling with function declarations and thought signatures
     * for multi-turn tool use conversations.
     *
     * @return bool Always true for Google Gemini
     */
    public function supportsTool(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports vision (image inputs in messages).
     *
     * Gemini natively handles images via inlineData (base64) and fileData (URLs).
     *
     * @return bool Always true for Google Gemini
     */
    public function supportsVision(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports structured JSON output with a schema.
     *
     * Gemini supports JSON mode via responseMimeType and responseSchema in generation config.
     *
     * @return bool Always true for Google Gemini
     */
    public function supportsStructuredOutput(): bool
    {
        return true; // Gemini supports JSON mode
    }

    /**
     * Get the unique identifier for this provider.
     *
     * Used for error reporting and provider selection in multi-provider setups.
     *
     * @return string The provider name "google"
     */
    public function getName(): string
    {
        return 'google';
    }

    /**
     * Whether this provider supports image generation from text prompts.
     *
     * Supported via Google's Imagen 4 model family through the predict endpoint.
     *
     * @return bool Always true for Google
     */
    public function supportsImageGeneration(): bool
    {
        return true;
    }

    /**
     * Whether this provider supports AI-powered image editing.
     *
     * Supported via Gemini's multimodal models (e.g., gemini-3-pro-image) which
     * can accept an image + text prompt and return a modified image.
     *
     * @return bool Always true for Google
     */
    public function supportsImageEditing(): bool
    {
        return true;
    }

    /**
     * Generate images from a text prompt using Google's Imagen 4 API.
     *
     * Sends the prompt to the Imagen predict endpoint and parses the response,
     * handling both the "predictions" and "generatedImages" response formats.
     *
     * @param string $prompt Descriptive text prompt for image generation
     * @param array{
     *     model?: string,
     *     numberOfImages?: int,
     *     aspectRatio?: string,
     *     imageSize?: int,
     * } $options Generation options (defaults: model=imagen-4.0-fast, 1 image, 1:1 ratio)
     *
     * @return array{images: array<array{mimeType: string, data: string}>} Base64-encoded images with MIME types
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
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
     * Edit an existing image using Gemini's multimodal generation.
     *
     * Fetches the source image, encodes it as base64 inline data, and sends it
     * alongside the edit prompt to a Gemini image model. The response may contain
     * both modified images and descriptive text. Intermediate "thought" images from
     * the model's reasoning process are filtered out.
     *
     * @param string $imageUrl URL of the source image to fetch and edit
     * @param string $prompt   Natural language instructions for how to modify the image
     * @param array{
     *     model?: string,
     *     aspectRatio?: string,
     *     imageSize?: int,
     * } $options Edit options (defaults: model=gemini-3-pro-image, 1:1 ratio, 2K size)
     *
     * @return array{images: array<array{mimeType: string, data: string}>, text: string} Edited images and any descriptive text
     *
     * @throws RuntimeException        When the source image cannot be fetched
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
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
     * Generate text embeddings using Google's embedding models.
     *
     * For a single input, uses the embedContent endpoint. For multiple inputs,
     * uses batchEmbedContents for efficiency. Returns float vectors suitable
     * for similarity search, clustering, and classification tasks.
     *
     * @param string|array<string> $input One or more texts to generate embeddings for
     * @param array{
     *     model?: string,
     *     dimensions?: int,
     * } $options Provider-specific options (defaults: model=text-embedding-004)
     *
     * @return EmbeddingResponse Embedding vectors and model metadata
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
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
     * Parse a Gemini image generation/editing response into a normalized format.
     *
     * Filters out intermediate "thought" images (parts with thought=true) that
     * represent the model's internal reasoning, keeping only final output images
     * and any accompanying text.
     *
     * @param array $response Raw decoded JSON response from the Gemini API
     *
     * @return array{images: array<array{mimeType: string, data: string}>, text: string} Extracted images and text
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
     * Fetch an image from a URL with browser-like headers to avoid blocking.
     *
     * Uses file_get_contents with a custom stream context that mimics a real
     * browser request. SSL verification is disabled to handle self-signed certs.
     *
     * @param string $url The image URL to fetch
     *
     * @return string|false Raw image binary data, or false on failure
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
     * Detect the MIME type of an image from its URL file extension.
     *
     * Falls back to image/jpeg for unrecognized extensions, which covers
     * most web images. Supports JPEG, PNG, GIF, and WebP.
     *
     * @param string $url  The image URL to extract the extension from
     * @param string $data Raw image data (reserved for future content-based detection)
     *
     * @return string The detected MIME type (e.g., "image/png")
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
     * Generate an image and save it directly to a file on disk.
     *
     * Convenience method that calls generateImage() and writes the first
     * resulting image to the specified path. Useful for CLI tools and
     * batch processing where you want the image on disk immediately.
     *
     * @param string $prompt     Descriptive text prompt for image generation
     * @param string $outputPath Filesystem path where the image will be saved
     * @param array{
     *     model?: string,
     *     numberOfImages?: int,
     *     aspectRatio?: string,
     *     imageSize?: int,
     * } $options Generation options passed through to generateImage()
     *
     * @return string The output path where the image was saved
     *
     * @throws RuntimeException        When no images are generated, decoding fails, or file write fails
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
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
     * Build the Gemini API request payload from messages and options.
     *
     * Converts the PapiAI message array into Gemini's contents format, extracts
     * system instructions into a separate field, and applies generation config
     * options (maxTokens, temperature, stop sequences, JSON schema, tools).
     *
     * @param array<Message> $messages Conversation messages to convert
     * @param array{
     *     model?: string,
     *     tools?: array,
     *     maxTokens?: int,
     *     temperature?: float,
     *     stopSequences?: array<string>,
     *     outputSchema?: array,
     * } $options Request options controlling generation behavior
     *
     * @return array The complete Gemini API payload ready for JSON encoding
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
     * Convert a single PapiAI Message into Gemini's content format.
     *
     * Handles all message types: tool responses (functionResponse), assistant
     * messages with tool calls (functionCall with optional thoughtSignature),
     * multimodal content (images + text), and plain text messages. Maps PapiAI
     * roles to Gemini roles (User/Tool become "user", Assistant becomes "model").
     *
     * @param Message $message The PapiAI message to convert
     *
     * @return array Gemini content object with "role" and "parts" keys
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
     * Convert multimodal content parts (text + images) to Gemini format.
     *
     * Translates PapiAI's content array format into Gemini's parts format.
     * Images are converted to either fileData (for URLs) or inlineData (for base64).
     *
     * @param array<array{type: string, text?: string, source?: array}> $content Multimodal content parts
     *
     * @return array<array> Gemini-formatted parts array
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
     * Convert PapiAI tool definitions to Gemini's functionDeclarations format.
     *
     * Maps each tool's name, description, and input schema (JSON Schema) into
     * the structure expected by Gemini's tools API. Only array-format tools
     * are supported; object-format tools are skipped.
     *
     * @param array<array{name: string, description: string, input_schema?: array, parameters?: array}> $tools PapiAI tool definitions
     *
     * @return array<array{name: string, description: string, parameters: array}> Gemini function declarations
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
     * Parse a Gemini API response into a PapiAI Response object.
     *
     * Extracts text content, function calls (with thought signatures for multi-turn),
     * token usage metadata, and the finish reason from the first candidate. Tool call
     * thought signatures are cached in $this->thoughtSignatures so they can be
     * re-attached when converting the assistant's tool call message back to Gemini format.
     *
     * @param array          $response Raw decoded JSON response from the Gemini API
     * @param array<Message> $messages Original conversation messages (passed through to Response)
     *
     * @return Response Parsed response with text, tool calls, usage stats, and stop reason
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
     * Extract concatenated text from a single streaming SSE event.
     *
     * Iterates through all parts of the first candidate and concatenates any
     * text parts. Non-text parts (e.g., function calls) are ignored during streaming.
     *
     * @param array $event A single decoded SSE event from the streaming response
     *
     * @return string The extracted text, or empty string if no text parts exist
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
     * Send a synchronous POST request to the Gemini API via cURL.
     *
     * Encodes the payload as JSON, sends it to the given URL, and decodes the
     * JSON response. Delegates error handling to throwForStatusCode() for any
     * HTTP 4xx/5xx responses. Protected to allow test doubles to intercept requests.
     *
     * @param string $url     Full API endpoint URL including query parameters
     * @param array  $payload Request body to be JSON-encoded
     *
     * @return array Decoded JSON response body
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails (network error, timeout, etc.)
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
     * Send a synchronous POST request to the Gemini embedding API via cURL.
     *
     * Functionally identical to request() but kept separate so test doubles
     * can independently stub embedding vs. generation endpoints. Protected
     * to allow test subclasses to override.
     *
     * @param array  $payload Request body to be JSON-encoded
     * @param string $url     Full embedding API endpoint URL including query parameters
     *
     * @return array Decoded JSON response body containing embedding vectors
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails (network error, timeout, etc.)
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
     * Map an HTTP error status code to the appropriate PapiAI exception.
     *
     * Translates Google API errors into PapiAI's exception hierarchy so callers
     * can handle authentication failures, rate limits, and general errors uniformly
     * across all providers. This method never returns (marked as `never`).
     *
     * @param int        $httpCode HTTP status code (4xx or 5xx)
     * @param array|null $data     Decoded JSON error response body (may be null if response was not valid JSON)
     *
     * @throws AuthenticationException When the API key is invalid or missing (HTTP 401)
     * @throws RateLimitException      When quota or rate limits are exceeded (HTTP 429)
     * @throws ProviderException       For all other API errors (HTTP 4xx/5xx)
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
     * Send a streaming POST request and yield parsed SSE events.
     *
     * Buffers the entire SSE response (Gemini streams via `alt=sse` query param),
     * then parses each `data:` line as JSON and yields the decoded events. Protected
     * to allow test doubles to provide pre-built event sequences.
     *
     * @param string $url     Full API endpoint URL with alt=sse for streaming
     * @param array  $payload Request body to be JSON-encoded
     *
     * @return Generator<int, array> Yields decoded JSON objects, one per SSE data line
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
