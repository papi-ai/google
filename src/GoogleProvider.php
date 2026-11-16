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
use PapiAI\Core\Contracts\NamedToolSelectableInterface;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Contracts\VideoProviderInterface;
use PapiAI\Core\Effort;
use PapiAI\Core\EmbeddingResponse;
use PapiAI\Core\Exception\AuthenticationException;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Exception\RateLimitException;
use PapiAI\Core\Exception\UnknownEffortException;
use PapiAI\Core\JobStatus;
use PapiAI\Core\Message;
use PapiAI\Core\Response;
use PapiAI\Core\Role;
use PapiAI\Core\StopReason;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Core\ToolChoice;
use PapiAI\Core\VideoResponse;
use RuntimeException;

/**
 * Google Gemini API provider for PapiAI.
 *
 * Bridges PapiAI's core types (Message, Response, ToolCall) with Google's Generative Language
 * API, handling format conversion in both directions. Supports chat completions, streaming,
 * tool calling with thought signatures, vision (multimodal), structured JSON output, image
 * generation and editing, video generation via Veo, and text embeddings.
 *
 * Chat, streaming, tools, structured output and image generation go through the Interactions
 * API; embeddings and Veo keep their own endpoints. Authentication is the x-goog-api-key header.
 * All HTTP is done with ext-curl directly, with no HTTP abstraction layer.
 *
 * Supported model families:
 *
 * Gemini 4:
 *   - gemini-4-carbon, gemini-4-argon, gemini-4-hydrogen
 *
 * Gemini 3.x:
 *   - gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite
 *   - gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3-flash-preview
 *
 * Gemini 2.x:
 *   - gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
 *   - gemini-2.0-flash, gemini-2.0-flash-lite
 *
 * Image generation and editing:
 *   - gemini-3.1-flash-image, gemini-3.1-flash-lite-image
 *   - gemini-3-pro-image, gemini-2.5-flash-image
 *
 * Video generation:
 *   - veo-3.1-generate-preview, veo-3.1-lite-generate-preview
 *
 * @see https://ai.google.dev/gemini-api/docs
 *
 * @psalm-import-type ChatOptions from ProviderInterface
 */
class GoogleProvider implements ProviderInterface, ImageProviderInterface, EmbeddingProviderInterface, VideoProviderInterface, NamedToolSelectableInterface
{
    private const API_ROOT = 'https://generativelanguage.googleapis.com/v1beta';
    private const API_BASE = self::API_ROOT . '/models';
    private const INTERACTIONS_URL = self::API_ROOT . '/interactions';

    /**
     * Google's own spelling for the one thinking level whose neutral name differs.
     */
    private const NATIVE_LEVEL = ['extra-high' => 'xhigh'];

    // Gemini model aliases
    public const MODEL_4_CARBON = GeminiModel::Carbon4->value;
    public const MODEL_4_ARGON = GeminiModel::Argon4->value;
    public const MODEL_4_HYDROGEN = GeminiModel::Hydrogen4->value;
    public const MODEL_3_8_FLASH = GeminiModel::Flash38->value;
    /** @deprecated No longer published by Google. Use MODEL_3_8_FLASH. */
    public const MODEL_3_7_FLASH = GeminiModel::Flash37->value;
    public const MODEL_3_6_FLASH = GeminiModel::Flash36->value;
    /** @deprecated No longer published by Google. Use MODEL_3_8_FLASH. */
    public const MODEL_3_5_FLASH = GeminiModel::Flash35->value;
    public const MODEL_3_5_FLASH_LITE = GeminiModel::Flash35Lite->value;
    public const MODEL_3_1_PRO = GeminiModel::Pro31->value;
    /** @deprecated Shut down 9 March 2026; the alias now redirects to gemini-3.1-pro-preview. */
    public const MODEL_3_0_PRO = GeminiModel::Pro3->value;
    public const MODEL_3_1_FLASH_LITE = GeminiModel::Flash31Lite->value;
    public const MODEL_3_FLASH = GeminiModel::Flash3->value;
    public const MODEL_3_PRO_IMAGE = GeminiModel::Pro3Image->value;
    public const MODEL_2_5_PRO = GeminiModel::Pro25->value;
    public const MODEL_2_5_FLASH = GeminiModel::Flash25->value;
    public const MODEL_2_5_FLASH_LITE = GeminiModel::Flash25Lite->value;
    public const MODEL_2_0_FLASH = GeminiModel::Flash20->value;
    public const MODEL_2_0_FLASH_LITE = GeminiModel::Flash20Lite->value;
    /** @deprecated Retired; no longer published by Google. */
    public const MODEL_1_5_PRO = GeminiModel::Pro15->value;
    /** @deprecated Retired; no longer published by Google. */
    public const MODEL_1_5_FLASH = GeminiModel::Flash15->value;

    // Image generation and editing, all reached through generateContent. Imagen is retired as a
    // product line and its separate predict endpoint went with it; these replace both.
    public const MODEL_NANO_BANANA_2_1 = GeminiModel::NanoBanana21->value;
    public const MODEL_3_1_FLASH_IMAGE = GeminiModel::Flash31Image->value;
    public const MODEL_3_1_FLASH_LITE_IMAGE = GeminiModel::Flash31LiteImage->value;
    public const MODEL_2_5_FLASH_IMAGE = GeminiModel::Flash25Image->value;
    public const MODEL_EMBEDDING_2 = GeminiModel::Embedding2->value;
    public const MODEL_EMBEDDING_001 = GeminiModel::Embedding001->value;

    /** @deprecated Imagen shuts down 17 August 2026. Use MODEL_3_1_FLASH_IMAGE. */
    public const IMAGEN_4 = GeminiModel::Imagen4->value;
    /** @deprecated Imagen shuts down 17 August 2026. Use MODEL_3_1_FLASH_IMAGE. */
    public const IMAGEN_4_ULTRA = GeminiModel::Imagen4Ultra->value;
    /** @deprecated Imagen shuts down 17 August 2026. Use MODEL_3_1_FLASH_IMAGE. */
    public const IMAGEN_4_FAST = GeminiModel::Imagen4Fast->value;
    /** @deprecated Imagen 3 is already shut down. Use MODEL_3_1_FLASH_IMAGE. */
    public const IMAGEN_EDIT = GeminiModel::ImagenEdit->value;

    // Veo model aliases for video generation
    public const MODEL_VEO_3_1 = GeminiModel::Veo31->value;
    public const MODEL_VEO_3_1_LITE = GeminiModel::Veo31Lite->value;
    /** @deprecated Shut down 30 June 2026; requests fail. Use MODEL_VEO_3_1. */
    public const MODEL_VEO_3 = GeminiModel::Veo3->value;
    /** @deprecated Shut down 30 June 2026; requests fail. Use MODEL_VEO_3_1. */
    public const MODEL_VEO_2 = GeminiModel::Veo2->value;

    /**
     * The model steps of each turn that called tools, keyed by call id, so the next request can
     * replay them exactly as received. Interactions requires that in stateless mode.
     *
     * @var array<string, list<array<string, mixed>>>
     */
    private array $turnSteps = [];

    /** @var array<string, string> tool name by call id, which a function_result item must name */
    private array $callNames = [];

    /**
     * Create a new Google Gemini provider instance.
     *
     * @param string $apiKey         Google AI API key for authentication
     * @param string $defaultModel   Gemini model to use when not specified in options
     * @param int    $defaultMaxTokens Maximum output tokens when not specified in options
     * @param Effort|null $defaultEffort Thinking effort when none is given per call
     */
    public function __construct(
        private readonly string $apiKey,
        private readonly string $defaultModel = self::MODEL_3_8_FLASH,
        private readonly int $defaultMaxTokens = 8192,
        private readonly ?Effort $defaultEffort = null,
    ) {
    }

    /**
     * Send a conversation to the Interactions API and return the model's turn.
     *
     * @param array<Message> $messages Conversation history
     * @param ChatOptions     $options  Request options (model, tools, maxTokens, temperature, toolChoice, effort, outputSchema)
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error, or a request option cannot be honoured
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function chat(array $messages, array $options = []): Response
    {
        $response = $this->request($this->interactionsUrl(), $this->buildPayload($messages, $options));

        return $this->parseResponse($response, $messages);
    }

    /**
     * Stream an interaction, yielding text deltas and a final completion marker.
     *
     * @param array<Message> $messages Conversation history
     * @param array<array-key, mixed> $options Request options
     *
     * @return iterable<StreamChunk>
     */
    public function stream(array $messages, array $options = []): iterable
    {
        $payload = $this->buildPayload($messages, $options);
        $payload['stream'] = true;

        foreach ($this->streamRequest($this->interactionsUrl() . '?alt=sse', $payload) as $event) {
            $type = $event['event_type'] ?? '';

            if ($type === 'step.delta' && ($event['delta']['type'] ?? '') === 'text') {
                yield new StreamChunk((string) ($event['delta']['text'] ?? ''));
            }

            if ($type === 'interaction.completed' || $type === 'error') {
                yield new StreamChunk('', isComplete: true);
            }
        }
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
     * Images travel inline as base64 image parts; URLs are fetched first.
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
     * Supported via Gemini's native image models (e.g. gemini-3.1-flash-image).
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
     * Supported via Gemini's native image models (e.g. gemini-3-pro-image) which
     * can accept an image and a text prompt and return a modified image.
     *
     * @return bool Always true for Google
     */
    public function supportsImageEditing(): bool
    {
        return true;
    }

    /**
     * Generate images from a text prompt using Gemini's native image models.
     *
     * Goes through the Interactions API, asking for an image back through response_format. Imagen's separate
     * predict endpoint is gone: the whole Imagen line shuts down on 17 August 2026 and the
     * Gemini image models that replace it do not speak that endpoint.
     *
     * @param string $prompt Descriptive text prompt for image generation
     * @param array{
     *     model?: string,
     *     numberOfImages?: int,
     *     aspectRatio?: string,
     *     imageSize?: string,
     * } $options Generation options (aspect ratio and size default to the model's own)
     *
     * @return array{images: array<array{mimeType: string, data: string}>} Base64-encoded images with MIME types
     *
     * @throws ProviderException       When more than one image is asked for, or the API returns an error
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function generateImage(string $prompt, array $options = []): array
    {
        $numberOfImages = (int) ($options['numberOfImages'] ?? 1);

        // Imagen's sampleCount has no equivalent here, so a caller asking for four images
        // would silently receive one. Refusing is the same posture as Cohere declining to
        // fake a named tool: better a loud failure than a quiet one.
        if ($numberOfImages > 1) {
            throw new ProviderException(
                sprintf(
                    'Gemini image models return one image per request, so "numberOfImages" of %d cannot be honoured. Call generateImage() once per image instead.',
                    $numberOfImages,
                ),
                $this->getName(),
            );
        }

        $result = $this->requestImage(
            [['type' => 'text', 'text' => $prompt]],
            $options['model'] ?? self::MODEL_3_1_FLASH_IMAGE,
            $options,
        );

        return ['images' => $result['images']];
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
     *     imageSize?: string,
     * } $options Edit options (aspect ratio and size default to the source image's own)
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

        $parts = [
            ['type' => 'image', 'mime_type' => $this->detectMimeType($imageUrl, $imageData), 'data' => base64_encode($imageData)],
            ['type' => 'text', 'text' => $prompt],
        ];

        return $this->requestImage($parts, $options['model'] ?? self::MODEL_3_PRO_IMAGE, $options);
    }

    /**
     * Ask a Gemini image model for an image and parse what comes back.
     *
     * Generation and editing differ only in the parts they send. Image output is requested
     * through response_format, which also carries the shape and size knobs.
     *
     * @param list<array<string, mixed>> $parts   The content parts, text alone or an image and text
     * @param string                     $model   The image model to call
     * @param array<string, mixed>       $options The caller's options, read for aspectRatio and imageSize
     *
     * @return array{images: array<array{mimeType: string, data: string}>, text: string}
     */
    private function requestImage(array $parts, string $model, array $options): array
    {
        $format = array_filter([
            'type' => 'image',
            'aspect_ratio' => $options['aspectRatio'] ?? null,
            'image_size' => $options['imageSize'] ?? null,
        ], static fn ($value) => $value !== null);

        $response = $this->request($this->interactionsUrl(), [
            'model' => $model,
            'store' => false,
            'input' => [['type' => 'user_input', 'content' => $parts]],
            'response_format' => $format,
        ]);

        return $this->parseImageResponse($response);
    }

    /**
     * The Interactions endpoint. Protected so a test double can observe it.
     */
    protected function interactionsUrl(): string
    {
        return self::INTERACTIONS_URL;
    }

    /**
     * Whether this provider supports video generation from text prompts.
     *
     * Supported via Google's Veo model family through the predictLongRunning endpoint.
     *
     * @return bool Always true for Google
     */
    public function supportsVideoGeneration(): bool
    {
        return true;
    }

    /**
     * Generate a video from a text prompt using Google's Veo API, blocking until ready.
     *
     * Submits the request, then polls the long-running operation with exponential backoff
     * (1s doubling up to 10s) until Veo finishes, and returns the finished clip.
     *
     * @param string $prompt Descriptive text prompt for video generation
     * @param array{
     *     model?: string,
     *     aspectRatio?: string,
     *     durationSeconds?: int,
     *     resolution?: string,
     *     fps?: int,
     *     image?: string,
     *     imageMimeType?: string,
     *     negativePrompt?: string,
     *     pollTimeoutSeconds?: int,
     * } $options Generation options (image = base64 seed for image-to-video)
     *
     * @return VideoResponse The generated video (bytes downloaded when a URI is returned)
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When generation fails or times out
     * @throws RuntimeException        When a cURL request itself fails
     */
    public function generateVideo(string $prompt, array $options = []): VideoResponse
    {
        $jobId = $this->startVideo($prompt, $options);
        $timeoutSeconds = $options['pollTimeoutSeconds'] ?? 300;
        $delaySeconds = 1;
        $waited = 0;

        while (true) {
            $status = $this->videoStatus($jobId);

            if ($status->isCompleted()) {
                return $this->fetchVideo($jobId);
            }

            if ($status->isFailed()) {
                throw new ProviderException(
                    sprintf('Veo video generation failed: %s.', $status->error ?? 'unknown error'),
                    $this->getName(),
                );
            }

            if ($waited >= $timeoutSeconds) {
                throw new ProviderException(
                    sprintf('Veo video generation timed out after %d seconds.', $timeoutSeconds),
                    $this->getName(),
                );
            }

            $this->pause($delaySeconds);
            $waited += $delaySeconds;
            $delaySeconds = min($delaySeconds * 2, 10);
        }
    }

    /**
     * Submit a Veo video generation request and return the operation name immediately.
     *
     * @param string $prompt  Descriptive text prompt for video generation
     * @param array  $options Generation options (see generateVideo())
     *
     * @return string The long-running operation name, used as the job identifier
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns an error or no operation name
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function startVideo(string $prompt, array $options = []): string
    {
        $model = $options['model'] ?? self::MODEL_VEO_3_1;
        $payload = $this->buildVideoPayload($prompt, $options);

        $url = self::API_BASE . "/{$model}:predictLongRunning?key={$this->apiKey}";
        $response = $this->request($url, $payload);

        $name = $response['name'] ?? '';

        if ($name === '') {
            throw new ProviderException('Veo did not return an operation name.', $this->getName());
        }

        return $name;
    }

    /**
     * Poll the status of a submitted Veo video generation job.
     *
     * @param string $jobId The operation name returned by startVideo()
     *
     * @return JobStatus Current status (running until Veo reports done, then completed/failed)
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function videoStatus(string $jobId): JobStatus
    {
        $operation = $this->getRequest($this->operationUrl($jobId));

        if (isset($operation['error'])) {
            return new JobStatus($jobId, JobStatus::FAILED, null, $operation['error']['message'] ?? 'unknown error');
        }

        if (($operation['done'] ?? false) === true) {
            return new JobStatus($jobId, JobStatus::COMPLETED);
        }

        return new JobStatus($jobId, JobStatus::RUNNING);
    }

    /**
     * Retrieve the finished video for a completed Veo job.
     *
     * @param string $jobId The operation name returned by startVideo()
     *
     * @return VideoResponse The generated video (bytes downloaded when a URI is returned)
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the job is unfinished, failed, or returned no video
     * @throws RuntimeException        When the cURL request itself fails
     */
    public function fetchVideo(string $jobId): VideoResponse
    {
        $operation = $this->getRequest($this->operationUrl($jobId));

        if (isset($operation['error'])) {
            throw new ProviderException(
                sprintf('Veo video generation failed: %s.', $operation['error']['message'] ?? 'unknown error'),
                $this->getName(),
            );
        }

        if (($operation['done'] ?? false) !== true) {
            throw new ProviderException(
                sprintf('Video job "%s" is not complete yet.', $jobId),
                $this->getName(),
            );
        }

        return $this->parseVideoResponse($operation, $jobId);
    }

    /**
     * Build the predictLongRunning request body for a Veo generation.
     *
     * @param string $prompt  The generation prompt
     * @param array  $options Generation options (see generateVideo())
     *
     * @return array The instances/parameters payload
     */
    private function buildVideoPayload(string $prompt, array $options): array
    {
        $instance = ['prompt' => $prompt];

        if (isset($options['image'])) {
            $instance['image'] = [
                'bytesBase64Encoded' => $options['image'],
                'mimeType' => $options['imageMimeType'] ?? 'image/png',
            ];
        }

        $parameters = [];
        foreach (['aspectRatio', 'durationSeconds', 'negativePrompt', 'fps', 'resolution'] as $key) {
            if (isset($options[$key])) {
                $parameters[$key] = $options[$key];
            }
        }

        $payload = ['instances' => [$instance]];

        if ($parameters !== []) {
            $payload['parameters'] = $parameters;
        }

        return $payload;
    }

    /**
     * Extract a VideoResponse from a completed operation payload.
     *
     * Handles the several response shapes Veo has used: inline base64 bytes, a
     * download URI, or a Cloud Storage URI. When only a URI is present the bytes
     * are downloaded so callers receive a usable clip.
     *
     * @param array  $operation The completed long-running operation payload
     * @param string $jobId     The operation name (used to infer the model)
     *
     * @return VideoResponse The generated video
     *
     * @throws ProviderException When the payload contains no video
     */
    private function parseVideoResponse(array $operation, string $jobId): VideoResponse
    {
        $model = $this->videoModelFromOperation($jobId);
        $samples = $operation['response']['generateVideoResponse']['generatedSamples']
            ?? $operation['response']['generateVideoResponse']['generatedVideos']
            ?? $operation['response']['videos']
            ?? [];

        foreach ($samples as $sample) {
            $video = $sample['video'] ?? $sample;

            if (isset($video['bytesBase64Encoded'])) {
                $bytes = base64_decode($video['bytesBase64Encoded'], true);

                if ($bytes !== false) {
                    return new VideoResponse($bytes, null, $video['mimeType'] ?? 'video/mp4', $model);
                }
            }

            $uri = $video['uri'] ?? $video['gcsUri'] ?? null;

            if ($uri !== null) {
                $bytes = $this->downloadVideo($uri);

                if ($bytes !== false) {
                    return new VideoResponse($bytes, $uri, 'video/mp4', $model);
                }

                return VideoResponse::fromUri($uri, $model);
            }
        }

        throw new ProviderException(
            sprintf('Veo returned no video for job "%s".', $jobId),
            $this->getName(),
        );
    }

    /**
     * Infer the Veo model id from an operation name of the form models/{model}/operations/{id}.
     *
     * @param string $jobId The operation name
     *
     * @return string The model id, or an empty string when it cannot be determined
     */
    private function videoModelFromOperation(string $jobId): string
    {
        if (preg_match('#models/([^/]+)/#', $jobId, $matches) === 1) {
            return $matches[1];
        }

        return '';
    }

    /**
     * Build the poll URL for a long-running operation name.
     */
    private function operationUrl(string $jobId): string
    {
        return self::API_ROOT . '/' . ltrim($jobId, '/') . "?key={$this->apiKey}";
    }

    /**
     * Pause between poll attempts. Isolated so tests can override it to no-op.
     *
     * @param int $seconds Seconds to sleep
     */
    protected function pause(int $seconds): void
    {
        sleep($seconds);
    }

    /**
     * Send a synchronous GET request to the Gemini API via cURL.
     *
     * Used to poll long-running operations. Protected so test doubles can stub it.
     *
     * @param string $url Full API endpoint URL including query parameters
     *
     * @return array Decoded JSON response body
     *
     * @throws AuthenticationException When the API key is invalid (HTTP 401)
     * @throws RateLimitException      When rate limits are exceeded (HTTP 429)
     * @throws ProviderException       When the API returns any other error (HTTP 4xx/5xx)
     * @throws RuntimeException        When the cURL request itself fails
     */
    protected function getRequest(string $url): array
    {
        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

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
     * Download the generated video bytes from a Veo file URI.
     *
     * The Gemini file URI requires the API key. Protected so test doubles can stub it.
     *
     * @param string $uri The video file URI returned by the operation
     *
     * @return string|false Raw video bytes, or false when the download fails
     */
    protected function downloadVideo(string $uri): string|false
    {
        $separator = str_contains($uri, '?') ? '&' : '?';
        $url = $uri . $separator . 'key=' . $this->apiKey;

        $ch = curl_init($url);

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => true,
        ]);

        $data = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

        if ($error !== '' || $httpCode >= 400 || !is_string($data)) {
            return false;
        }

        return $data;
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
     * } $options Provider-specific options (defaults: model=gemini-embedding-2, dimensions=3072)
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
        $model = $options['model'] ?? self::MODEL_EMBEDDING_2;
        $dimensions = $options['dimensions'] ?? null;
        $inputs = is_array($input) ? $input : [$input];

        if (count($inputs) === 1) {
            $url = self::API_BASE . "/{$model}:embedContent?key={$this->apiKey}";
            $response = $this->embeddingRequest($this->embeddingRequestFor($model, $inputs[0], $dimensions), $url);

            return new EmbeddingResponse(embeddings: [$response['embedding']['values']], model: $model);
        }

        $url = self::API_BASE . "/{$model}:batchEmbedContents?key={$this->apiKey}";
        $requests = [];

        foreach ($inputs as $text) {
            $requests[] = $this->embeddingRequestFor($model, $text, $dimensions);
        }

        $response = $this->embeddingRequest(['requests' => $requests], $url);
        $embeddings = [];

        foreach ($response['embeddings'] as $embedding) {
            $embeddings[] = $embedding['values'];
        }

        return new EmbeddingResponse(embeddings: $embeddings, model: $model);
    }

    /**
     * One embedContent request, with the output size when the caller asks for one.
     *
     * @return array<string, mixed>
     */
    private function embeddingRequestFor(string $model, string $text, ?int $dimensions): array
    {
        $request = ['model' => "models/{$model}", 'content' => ['parts' => [['text' => $text]]]];

        if ($dimensions !== null) {
            $request['outputDimensionality'] = $dimensions;
        }

        return $request;
    }

    /**
     * Usage in core's vocabulary. Interactions counts thought tokens apart from the visible
     * output; both are output tokens for billing, so they are summed here as every other provider
     * already reports them.
     *
     * @param array<string, mixed> $usage
     *
     * @return array<string, int>
     */
    private function usageOf(array $usage): array
    {
        $normalised = [
            'input_tokens' => (int) ($usage['total_input_tokens'] ?? 0),
            'output_tokens' => (int) ($usage['total_output_tokens'] ?? 0) + (int) ($usage['total_thought_tokens'] ?? 0),
        ];

        if (isset($usage['total_tokens'])) {
            $normalised['total_tokens'] = (int) $usage['total_tokens'];
        }

        return $normalised;
    }

    /**
     * "stop" for a finished turn, "tool_calls" when the model wants tools run, otherwise the
     * status Google reported, which is "incomplete" when the output ceiling cut the turn short.
     *
     * @param array<string, mixed> $response
     */
    private function stopReasonOf(array $response, bool $calledTools): string
    {
        if ($calledTools) {
            return 'tool_calls';
        }

        $status = (string) ($response['status'] ?? 'completed');

        return $status === 'completed' ? 'stop' : $status;
    }

    /**
     * Images and text from the model_output steps of an interaction.
     *
     * @param array<string, mixed> $response Raw decoded body
     *
     * @return array{images: array<array{mimeType: string, data: string}>, text: string}
     */
    private function parseImageResponse(array $response): array
    {
        $images = [];
        $text = '';

        foreach ($response['steps'] ?? [] as $step) {
            if (($step['type'] ?? '') !== 'model_output') {
                continue;
            }

            foreach ($step['content'] ?? [] as $part) {
                if (($part['type'] ?? '') === 'image' && isset($part['data'])) {
                    $images[] = ['mimeType' => $part['mime_type'] ?? 'image/png', 'data' => $part['data']];
                } elseif (($part['type'] ?? '') === 'text') {
                    $text .= $part['text'] ?? '';
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
     *     imageSize?: string,
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
     * Build the Interactions request body.
     *
     * The history travels in full on every request (`store: false`): papi owns the transcript,
     * Google does not keep one.
     *
     * @param array<Message> $messages
     * @param ChatOptions    $options
     *
     * @return array<string, mixed>
     */
    private function buildPayload(array $messages, array $options): array
    {
        $model = (string) ($options['model'] ?? $this->defaultModel);
        [$instruction, $input] = $this->splitConversation($messages);

        $payload = ['model' => $model, 'store' => false, 'input' => $input];

        if ($instruction !== null) {
            $payload['system_instruction'] = $instruction;
        }

        if (isset($options['outputSchema'])) {
            $payload['response_format'] = ['type' => 'text', 'mime_type' => 'application/json', 'schema' => $options['outputSchema']];
        }

        if (!empty($options['tools'])) {
            $payload['tools'] = $this->convertTools($options['tools']);
        }

        $payload['generation_config'] = $this->generationConfig($model, $options);

        return $payload;
    }

    /**
     * The system message, which Interactions takes as `system_instruction`, and everything else
     * as input items.
     *
     * @param array<Message> $messages
     *
     * @return array{0: string|null, 1: list<array<string, mixed>>}
     */
    private function splitConversation(array $messages): array
    {
        $instruction = null;
        $input = [];

        foreach ($messages as $message) {
            if (!$message instanceof Message) {
                continue;
            }

            if ($message->isSystem()) {
                $instruction = $message->getText();

                continue;
            }

            foreach ($this->toInputItems($message) as $item) {
                $input[] = $item;
            }
        }

        return [$instruction, $input];
    }

    /**
     * The generation_config block: output ceiling, sampling, stop sequences, forced tool choice
     * and thinking level.
     *
     * @return array<string, mixed>
     */
    private function generationConfig(string $model, array $options): array
    {
        $config = ['max_output_tokens' => $options['maxTokens'] ?? $this->defaultMaxTokens];

        if (isset($options['temperature'])) {
            $this->assertAcceptsSampling($model);
            $config['temperature'] = $options['temperature'];
        }

        if (isset($options['stopSequences'])) {
            $config['stop_sequences'] = $options['stopSequences'];
        }

        // Validation lives in core and throws before any HTTP call, tools or no tools.
        if (isset($options['toolChoice'])) {
            $choice = ToolChoice::fromOption($options['toolChoice'], $options['tools'] ?? []);

            if (!empty($options['tools'])) {
                $config['tool_choice'] = $this->toolChoiceFor($choice);
            }
        }

        $effort = $this->effortFor($options);

        if ($effort !== null) {
            $level = $this->thinkingLevelFor($effort, $model);

            if ($level !== null) {
                $config['thinking_level'] = $level;
            }
        }

        return $config;
    }

    /**
     * Interactions takes the bare mode, or allowed_tools when one tool is named.
     *
     * @return array{allowed_tools: array{mode: string, tools: list<string>}}|string
     */
    private function toolChoiceFor(ToolChoice $choice): array|string
    {
        if ($choice->toolName !== null) {
            return ['allowed_tools' => ['mode' => 'any', 'tools' => [$choice->toolName]]];
        }

        return match ($choice->mode) {
            ToolChoice::NONE => 'none',
            ToolChoice::REQUIRED => 'any',
            default => 'auto',
        };
    }

    /**
     * The thinking_level for a level of effort, or null to send nothing.
     *
     * No Gemini model can be told to stop thinking through a level, so "none" narrows to the
     * shallowest level the model offers. The one model that is off by default, 2.5 Flash-Lite,
     * gets nothing for "none", which leaves it off.
     */
    private function thinkingLevelFor(Effort $effort, string $model): ?string
    {
        $known = GeminiModel::tryFrom($model);

        if (!$effort->thinks() && $known !== null && !$known->thinksByDefault()) {
            return null;
        }

        $level = $effort->nearestOf($this->levelsFor($model))->value;

        return self::NATIVE_LEVEL[$level] ?? $level;
    }

    /**
     * Refuse temperature on the models that ignore it and warn, rather than send a dead knob.
     *
     * @throws ProviderException When the model does not honour sampling parameters
     */
    private function assertAcceptsSampling(string $model): void
    {
        if (GeminiModel::tryFrom($model)?->acceptsSampling() ?? false) {
            return;
        }

        throw new ProviderException(
            sprintf('Model "%s" ignores "temperature": Gemini 3.6 and later return a deprecation warning for sampling parameters. Remove it and let the model use its own default.', $model),
            $this->getName(),
        );
    }

    /**
     * The effort this request asks for: the per-call option, else the provider default.
     *
     * @param array<string, mixed> $options The caller's request options
     *
     * @throws UnknownEffortException When the level is not one core defines
     */
    private function effortFor(array $options): ?Effort
    {
        if (!isset($options['effort'])) {
            return $this->defaultEffort;
        }

        $level = (string) $options['effort'];

        return Effort::tryFrom($level) ?? throw new UnknownEffortException($level);
    }

    /**
     * The thinking levels a model accepts. The model answers when we know it; an unknown ID is
     * assumed to take the three middle levels, which every current generation does.
     *
     * @return non-empty-list<Effort>
     */
    private function levelsFor(string $model): array
    {
        $known = GeminiModel::tryFrom($model)?->effortLevels() ?? [];

        return $known !== [] ? $known : [Effort::Low, Effort::Medium, Effort::High];
    }

    /**
     * Translate one PapiAI message into Interactions input items.
     *
     * A user turn is a user_input item. An assistant turn that called tools is replayed as the
     * model steps this instance received for it, exactly as received, which is what stateless
     * mode requires; a turn it never saw is rebuilt from the message. A tool result is a
     * function_result item naming the tool and the call.
     *
     * @return list<array<string, mixed>>
     */
    private function toInputItems(Message $message): array
    {
        if ($message->isTool()) {
            $callId = (string) $message->toolCallId;
            $result = is_string($message->content) ? $message->content : (string) json_encode($message->content);

            return [[
                'type' => 'function_result',
                'name' => $this->callNames[$callId] ?? $callId,
                'call_id' => $callId,
                'result' => [['type' => 'text', 'text' => $result]],
            ]];
        }

        if ($message->hasToolCalls()) {
            foreach ($message->toolCalls as $toolCall) {
                if (isset($this->turnSteps[$toolCall->id])) {
                    return $this->turnSteps[$toolCall->id];
                }
            }

            $items = [];

            if ($message->getText() !== '') {
                $items[] = ['type' => 'model_output', 'content' => [['type' => 'text', 'text' => $message->getText()]]];
            }

            foreach ($message->toolCalls as $toolCall) {
                $items[] = ['type' => 'function_call', 'id' => $toolCall->id, 'name' => $toolCall->name, 'arguments' => $toolCall->argumentsAsObject()];
            }

            return $items;
        }

        if ($message->role === Role::Assistant) {
            return [['type' => 'model_output', 'content' => [['type' => 'text', 'text' => $message->getText()]]]];
        }

        $content = is_array($message->content)
            ? $this->convertMultimodalContent($message->content)
            : [['type' => 'text', 'text' => $message->content]];

        return [['type' => 'user_input', 'content' => $content]];
    }

    /**
     * Convert multimodal parts to Interactions content parts. Images travel inline as base64; a
     * URL is fetched first.
     *
     * @param array<array{type: string, text?: string, source?: array}> $content
     *
     * @return list<array<string, string>>
     *
     * @throws RuntimeException When an image URL cannot be fetched
     */
    private function convertMultimodalContent(array $content): array
    {
        $parts = [];

        foreach ($content as $part) {
            if ($part['type'] === 'text') {
                $parts[] = ['type' => 'text', 'text' => $part['text']];

                continue;
            }

            if ($part['type'] !== 'image') {
                continue;
            }

            $source = $part['source'];

            if ($source['type'] === 'url') {
                $bytes = $this->fetchImage($source['url']);

                if ($bytes === false) {
                    throw new RuntimeException(sprintf('Failed to fetch image from "%s".', $source['url']));
                }

                $parts[] = ['type' => 'image', 'mime_type' => $source['media_type'] ?? $this->detectMimeType($source['url'], $bytes), 'data' => base64_encode($bytes)];

                continue;
            }

            $parts[] = ['type' => 'image', 'mime_type' => $source['media_type'], 'data' => $source['data']];
        }

        return $parts;
    }

    /**
     * Convert PapiAI tool definitions to Interactions function tools.
     *
     * @param array<array{name: string, description: string, input_schema?: array, parameters?: array}> $tools
     *
     * @return list<array{type: string, name: string, description: string, parameters: array}>
     */
    private function convertTools(array $tools): array
    {
        $declared = [];

        foreach ($tools as $tool) {
            if (is_array($tool)) {
                $declared[] = [
                    'type' => 'function',
                    'name' => $tool['name'],
                    'description' => $tool['description'],
                    'parameters' => $tool['input_schema'] ?? $tool['parameters'] ?? ['type' => 'object', 'properties' => []],
                ];
            }
        }

        return $declared;
    }

    /**
     * Read the model's turn out of an interaction: text from model_output steps, calls from
     * function_call steps. Thought steps are kept only for replay.
     *
     * @param array<string, mixed> $response Raw decoded body
     * @param array<Message>       $messages The conversation this answers
     */
    private function parseResponse(array $response, array $messages): Response
    {
        $text = '';
        $toolCalls = [];
        $steps = $response['steps'] ?? [];

        foreach ($steps as $step) {
            $type = $step['type'] ?? '';

            if ($type === 'model_output') {
                foreach ($step['content'] ?? [] as $part) {
                    if (($part['type'] ?? '') === 'text') {
                        $text .= $part['text'] ?? '';
                    }
                }
            }

            if ($type === 'function_call') {
                $arguments = $step['arguments'] ?? [];
                $toolCall = new ToolCall(
                    id: (string) ($step['id'] ?? $step['name'] ?? ''),
                    name: (string) ($step['name'] ?? ''),
                    arguments: is_array($arguments) ? $arguments : (json_decode((string) $arguments, true) ?? []),
                );
                $this->callNames[$toolCall->id] = $toolCall->name;
                $this->turnSteps[$toolCall->id] = $this->replayable($steps);
                $toolCalls[] = $toolCall;
            }
        }

        return new Response(
            text: $text,
            toolCalls: $toolCalls,
            messages: $messages,
            usage: $this->usageOf($response['usage'] ?? []),
            stopReason: $this->stopReasonOf($response, $toolCalls !== []),
            stoppedBecause: $this->stoppedBecauseOf($response, $toolCalls !== []),
        );
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
                'x-goog-api-key: ' . $this->apiKey,
            ],
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);

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
                'x-goog-api-key: ' . $this->apiKey,
            ],
            CURLOPT_WRITEFUNCTION => function ($ch, $data) use (&$buffer) {
                $buffer .= $data;

                return strlen($data);
            },
        ]);

        curl_exec($ch);

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

    /**
     * Why the model stopped, in core's vocabulary. Google reports a turn cut short by the output
     * ceiling as "incomplete", or as "requires_continuation" when it can be resumed.
     *
     * @param array<string, mixed> $response
     */
    private function stoppedBecauseOf(array $response, bool $calledTools): StopReason
    {
        if ($calledTools) {
            return StopReason::ToolCalls;
        }

        return match ($response['status'] ?? 'completed') {
            'completed' => StopReason::Stop,
            'incomplete', 'requires_continuation' => StopReason::MaxTokens,
            default => StopReason::Other,
        };
    }

    /**
     * A turn's steps as they must go back: exactly as received, except that a function call's
     * arguments stay an object when the model passed none, which decoding turns into a list.
     *
     * @param array<array-key, mixed> $steps
     *
     * @return list<mixed>
     */
    private function replayable(array $steps): array
    {
        return array_map(
            static fn (mixed $step): mixed => is_array($step) && ($step['type'] ?? '') === 'function_call' && is_array($step['arguments'] ?? null)
                ? array_replace($step, ['arguments' => (object) $step['arguments']])
                : $step,
            array_values($steps),
        );
    }
}
