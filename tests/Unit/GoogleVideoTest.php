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

use PapiAI\Core\Contracts\VideoProviderInterface;
use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\VideoResponse;
use PapiAI\Google\GoogleProvider;

/**
 * Test subclass that stubs the HTTP + sleep seams for Veo unit testing.
 */
class TestableVeoProvider extends GoogleProvider
{
    public string $lastUrl = '';
    public array $lastPayload = [];
    public array $startResponse = ['name' => 'models/veo-3.1-generate-preview/operations/abc123'];

    /** @var array<int, array> Queue of getRequest() responses, consumed in order */
    public array $pollResponses = [];
    public string $fakeVideoBytes = 'fake-video-binary';
    public bool $downloadShouldFail = false;
    public int $pauseCalls = 0;

    protected function request(string $url, array $payload): array
    {
        $this->lastUrl = $url;
        $this->lastPayload = $payload;

        return $this->startResponse;
    }

    protected function getRequest(string $url): array
    {
        $this->lastUrl = $url;

        return array_shift($this->pollResponses) ?? [];
    }

    protected function downloadVideo(string $uri): string|false
    {
        return $this->downloadShouldFail ? false : $this->fakeVideoBytes;
    }

    protected function pause(int $seconds): void
    {
        ++$this->pauseCalls;
    }
}

describe('GoogleProvider video generation', function () {
    beforeEach(function () {
        $this->provider = new TestableVeoProvider('test-api-key');
    });

    describe('capabilities', function () {
        it('implements VideoProviderInterface', function () {
            expect($this->provider)->toBeInstanceOf(VideoProviderInterface::class);
        });

        it('supports video generation', function () {
            expect($this->provider->supportsVideoGeneration())->toBeTrue();
        });
    });

    describe('startVideo', function () {
        it('posts to the predictLongRunning endpoint and returns the operation name', function () {
            $jobId = $this->provider->startVideo('a cat surfing');

            expect($jobId)->toBe('models/veo-3.1-generate-preview/operations/abc123');
            expect($this->provider->lastUrl)
                ->toContain('veo-3.1-generate-preview:predictLongRunning')
                ->toContain('key=test-api-key');
            expect($this->provider->lastPayload['instances'][0]['prompt'])->toBe('a cat surfing');
        });

        it('passes generation parameters and an image seed', function () {
            $this->provider->startVideo('a dog', [
                'aspectRatio' => '16:9',
                'durationSeconds' => 8,
                'negativePrompt' => 'blurry',
                'image' => 'base64seed',
            ]);

            $payload = $this->provider->lastPayload;
            expect($payload['parameters']['aspectRatio'])->toBe('16:9');
            expect($payload['parameters']['durationSeconds'])->toBe(8);
            expect($payload['parameters']['negativePrompt'])->toBe('blurry');
            expect($payload['instances'][0]['image']['bytesBase64Encoded'])->toBe('base64seed');
        });

        it('honours a custom model', function () {
            $this->provider->startVideo('a bird', ['model' => GoogleProvider::MODEL_VEO_3_1]);

            expect($this->provider->lastUrl)->toContain('veo-3.1-generate-preview:predictLongRunning');
        });

        it('throws when no operation name is returned', function () {
            $this->provider->startResponse = [];

            expect(fn () => $this->provider->startVideo('x'))->toThrow(ProviderException::class);
        });
    });

    describe('videoStatus', function () {
        it('reports running while the operation is not done', function () {
            $this->provider->pollResponses = [['done' => false]];

            $status = $this->provider->videoStatus('operations/abc');

            expect($status->isRunning())->toBeTrue();
            expect($this->provider->lastUrl)->toContain('operations/abc');
        });

        it('reports completed when the operation is done', function () {
            $this->provider->pollResponses = [['done' => true]];

            expect($this->provider->videoStatus('operations/abc')->isCompleted())->toBeTrue();
        });

        it('reports failed and carries the error message', function () {
            $this->provider->pollResponses = [['error' => ['message' => 'quota exceeded']]];

            $status = $this->provider->videoStatus('operations/abc');

            expect($status->isFailed())->toBeTrue();
            expect($status->error)->toBe('quota exceeded');
        });
    });

    describe('fetchVideo', function () {
        it('downloads the clip from a returned uri', function () {
            $this->provider->pollResponses = [[
                'done' => true,
                'response' => ['generateVideoResponse' => ['generatedSamples' => [
                    ['video' => ['uri' => 'https://files.test/clip']],
                ]]],
            ]];

            $video = $this->provider->fetchVideo('models/veo-3.1-generate-preview/operations/abc');

            expect($video)->toBeInstanceOf(VideoResponse::class);
            expect($video->data)->toBe('fake-video-binary');
            expect($video->uri)->toBe('https://files.test/clip');
            expect($video->model)->toBe('veo-3.1-generate-preview');
        });

        it('decodes inline base64 video bytes', function () {
            $this->provider->pollResponses = [[
                'done' => true,
                'response' => ['generateVideoResponse' => ['generatedSamples' => [
                    ['video' => ['bytesBase64Encoded' => base64_encode('inline-clip'), 'mimeType' => 'video/webm']],
                ]]],
            ]];

            $video = $this->provider->fetchVideo('operations/abc');

            expect($video->data)->toBe('inline-clip');
            expect($video->mimeType)->toBe('video/webm');
        });

        it('falls back to a uri-only response when the download fails', function () {
            $this->provider->downloadShouldFail = true;
            $this->provider->pollResponses = [[
                'done' => true,
                'response' => ['generateVideoResponse' => ['generatedSamples' => [
                    ['video' => ['uri' => 'https://files.test/clip']],
                ]]],
            ]];

            $video = $this->provider->fetchVideo('operations/abc');

            expect($video->hasData())->toBeFalse();
            expect($video->uri)->toBe('https://files.test/clip');
        });

        it('throws when the job is not complete', function () {
            $this->provider->pollResponses = [['done' => false]];

            expect(fn () => $this->provider->fetchVideo('operations/abc'))->toThrow(ProviderException::class);
        });

        it('throws when no video is present', function () {
            $this->provider->pollResponses = [['done' => true, 'response' => []]];

            expect(fn () => $this->provider->fetchVideo('operations/abc'))->toThrow(ProviderException::class);
        });
    });

    describe('generateVideo (blocking)', function () {
        it('polls until done and returns the finished clip', function () {
            $this->provider->pollResponses = [
                ['done' => false],
                ['done' => true],
                ['done' => true, 'response' => ['generateVideoResponse' => ['generatedSamples' => [
                    ['video' => ['uri' => 'https://files.test/clip']],
                ]]]],
            ];

            $video = $this->provider->generateVideo('a cat surfing');

            expect($video->data)->toBe('fake-video-binary');
            expect($this->provider->pauseCalls)->toBe(1);
        });

        it('throws when generation fails', function () {
            $this->provider->pollResponses = [['error' => ['message' => 'safety block']]];

            expect(fn () => $this->provider->generateVideo('x'))->toThrow(ProviderException::class);
        });

        it('throws when polling exceeds the timeout', function () {
            $this->provider->pollResponses = [['done' => false]];

            expect(fn () => $this->provider->generateVideo('x', ['pollTimeoutSeconds' => 0]))
                ->toThrow(ProviderException::class);
        });
    });
});
