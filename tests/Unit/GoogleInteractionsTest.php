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

use PapiAI\Core\Exception\ProviderException;
use PapiAI\Core\Message;
use PapiAI\Core\StopReason;
use PapiAI\Core\StreamChunk;
use PapiAI\Core\ToolCall;
use PapiAI\Google\GoogleProvider;

/**
 * Captures the request and replays a canned interaction, so the wire format can be asserted
 * without HTTP.
 */
class TestableInteractionsProvider extends GoogleProvider
{
    public string $lastUrl = '';
    public array $lastPayload = [];
    public array $fakeResponse = ['status' => 'completed', 'steps' => [], 'usage' => []];
    public array $fakeStreamEvents = [];
    public string $fakeImageData = 'raw-bytes';

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

        yield from $this->fakeStreamEvents;
    }

    protected function fetchImage(string $url): string|false
    {
        return $this->fakeImageData;
    }
}

describe('GoogleProvider on the Interactions API', function () {
    beforeEach(function () {
        $this->provider = new TestableInteractionsProvider('test-api-key');
        $this->chat = fn (array $messages, array $options = []) => $this->provider->chat($messages, $options);
        $this->textTurn = fn (string $text) => [
            'id' => 'v1_abc', 'status' => 'completed',
            'steps' => [
                ['type' => 'thought', 'signature' => 'sig'],
                ['type' => 'model_output', 'content' => [['type' => 'text', 'text' => $text]]],
            ],
            'usage' => ['total_input_tokens' => 11, 'total_output_tokens' => 7, 'total_thought_tokens' => 58, 'total_tokens' => 76],
        ];
    });

    describe('request shape', function () {
        it('posts to /interactions and keeps the request stateless', function () {
            ($this->chat)([Message::user('Hi')]);

            expect($this->provider->lastUrl)->toBe('https://generativelanguage.googleapis.com/v1beta/interactions');
            expect($this->provider->lastPayload['store'])->toBeFalse();
            expect($this->provider->lastPayload['model'])->toBe(GoogleProvider::MODEL_3_8_FLASH);
        });

        it('sends the system message as system_instruction and turns as input items', function () {
            ($this->chat)([Message::system('Be terse.'), Message::user('Hi')]);

            expect($this->provider->lastPayload['system_instruction'])->toBe('Be terse.');
            expect($this->provider->lastPayload['input'])->toBe([
                ['type' => 'user_input', 'content' => [['type' => 'text', 'text' => 'Hi']]],
            ]);
            expect($this->provider->lastPayload)->not->toHaveKey('contents');
        });

        it('puts the output ceiling and stop sequences in generation_config', function () {
            ($this->chat)([Message::user('Hi')], ['maxTokens' => 512, 'stopSequences' => ['END']]);

            expect($this->provider->lastPayload['generation_config']['max_output_tokens'])->toBe(512);
            expect($this->provider->lastPayload['generation_config']['stop_sequences'])->toBe(['END']);
        });

        it('declares tools as function tools', function () {
            ($this->chat)([Message::user('Hi')], ['tools' => [
                ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object', 'properties' => []]],
            ]]);

            expect($this->provider->lastPayload['tools'])->toBe([[
                'type' => 'function', 'name' => 'get_weather', 'description' => 'Weather',
                'parameters' => ['type' => 'object', 'properties' => []],
            ]]);
        });

        it('asks for JSON through response_format', function () {
            ($this->chat)([Message::user('Hi')], ['outputSchema' => ['type' => 'object', 'properties' => []]]);

            expect($this->provider->lastPayload['response_format'])->toBe([
                'type' => 'text', 'mime_type' => 'application/json', 'schema' => ['type' => 'object', 'properties' => []],
            ]);
        });

        it('inlines images, fetching a URL first, as image parts with data and mime_type', function () {
            ($this->chat)([Message::user([
                ['type' => 'text', 'text' => 'What is this?'],
                ['type' => 'image', 'source' => ['type' => 'url', 'url' => 'https://example.com/a.png']],
                ['type' => 'image', 'source' => ['type' => 'base64', 'media_type' => 'image/jpeg', 'data' => 'AAAA']],
            ])]);

            expect($this->provider->lastPayload['input'][0]['content'])->toBe([
                ['type' => 'text', 'text' => 'What is this?'],
                ['type' => 'image', 'mime_type' => 'image/png', 'data' => base64_encode('raw-bytes')],
                ['type' => 'image', 'mime_type' => 'image/jpeg', 'data' => 'AAAA'],
            ]);
        });
    });

    describe('tool calls across turns', function () {
        beforeEach(function () {
            $this->provider->fakeResponse = [
                'status' => 'completed',
                'steps' => [
                    ['type' => 'thought', 'signature' => 'sig-1'],
                    ['type' => 'function_call', 'id' => 'fc_1', 'name' => 'get_weather', 'arguments' => ['city' => 'Lisbon']],
                ],
                'usage' => ['total_input_tokens' => 1, 'total_output_tokens' => 1],
            ];
        });

        it('turns function_call steps into ToolCalls keyed by the step id', function () {
            $response = ($this->chat)([Message::user('Weather?')]);

            expect($response->toolCalls)->toHaveCount(1);
            expect($response->toolCalls[0]->id)->toBe('fc_1');
            expect($response->toolCalls[0]->name)->toBe('get_weather');
            expect($response->toolCalls[0]->arguments)->toBe(['city' => 'Lisbon']);
            expect($response->stopReason)->toBe('tool_calls');
        });

        it('replays the model steps exactly as received, then the function_result naming the tool', function () {
            $first = ($this->chat)([Message::user('Weather?')]);

            ($this->chat)([
                Message::user('Weather?'),
                Message::assistant('', $first->toolCalls),
                Message::toolResult('fc_1', '{"temp": 21}'),
            ]);

            expect($this->provider->lastPayload['input'])->toBe([
                ['type' => 'user_input', 'content' => [['type' => 'text', 'text' => 'Weather?']]],
                ['type' => 'thought', 'signature' => 'sig-1'],
                ['type' => 'function_call', 'id' => 'fc_1', 'name' => 'get_weather', 'arguments' => ['city' => 'Lisbon']],
                ['type' => 'function_result', 'name' => 'get_weather', 'call_id' => 'fc_1', 'result' => [['type' => 'text', 'text' => '{"temp": 21}']]],
            ]);
        });

        it('rebuilds a turn it never saw from the message, rather than failing', function () {
            ($this->chat)([
                Message::user('Weather?'),
                Message::assistant('Checking.', [new ToolCall('ext_1', 'get_weather', ['city' => 'Porto'])]),
                Message::toolResult('ext_1', 'rain'),
            ]);

            $input = $this->provider->lastPayload['input'];
            expect($input[1])->toBe(['type' => 'model_output', 'content' => [['type' => 'text', 'text' => 'Checking.']]]);
            expect($input[2]['type'])->toBe('function_call');
            expect($input[2]['id'])->toBe('ext_1');
            expect($input[3]['name'])->toBe('ext_1');
        });
    });

    describe('response parsing', function () {
        it('reads the text from model_output and ignores thoughts', function () {
            $this->provider->fakeResponse = ($this->textTurn)('Hello back');

            $response = ($this->chat)([Message::user('Hi')]);

            expect($response->text)->toBe('Hello back');
            expect($response->stopReason)->toBe('stop');
        });

        it('counts thought tokens as output, as every other provider does', function () {
            $this->provider->fakeResponse = ($this->textTurn)('ok');

            $response = ($this->chat)([Message::user('Hi')]);

            expect($response->usage->inputTokens)->toBe(11);
            expect($response->usage->outputTokens)->toBe(65);
            expect($response->usage->totalTokens)->toBe(76);
        });

        it('reports incomplete when the output ceiling cut the turn short', function () {
            $this->provider->fakeResponse = ['status' => 'incomplete', 'steps' => [], 'usage' => []];

            expect(($this->chat)([Message::user('Hi')])->stopReason)->toBe('incomplete');
        });
    });

    describe('streaming', function () {
        it('yields text deltas and completes on interaction.completed', function () {
            $this->provider->fakeStreamEvents = [
                ['event_type' => 'interaction.created', 'interaction' => ['id' => 'v1_x']],
                ['event_type' => 'step.start', 'index' => 0, 'step' => ['type' => 'thought']],
                ['event_type' => 'step.delta', 'index' => 0, 'delta' => ['type' => 'thought_signature', 'signature' => 'sig']],
                ['event_type' => 'step.delta', 'index' => 1, 'delta' => ['type' => 'text', 'text' => 'Hel']],
                ['event_type' => 'step.delta', 'index' => 1, 'delta' => ['type' => 'text', 'text' => 'lo']],
                ['event_type' => 'interaction.completed', 'interaction' => ['status' => 'completed']],
            ];

            $chunks = iterator_to_array($this->provider->stream([Message::user('Hi')]));

            expect($this->provider->lastUrl)->toBe('https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse');
            expect($this->provider->lastPayload['stream'])->toBeTrue();
            expect(array_map(fn (StreamChunk $c) => $c->text, $chunks))->toBe(['Hel', 'lo', '']);
            expect(end($chunks)->isComplete)->toBeTrue();
        });
    });

    describe('sampling parameters', function () {
        it('refuses temperature on Gemini 3.6 and later, which ignore it', function () {
            foreach ([GoogleProvider::MODEL_3_8_FLASH, GoogleProvider::MODEL_4_CARBON] as $model) {
                expect(fn () => ($this->chat)([Message::user('hi')], ['temperature' => 0.2, 'model' => $model]))
                    ->toThrow(ProviderException::class, 'temperature');
            }
        });

        it('still forwards temperature to models that honour it', function () {
            ($this->chat)([Message::user('hi')], ['temperature' => 0.2, 'model' => GoogleProvider::MODEL_2_5_FLASH]);

            expect($this->provider->lastPayload['generation_config']['temperature'])->toBe(0.2);
        });
    });

    describe('images', function () {
        beforeEach(function () {
            $this->provider->fakeResponse = [
                'status' => 'completed',
                'steps' => [
                    ['type' => 'thought', 'signature' => 's'],
                    ['type' => 'model_output', 'content' => [
                        ['type' => 'text', 'text' => 'Here it is'],
                        ['type' => 'image', 'mime_type' => 'image/jpeg', 'data' => 'IMG'],
                    ]],
                ],
                'usage' => [],
            ];
        });

        it('generates through /interactions with an image response_format', function () {
            $result = $this->provider->generateImage('A cat', ['aspectRatio' => '16:9', 'imageSize' => '2K']);

            expect($this->provider->lastUrl)->toBe('https://generativelanguage.googleapis.com/v1beta/interactions');
            expect($this->provider->lastPayload['model'])->toBe(GoogleProvider::MODEL_3_1_FLASH_IMAGE);
            expect($this->provider->lastPayload['input'])->toBe([['type' => 'user_input', 'content' => [['type' => 'text', 'text' => 'A cat']]]]);
            expect($this->provider->lastPayload['response_format'])->toBe(['type' => 'image', 'aspect_ratio' => '16:9', 'image_size' => '2K']);
            expect($result['images'])->toBe([['mimeType' => 'image/jpeg', 'data' => 'IMG']]);
        });

        it('sends only the type when the caller sets neither ratio nor size', function () {
            $this->provider->generateImage('A cat');

            expect($this->provider->lastPayload['response_format'])->toBe(['type' => 'image']);
        });

        it('edits by sending the source image as an image part before the prompt', function () {
            $result = $this->provider->editImage('https://example.com/photo.png', 'Make it blue');

            $content = $this->provider->lastPayload['input'][0]['content'];
            expect($content[0]['type'])->toBe('image');
            expect($content[0]['mime_type'])->toBe('image/png');
            expect($content[1])->toBe(['type' => 'text', 'text' => 'Make it blue']);
            expect($result['text'])->toBe('Here it is');
        });

        it('still refuses numberOfImages above one', function () {
            expect(fn () => $this->provider->generateImage('A cat', ['numberOfImages' => 3]))
                ->toThrow(ProviderException::class, 'one image per request');
        });
    });
});

describe('GoogleProvider stop reason', function () {
    beforeEach(function () {
        $this->provider = new TestableInteractionsProvider('test-api-key');
        $this->stoppedBecause = function (array $response): StopReason {
            $this->provider->fakeResponse = $response + [
                'steps' => [['type' => 'model_output', 'content' => [['type' => 'text', 'text' => 'x']]]],
                'usage' => [],
            ];

            return $this->provider->chat([Message::user('Hi')])->stoppedBecause;
        };
    });

    it('says why the model stopped in core\'s vocabulary', function () {
        expect(($this->stoppedBecause)(['status' => 'completed']))->toBe(StopReason::Stop);
        expect(($this->stoppedBecause)(['status' => 'incomplete']))->toBe(StopReason::MaxTokens);
        expect(($this->stoppedBecause)(['status' => 'requires_continuation']))->toBe(StopReason::MaxTokens);
        expect(($this->stoppedBecause)(['status' => 'failed']))->toBe(StopReason::Other);
        expect(($this->stoppedBecause)(['steps' => [['type' => 'function_call', 'id' => 'c1', 'name' => 'f', 'arguments' => []]]]))->toBe(StopReason::ToolCalls);
    });
});
