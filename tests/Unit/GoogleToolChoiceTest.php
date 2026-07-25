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

use PapiAI\Core\Message;
use PapiAI\Google\GoogleProvider;

/**
 * Captures the request payload so tool-choice mapping can be asserted without HTTP.
 */
class TestableGoogleToolChoiceProvider extends GoogleProvider
{
    public array $lastPayload = [];

    protected function request(string $url, array $payload): array
    {
        $this->lastPayload = $payload;

        return ['candidates' => [['content' => ['parts' => [['text' => 'ok']]], 'finishReason' => 'STOP']]];
    }
}

describe('GoogleProvider tool choice', function () {
    beforeEach(function () {
        $this->provider = new TestableGoogleToolChoiceProvider('test-api-key');
        $this->tools = [
            ['name' => 'get_weather', 'description' => 'Weather', 'parameters' => ['type' => 'object']],
        ];
    });

    $callConfig = function ($provider) {
        return $provider->lastPayload['toolConfig']['functionCallingConfig'] ?? null;
    };

    it('maps auto to AUTO', function () use ($callConfig) {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'auto']);

        expect($callConfig($this->provider))->toBe(['mode' => 'AUTO']);
    });

    it('maps none to NONE', function () use ($callConfig) {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'none']);

        expect($callConfig($this->provider))->toBe(['mode' => 'NONE']);
    });

    it('maps required to ANY without allowedFunctionNames', function () use ($callConfig) {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'required']);

        expect($callConfig($this->provider))->toBe(['mode' => 'ANY']);
    });

    it('maps a specific tool to ANY + allowedFunctionNames', function () use ($callConfig) {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => ['name' => 'get_weather']]);

        expect($callConfig($this->provider))->toBe(['mode' => 'ANY', 'allowedFunctionNames' => ['get_weather']]);
    });

    it('emits no toolConfig when toolChoice is absent (backward compatible)', function () {
        $this->provider->chat([Message::user('hi')], ['tools' => $this->tools]);

        expect($this->provider->lastPayload)->not->toHaveKey('toolConfig');
    });

    it('throws for required with no tools, before any HTTP call', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['toolChoice' => 'required']))
            ->toThrow(InvalidArgumentException::class);
        expect($this->provider->lastPayload)->toBe([]);
    });

    it('throws for an unknown tool name', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => ['name' => 'nope']]))
            ->toThrow(InvalidArgumentException::class);
    });

    it('throws for an unknown toolChoice value', function () {
        expect(fn () => $this->provider->chat([Message::user('hi')], ['tools' => $this->tools, 'toolChoice' => 'always']))
            ->toThrow(InvalidArgumentException::class);
    });
});
