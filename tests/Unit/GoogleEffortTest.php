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

use PapiAI\Core\Effort;
use PapiAI\Core\Message;
use PapiAI\Google\GoogleProvider;

/**
 * Captures the request payload so effort mapping can be asserted without HTTP.
 */
class TestableGoogleEffortProvider extends GoogleProvider
{
    public array $lastPayload = [];

    protected function request(string $url, array $payload): array
    {
        $this->lastPayload = $payload;

        return ['status' => 'completed', 'steps' => [['type' => 'model_output', 'content' => [['type' => 'text', 'text' => 'ok']]]], 'usage' => []];
    }
}

describe('GoogleProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableGoogleEffortProvider('test-api-key');
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
        $this->level = fn () => $this->provider->lastPayload['generation_config']['thinking_level'] ?? null;
    });

    describe('thinking_level, which every text model takes on Interactions', function () {
        it('carries all four levels on a Flash that offers them', function () {
            $levels = [];

            foreach (['minimal', 'low', 'medium', 'high'] as $level) {
                ($this->chat)(['effort' => $level, 'model' => 'gemini-3.6-flash']);
                $levels[] = ($this->level)();
            }

            expect($levels)->toBe(['minimal', 'low', 'medium', 'high']);
        });

        it('has no off switch, so none becomes the shallowest level available', function () {
            ($this->chat)(['effort' => 'none', 'model' => 'gemini-3.6-flash']);

            expect(($this->level)())->toBe('minimal');
        });

        it('keeps the models without MINIMAL at LOW', function () {
            foreach (['gemini-3.8-flash', 'gemini-3.1-pro-preview', 'gemini-4-carbon', 'gemini-2.5-flash'] as $model) {
                ($this->chat)(['effort' => 'none', 'model' => $model]);

                expect(($this->level)())->toBe('low', $model);
            }
        });

        it('gives Hydrogen xhigh, the only Gemini that takes it', function () {
            ($this->chat)(['effort' => 'extra-high', 'model' => 'gemini-4-hydrogen']);
            expect(($this->level)())->toBe('xhigh');

            ($this->chat)(['effort' => 'extra-high', 'model' => 'gemini-4-carbon']);
            expect(($this->level)())->toBe('high');
        });

        it('sends no level for none on 2.5 Flash-Lite, which is off by default', function () {
            ($this->chat)(['effort' => 'none', 'model' => 'gemini-2.5-flash-lite']);

            expect(($this->level)())->toBeNull();
        });

        it('assumes the three middle levels for a model it has not heard of', function () {
            ($this->chat)(['effort' => 'maximum', 'model' => 'gemini-5-flash']);

            expect(($this->level)())->toBe('high');
        });
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)(['model' => 'gemini-2.5-pro']);

        expect($this->provider->lastPayload['generation_config'])->not->toHaveKey('thinking_level');
    });

    it('rejects a level it does not recognise', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');
    });

    it('accepts a provider-level default the call can override', function () {
        $provider = new TestableGoogleEffortProvider('k', 'gemini-3.6-flash', 8192, Effort::High);

        $provider->chat([Message::user('hi')], []);
        expect($provider->lastPayload['generation_config']['thinking_level'])->toBe('high');

        $provider->chat([Message::user('hi')], ['effort' => 'low']);
        expect($provider->lastPayload['generation_config']['thinking_level'])->toBe('low');
    });
});
