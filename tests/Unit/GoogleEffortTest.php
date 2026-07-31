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

        return ['candidates' => [['content' => ['parts' => [['text' => 'ok']]], 'finishReason' => 'STOP']]];
    }
}

describe('GoogleProvider reasoning effort', function () {
    beforeEach(function () {
        $this->provider = new TestableGoogleEffortProvider('test-api-key');
        $this->chat = fn (array $options) => $this->provider->chat([Message::user('hi')], $options);
        $this->thinking = fn () => $this->provider->lastPayload['generationConfig']['thinkingConfig'] ?? [];
    });

    describe('Gemini 3, which takes a thinking level', function () {
        it('uses the level knob, never the budget one', function () {
            // Sending both is an error, and a budget on Gemini 3 Pro is documented as unreliable.
            ($this->chat)(['effort' => 'medium', 'model' => 'gemini-3-flash-preview']);

            expect(($this->thinking)())->toHaveKey('thinkingLevel');
            expect(($this->thinking)())->not->toHaveKey('thinkingBudget');
        });

        it('carries all four levels, not just two', function () {
            $levels = [];

            foreach (['minimal', 'low', 'medium', 'high'] as $level) {
                ($this->chat)(['effort' => $level, 'model' => 'gemini-3-flash-preview']);
                $levels[] = ($this->thinking)()['thinkingLevel'];
            }

            expect($levels)->toBe(['minimal', 'low', 'medium', 'high']);
        });

        it('has no off switch, so none becomes the shallowest level available', function () {
            // Gemini 3 cannot disable thinking. MINIMAL is the closest, and does not guarantee it.
            ($this->chat)(['effort' => 'none', 'model' => 'gemini-3-flash-preview']);

            expect(($this->thinking)()['thinkingLevel'])->toBe('minimal');
        });

        it('keeps 3.1 Pro off MINIMAL, which it does not accept', function () {
            foreach (['none', 'minimal'] as $level) {
                ($this->chat)(['effort' => $level, 'model' => 'gemini-3.1-pro']);

                expect(($this->thinking)()['thinkingLevel'])->toBe('low');
            }
        });

        it('narrows the levels above what Gemini offers', function () {
            foreach (['extra-high', 'maximum'] as $level) {
                ($this->chat)(['effort' => $level, 'model' => 'gemini-3-flash-preview']);

                expect(($this->thinking)()['thinkingLevel'])->toBe('high');
            }
        });
    });

    describe('Gemini 2.5, which takes a thinking budget', function () {
        it('sets a budget that grows with effort', function () {
            $budgets = [];

            foreach (['low', 'medium', 'high'] as $level) {
                ($this->chat)(['effort' => $level, 'model' => 'gemini-2.5-flash', 'maxTokens' => 20_000]);
                $budgets[] = ($this->thinking)()['thinkingBudget'];
            }

            expect($budgets[0])->toBeLessThan($budgets[1]);
            expect($budgets[1])->toBeLessThan($budgets[2]);
        });

        it('stays inside the range Gemini accepts, however large the ceiling', function () {
            ($this->chat)(['effort' => 'maximum', 'model' => 'gemini-2.5-pro', 'maxTokens' => 200_000]);

            expect(($this->thinking)()['thinkingBudget'])->toBeLessThanOrEqual(32_768);
        });

        it('disables thinking on Flash, which is the only family that can', function () {
            ($this->chat)(['effort' => 'none', 'model' => 'gemini-2.5-flash']);

            expect(($this->thinking)()['thinkingBudget'])->toBe(0);
        });

        it('keeps 2.5 Pro above its floor, since it cannot disable thinking', function () {
            ($this->chat)(['effort' => 'none', 'model' => 'gemini-2.5-pro']);

            expect(($this->thinking)()['thinkingBudget'])->toBeGreaterThanOrEqual(128);
        });
    });

    it('sends nothing when the caller does not ask', function () {
        ($this->chat)(['model' => 'gemini-2.5-pro']);

        expect($this->provider->lastPayload['generationConfig'])->not->toHaveKey('thinkingConfig');
    });

    it('rejects a level it does not recognise', function () {
        expect(fn () => ($this->chat)(['effort' => 'enormous']))
            ->toThrow(InvalidArgumentException::class, 'enormous');
    });

    it('accepts a provider-level default the call can override', function () {
        $provider = new TestableGoogleEffortProvider('k', 'gemini-3-flash-preview', 8192, Effort::High);

        $provider->chat([Message::user('hi')], []);
        expect($provider->lastPayload['generationConfig']['thinkingConfig']['thinkingLevel'])->toBe('high');

        $provider->chat([Message::user('hi')], ['effort' => 'low']);
        expect($provider->lastPayload['generationConfig']['thinkingConfig']['thinkingLevel'])->toBe('low');
    });
});
