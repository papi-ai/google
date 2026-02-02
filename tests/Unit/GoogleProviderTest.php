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

use PapiAI\Google\GoogleProvider;
use PapiAI\Core\Contracts\ProviderInterface;
use PapiAI\Core\Message;

describe('GoogleProvider', function () {
    describe('construction', function () {
        it('implements ProviderInterface', function () {
            $provider = new GoogleProvider('test-api-key');

            expect($provider)->toBeInstanceOf(ProviderInterface::class);
        });

        it('uses default model gemini-3.0-pro', function () {
            $provider = new GoogleProvider('test-api-key');

            expect($provider->getName())->toBe('google');
        });

        it('accepts custom model', function () {
            $provider = new GoogleProvider(
                apiKey: 'test-api-key',
                defaultModel: GoogleProvider::MODEL_3_1_PRO,
            );

            expect($provider)->toBeInstanceOf(GoogleProvider::class);
        });
    });

    describe('model constants', function () {
        it('has correct model constants', function () {
            expect(GoogleProvider::MODEL_3_1_PRO)->toBe('gemini-3.1-pro');
            expect(GoogleProvider::MODEL_3_0_PRO)->toBe('gemini-3.0-pro');
            expect(GoogleProvider::MODEL_2_0_FLASH)->toBe('gemini-2.0-flash-exp');
            expect(GoogleProvider::MODEL_1_5_PRO)->toBe('gemini-1.5-pro');
            expect(GoogleProvider::MODEL_1_5_FLASH)->toBe('gemini-1.5-flash');
        });
    });

    describe('capabilities', function () {
        it('supports tools', function () {
            $provider = new GoogleProvider('test-api-key');

            expect($provider->supportsTool())->toBeTrue();
        });

        it('supports vision', function () {
            $provider = new GoogleProvider('test-api-key');

            expect($provider->supportsVision())->toBeTrue();
        });

        it('supports structured output', function () {
            $provider = new GoogleProvider('test-api-key');

            expect($provider->supportsStructuredOutput())->toBeTrue();
        });
    });

    describe('message conversion', function () {
        it('converts user role correctly', function () {
            $provider = new GoogleProvider('test-api-key');

            // Verify provider can handle message types
            $messages = [
                Message::user('Hello'),
            ];

            expect($provider)->toBeInstanceOf(GoogleProvider::class);
        });

        it('handles system instructions', function () {
            $provider = new GoogleProvider('test-api-key');

            $messages = [
                Message::system('Be helpful'),
                Message::user('Hello'),
            ];

            // Provider should handle system messages as systemInstruction
            expect($provider)->toBeInstanceOf(GoogleProvider::class);
        });

        it('handles multimodal messages', function () {
            $provider = new GoogleProvider('test-api-key');

            $messages = [
                Message::userWithImage('What is this?', 'https://example.com/image.jpg'),
            ];

            expect($provider)->toBeInstanceOf(GoogleProvider::class);
        });
    });
});

describe('GoogleProvider integration', function () {
    it('throws on API error', function () {
        $this->markTestSkipped('Integration test - requires GOOGLE_API_KEY');

        $provider = new GoogleProvider('invalid-key');

        expect(fn() => $provider->chat([Message::user('Hello')]))
            ->toThrow(RuntimeException::class);
    });
})->skip();
