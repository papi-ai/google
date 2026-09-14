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
use PapiAI\Google\GeminiModel;
use PapiAI\Google\GoogleProvider;

describe('GeminiModel', function () {
    it('is the source of truth the old constants alias', function () {
        expect(GoogleProvider::MODEL_3_6_FLASH)->toBe(GeminiModel::Flash36->value);
        expect(GoogleProvider::MODEL_3_8_FLASH)->toBe('gemini-3.8-flash');
        expect(GoogleProvider::IMAGEN_4)->toBe(GeminiModel::Imagen4->value);
        expect(GoogleProvider::MODEL_VEO_3_1)->toBe(GeminiModel::Veo31->value);
    });

    it('returns null for an ID it has not heard of', function () {
        expect(GeminiModel::tryFrom('gemini-4-flash'))->toBeNull();
    });

    it('ships unique IDs', function () {
        $ids = array_map(fn (GeminiModel $m) => $m->value, GeminiModel::cases());

        expect($ids)->toBe(array_unique($ids));
    });

    it('takes a level from Gemini 3 on and a budget before', function () {
        expect(GeminiModel::Flash38->takesThinkingLevel())->toBeTrue();
        expect(GeminiModel::Pro31->takesThinkingLevel())->toBeTrue();
        expect(GeminiModel::Flash25->takesThinkingLevel())->toBeFalse();
        expect(GeminiModel::Pro25->takesThinkingLevel())->toBeFalse();
    });

    it('keeps Pro off MINIMAL and gives Flash all four levels', function () {
        expect(GeminiModel::Pro31->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High]);
        expect(GeminiModel::Flash38->effortLevels())->toBe([Effort::Minimal, Effort::Low, Effort::Medium, Effort::High]);
        expect(GeminiModel::Flash25->effortLevels())->toBe([]);
    });

    it('lets only the pre-3 Flash families disable thinking', function () {
        expect(GeminiModel::Flash25->canDisableThinking())->toBeTrue();
        expect(GeminiModel::Pro25->canDisableThinking())->toBeFalse();
        expect(GeminiModel::Flash38->canDisableThinking())->toBeFalse();
    });

    it('knows the dead ones, their dates and their successors', function () {
        expect(GeminiModel::Imagen4->isDeprecated())->toBeTrue();
        expect(GeminiModel::Imagen4->retiredOn())->toBe('2026-08-17');
        expect(GeminiModel::Imagen4->replacement())->toBe(GeminiModel::Flash31Image);
        expect(GeminiModel::Veo2->replacement())->toBe(GeminiModel::Veo31);
        expect(GeminiModel::Flash38->isDeprecated())->toBeFalse();
    });
});
