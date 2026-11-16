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
        expect(GoogleProvider::MODEL_4_CARBON)->toBe('gemini-4-carbon');
        expect(GoogleProvider::MODEL_EMBEDDING_2)->toBe('gemini-embedding-2');
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

    it('knows the Gemini 4 line and what each takes', function () {
        expect(GeminiModel::from('gemini-4-hydrogen'))->toBe(GeminiModel::Hydrogen4);
        expect(GeminiModel::Hydrogen4->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High, Effort::ExtraHigh]);
        expect(GeminiModel::Carbon4->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High]);
        expect(GeminiModel::Flash36->effortLevels())->toBe([Effort::Minimal, Effort::Low, Effort::Medium, Effort::High]);
        expect(GeminiModel::Flash38->effortLevels())->toBe([Effort::Low, Effort::Medium, Effort::High]);
    });

    it('knows which models ignore sampling parameters', function () {
        expect(GeminiModel::Flash38->acceptsSampling())->toBeFalse();
        expect(GeminiModel::Argon4->acceptsSampling())->toBeFalse();
        expect(GeminiModel::Flash25->acceptsSampling())->toBeTrue();
    });

    it('knows the one model that is off by default', function () {
        expect(GeminiModel::Flash25Lite->thinksByDefault())->toBeFalse();
        expect(GeminiModel::Flash38->thinksByDefault())->toBeTrue();
    });

    it('knows the dead ones, their dates and their successors', function () {
        expect(GeminiModel::Imagen4->isDeprecated())->toBeTrue();
        expect(GeminiModel::Imagen4->retiredOn())->toBe('2026-08-17');
        expect(GeminiModel::Imagen4->replacement())->toBe(GeminiModel::Flash31Image);
        expect(GeminiModel::Veo2->replacement())->toBe(GeminiModel::Veo31);
        expect(GeminiModel::Flash38->isDeprecated())->toBeFalse();
    });
});
