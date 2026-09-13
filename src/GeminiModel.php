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

use PapiAI\Core\Effort;

/**
 * Every Google model this package knows, and what each one accepts.
 *
 * The model, not the provider, decides which thinking knob it takes, whether thinking can be
 * switched off and which effort levels exist. Putting that here keeps the provider free of
 * string-sniffing, and lets the model watch enumerate what we ship instead of parsing source.
 *
 * An ID we have not heard of is not an error: `tryFrom()` returns null and the provider reads the
 * generation from the name, assuming newer rather than older.
 *
 * Retirement dates are Google's published shutdown dates, ISO formatted.
 *
 * @see https://ai.google.dev/gemini-api/docs/models
 */
enum GeminiModel: string
{
    case Flash38 = 'gemini-3.8-flash';
    case Flash37 = 'gemini-3.7-flash';
    case Flash36 = 'gemini-3.6-flash';
    case Flash35 = 'gemini-3.5-flash';
    case Flash35Lite = 'gemini-3.5-flash-lite';
    case Pro31 = 'gemini-3.1-pro-preview';
    case Flash31Lite = 'gemini-3.1-flash-lite';
    case Flash3 = 'gemini-3-flash-preview';
    case Pro25 = 'gemini-2.5-pro';
    case Flash25 = 'gemini-2.5-flash';
    case Flash25Lite = 'gemini-2.5-flash-lite';
    case Flash20 = 'gemini-2.0-flash';
    case Flash20Lite = 'gemini-2.0-flash-lite';
    /** @deprecated Shut down 9 March 2026; the alias now redirects to gemini-3.1-pro-preview. */
    case Pro3 = 'gemini-3-pro-preview';
    /** @deprecated Retired; no longer published by Google. */
    case Pro15 = 'gemini-1.5-pro';
    /** @deprecated Retired; no longer published by Google. */
    case Flash15 = 'gemini-1.5-flash';

    case Flash31Image = 'gemini-3.1-flash-image';
    case Flash31LiteImage = 'gemini-3.1-flash-lite-image';
    case Pro3Image = 'gemini-3-pro-image';
    case Flash25Image = 'gemini-2.5-flash-image';
    /** @deprecated Imagen shut down 17 August 2026. Use Flash31Image. */
    case Imagen4 = 'imagen-4.0-generate-001';
    /** @deprecated Imagen shut down 17 August 2026. Use Flash31Image. */
    case Imagen4Ultra = 'imagen-4.0-ultra-generate-001';
    /** @deprecated Imagen shut down 17 August 2026. Use Flash31Image. */
    case Imagen4Fast = 'imagen-4.0-fast-generate-001';
    /** @deprecated Imagen 3 is already shut down. Use Flash31Image. */
    case ImagenEdit = 'imagen-3.0-capability-001';

    case Veo31 = 'veo-3.1-generate-preview';
    case Veo31Lite = 'veo-3.1-lite-generate-preview';
    /** @deprecated Shut down 30 June 2026; requests fail. Use Veo31. */
    case Veo3 = 'veo-3.0-generate-001';
    /** @deprecated Shut down 30 June 2026; requests fail. Use Veo31. */
    case Veo2 = 'veo-2.0-generate-001';

    /**
     * Whether this model takes a thinking level (Gemini 3 and later) rather than a token budget.
     *
     * Google warns that the budget knob behaves unpredictably on Gemini 3 Pro, so the level knob
     * is always used from 3 on.
     */
    public function takesThinkingLevel(): bool
    {
        return match ($this) {
            self::Flash38, self::Flash37, self::Flash36, self::Flash35, self::Flash35Lite,
            self::Pro31, self::Flash31Lite, self::Flash3, self::Pro3 => true,
            default => false,
        };
    }

    /**
     * The thinking levels this model accepts, in the neutral vocabulary.
     *
     * Only meaningful for models that take a level. No Gemini 3 model can switch thinking off,
     * and Pro does not accept MINIMAL at all, so its floor is LOW. Budget models return an empty
     * list: their knob is continuous.
     *
     * @return list<Effort>
     */
    public function effortLevels(): array
    {
        if (!$this->takesThinkingLevel()) {
            return [];
        }

        return match ($this) {
            self::Pro31, self::Pro3 => [Effort::Low, Effort::Medium, Effort::High],
            default => [Effort::Minimal, Effort::Low, Effort::Medium, Effort::High],
        };
    }

    /**
     * Whether a zero budget genuinely disables thinking. Only the pre-3 Flash families can;
     * Pro has a floor it will not go below.
     */
    public function canDisableThinking(): bool
    {
        return match ($this) {
            self::Flash25, self::Flash25Lite, self::Flash20, self::Flash20Lite, self::Flash15 => true,
            default => false,
        };
    }

    /**
     * Whether Google has retired this model.
     */
    public function isDeprecated(): bool
    {
        return match ($this) {
            self::Pro3, self::Pro15, self::Flash15,
            self::Imagen4, self::Imagen4Ultra, self::Imagen4Fast, self::ImagenEdit,
            self::Veo3, self::Veo2 => true,
            default => false,
        };
    }

    /**
     * The published shutdown date, ISO formatted, where Google gave one.
     */
    public function retiredOn(): ?string
    {
        return match ($this) {
            self::Pro3 => '2026-03-09',
            self::Imagen4, self::Imagen4Ultra, self::Imagen4Fast => '2026-08-17',
            self::Veo3, self::Veo2 => '2026-06-30',
            default => null,
        };
    }

    /**
     * What to use instead, for the retired models that have a successor here.
     */
    public function replacement(): ?self
    {
        return match ($this) {
            self::Pro3 => self::Pro31,
            self::Imagen4, self::Imagen4Ultra, self::Imagen4Fast, self::ImagenEdit => self::Flash31Image,
            self::Veo3, self::Veo2 => self::Veo31,
            default => null,
        };
    }
}
