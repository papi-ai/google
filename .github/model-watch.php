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

use PapiAI\Google\GeminiModel;

/*
 * Compares the model IDs this package ships against what the provider currently serves.
 *
 * Deliberately dependency-free and run outside the test suite: it talks to the network, so it must
 * never be able to fail CI on a PR. Exits non-zero only when a live model ID has gone, which is
 * what opens the issue.
 *
 * Constants marked @deprecated are expected to be missing and are reported without failing: we keep
 * them on purpose so callers get a deprecation note rather than an undefined-constant fatal.
 */

require __DIR__ . '/../vendor/autoload.php';

$source = file_get_contents(__DIR__ . '/../src/GoogleProvider.php');

if ($source === false) {
    fwrite(STDERR, "Cannot read the provider source.\n");

    exit(2);
}

/** @var array<string, array{value: string, deprecated: bool, isDefault: bool}> $shipped */
$shipped = [];

// The enum is the source of truth: every ID we ship is a case, and each case knows whether it
// has been retired. No parsing of docblocks, which is what silently skipped four models before.
foreach (GeminiModel::cases() as $model) {
    $shipped[$model->name] = [
        'value' => $model->value,
        'deprecated' => $model->isDeprecated(),
        'isDefault' => false,
    ];
}

// Anything used as a fallback is the dangerous kind: it breaks callers who pass no model at all.
// The provider spells a default as `?? self::CONSTANT`, and each constant aliases an enum case.
preg_match_all('/public const (?P<constant>\w+) = GeminiModel::(?P<case>\w+)->value;/', $source, $aliases, PREG_SET_ORDER);

foreach ($aliases as $alias) {
    if (isset($shipped[$alias['case']]) && preg_match('/\?\?\s*self::' . preg_quote($alias['constant'], '/') . '\b/', $source) === 1) {
        $shipped[$alias['case']]['isDefault'] = true;
    }
}

if ($shipped === []) {
    fwrite(STDERR, "Found no model constants to check, which means the parser needs updating.\n");

    exit(2);
}

// Google's published model list, in plain text and with no credential. Deliberately not the
// v1beta models endpoint: that needs an API key, and a watchdog on a public repository must not
// require anyone to attach a billable key to it.
$response = @file_get_contents('https://ai.google.dev/gemini-api/docs/models.md.txt');

if ($response === false) {
    fwrite(STDERR, "Could not reach the published model list. Our connectivity is not evidence of a retirement.\n");

    exit(0);
}

preg_match_all('/\b(?:gemini|imagen|veo)-[0-9][0-9a-z.\-]*/i', $response, $found);
$served = array_values(array_unique($found[0]));

if (count($served) < 5) {
    fwrite(STDERR, "The published list parsed to almost nothing, which means the page format changed.\n");

    exit(0);
}

$missingLive = [];
$missingDeprecated = [];

foreach ($shipped as $name => $entry) {
    if (in_array($entry['value'], $served, true)) {
        continue;
    }

    $label = sprintf('%s (%s)%s', $name, $entry['value'], $entry['isDefault'] ? ' <- USED AS A DEFAULT' : '');
    $entry['deprecated'] ? $missingDeprecated[] = $label : $missingLive[] = $label;
}

printf("Checked %d shipped model IDs against %d served by the provider.\n\n", count($shipped), count($served));

if ($missingDeprecated !== []) {
    echo "Already deprecated, so expected to be gone:\n  " . implode("\n  ", $missingDeprecated) . "\n\n";
}

// Reported, but never a failure on its own. We already know about it, and a scheduled job that
// stays red on a known issue becomes wallpaper: the next real failure gets ignored with it.
foreach ($shipped as $name => $entry) {
    if ($entry['deprecated'] && $entry['isDefault']) {
        printf("Note: %s (%s) is deprecated and still used as a default. Known, and tracked separately.\n\n", $name, $entry['value']);
    }
}

if ($missingLive === []) {
    echo "Every live model ID still appears in Google's published list.\n";

    exit(0);
}

// Failing only on the unacknowledged case makes this self-clearing: mark the constant
// @deprecated, and the job goes green again on the next run.
echo "NO LONGER PUBLISHED, and not marked deprecated:\n  " . implode("\n  ", $missingLive) . "\n\n";
echo "Mark each one @deprecated with its shutdown date, and repoint anything used as a default.\n";

exit(1);
