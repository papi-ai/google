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

$source = file_get_contents(__DIR__ . '/../src/GoogleProvider.php');

if ($source === false) {
    fwrite(STDERR, "Cannot read the provider source.\n");

    exit(2);
}

/** @var array<string, array{value: string, deprecated: bool, isDefault: bool}> $shipped */
$shipped = [];
$lines = explode("\n", $source);

// Line by line, not one big regex. A regex that also tries to capture the preceding docblock
// matches across earlier constants and silently drops them, and a checker that quietly skips
// models is worse than no checker: the ones it skipped here were the newest four.
foreach ($lines as $number => $line) {
    if (preg_match('/public const (?P<name>MODEL_\w+|IMAGEN_\w+) = \'(?P<value>[^\']+)\';/', $line, $match) !== 1) {
        continue;
    }

    // Only the line immediately above can carry the marker, which is where php-cs-fixer puts it.
    $shipped[$match['name']] = [
        'value' => $match['value'],
        'deprecated' => str_contains($lines[$number - 1] ?? '', '@deprecated'),
        'isDefault' => false,
    ];
}

// Anything used as a fallback is the dangerous kind: it breaks callers who pass no model at all.
foreach ($shipped as $name => $entry) {
    if (preg_match('/\?\?\s*self::' . preg_quote($name, '/') . '\b/', $source) === 1) {
        $shipped[$name]['isDefault'] = true;
    }
}

if ($shipped === []) {
    fwrite(STDERR, "Found no model constants to check, which means the parser needs updating.\n");

    exit(2);
}

$key = getenv('GEMINI_API_KEY');
$url = 'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000'
    . ($key === false || $key === '' ? '' : '&key=' . urlencode($key));

$response = @file_get_contents($url);

if ($response === false) {
    fwrite(STDERR, "Could not reach the model list. Not treating an unreachable provider as a retirement.\n");

    exit(0);
}

/** @var array{models?: list<array{name?: string}>} $decoded */
$decoded = json_decode($response, true, 512, JSON_THROW_ON_ERROR);
$served = [];

foreach ($decoded['models'] ?? [] as $model) {
    // Names come back as "models/gemini-3.6-flash".
    $served[] = str_replace('models/', '', (string) ($model['name'] ?? ''));
}

if ($served === []) {
    fwrite(STDERR, "The model list came back empty, which is more likely our problem than a mass retirement.\n");

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

// A default with a known shutdown date is a scheduled outage for every caller who does not pass a
// model. It is the exact shape of bug that has hit this package three times, so it fails too.
$deprecatedDefaults = [];

foreach ($shipped as $name => $entry) {
    if ($entry['deprecated'] && $entry['isDefault']) {
        $deprecatedDefaults[] = sprintf('%s (%s)', $name, $entry['value']);
    }
}

if ($missingLive === [] && $deprecatedDefaults === []) {
    echo "Every live model ID is still served, and no default is deprecated.\n";

    exit(0);
}

if ($missingLive !== []) {
    echo "NO LONGER SERVED, and not marked deprecated:\n  " . implode("\n  ", $missingLive) . "\n\n";
    echo "Mark each one @deprecated with its shutdown date, and repoint anything used as a default.\n\n";
}

if ($deprecatedDefaults !== []) {
    echo "DEPRECATED, AND STILL USED AS A DEFAULT:\n  " . implode("\n  ", $deprecatedDefaults) . "\n\n";
    echo "This breaks every caller who passes no model, on the day the provider switches it off.\n";
}

exit(1);
