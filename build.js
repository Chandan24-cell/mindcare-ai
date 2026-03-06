/**
 * Build script to obfuscate JavaScript for production
 * Run: node build.js
 */

const JavaScriptObfuscator = require('javascript-obfuscator');
const fs = require('fs');
const path = require('path');

const INPUT_FILE = path.join(__dirname, 'frontend', 'script.js');
const OUTPUT_FILE = path.join(__dirname, 'frontend', 'script-obf.js');

// Read the source file
const sourceCode = fs.readFileSync(INPUT_FILE, 'utf8');

// Obfuscate with strong settings
const obfuscationResult = JavaScriptObfuscator.obfuscate(
    sourceCode,
    {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 0.75,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.4,
        debugProtection: true,
        debugProtectionInterval: 1000,
        disableConsoleOutput: true,
        identifierNamesGenerator: 'hexadecimal',
        identifiersPrefix: '_',
        ignoreRequireImports: true,
        numbersToExpressions: true,
        output: 'string',
        renameGlobals: true,
        selfDefending: true,
        shuffleStringArray: true,
        stringArray: true,
        stringArrayThreshold: 0.75,
        transformObjectKeys: true,
        unicodeEscapeSequence: true
    }
);

// Write the obfuscated code
fs.writeFileSync(OUTPUT_FILE, obfuscationResult.getObfuscatedCode());

console.log(`✓ Obfuscated JavaScript created: ${OUTPUT_FILE}`);
console.log(`  Original size: ${(sourceCode.length / 1024).toFixed(2)} KB`);
console.log(`  Obfuscated size: ${(obfuscationResult.getObfuscatedCode().length / 1024).toFixed(2)} KB`);

