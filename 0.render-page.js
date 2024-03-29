exports.ids = [0];
exports.modules = {

/***/ "./node_modules/@deckdeckgo/highlight-code/dist/esm/deckdeckgo-highlight-code-languages-93c3ae76.js":
/*!**********************************************************************************************************!*\
  !*** ./node_modules/@deckdeckgo/highlight-code/dist/esm/deckdeckgo-highlight-code-languages-93c3ae76.js ***!
  \**********************************************************************************************************/
/*! exports provided: D, a, d */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "D", function() { return DeckdeckgoHighlightCodeTerminal; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "a", function() { return DeckdeckgoHighlightCodeCarbonTheme; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "d", function() { return deckdeckgoHighlightCodeLanguages; });
var DeckdeckgoHighlightCodeTerminal;
(function (DeckdeckgoHighlightCodeTerminal) {
  DeckdeckgoHighlightCodeTerminal["CARBON"] = "carbon";
  DeckdeckgoHighlightCodeTerminal["UBUNTU"] = "ubuntu";
  DeckdeckgoHighlightCodeTerminal["NONE"] = "none";
})(DeckdeckgoHighlightCodeTerminal || (DeckdeckgoHighlightCodeTerminal = {}));

var DeckdeckgoHighlightCodeCarbonTheme;
(function (DeckdeckgoHighlightCodeCarbonTheme) {
  DeckdeckgoHighlightCodeCarbonTheme["3024-NIGHT"] = "3024-night";
  DeckdeckgoHighlightCodeCarbonTheme["A11Y-DARK"] = "a11y-dark";
  DeckdeckgoHighlightCodeCarbonTheme["BLACKBOARD"] = "blackboard";
  DeckdeckgoHighlightCodeCarbonTheme["BASE16-DARK"] = "base16-dark";
  DeckdeckgoHighlightCodeCarbonTheme["BASE16-LIGHT"] = "base16-light";
  DeckdeckgoHighlightCodeCarbonTheme["COBALT"] = "cobalt";
  DeckdeckgoHighlightCodeCarbonTheme["DRACULA"] = "dracula";
  DeckdeckgoHighlightCodeCarbonTheme["DUOTONE"] = "duotone";
  DeckdeckgoHighlightCodeCarbonTheme["HOPSCOTCH"] = "hopscotch";
  DeckdeckgoHighlightCodeCarbonTheme["LUCARIO"] = "lucario";
  DeckdeckgoHighlightCodeCarbonTheme["MATERIAL"] = "material";
  DeckdeckgoHighlightCodeCarbonTheme["MONOKAI"] = "monokai";
  DeckdeckgoHighlightCodeCarbonTheme["NIGHT-OWL"] = "night-owl";
  DeckdeckgoHighlightCodeCarbonTheme["NORD"] = "nord";
  DeckdeckgoHighlightCodeCarbonTheme["OCEANIC-NEXT"] = "oceanic-next";
  DeckdeckgoHighlightCodeCarbonTheme["ONE-LIGHT"] = "one-light";
  DeckdeckgoHighlightCodeCarbonTheme["ONE-DARK"] = "one-dark";
  DeckdeckgoHighlightCodeCarbonTheme["PANDA"] = "panda";
  DeckdeckgoHighlightCodeCarbonTheme["PARAISO"] = "paraiso";
  DeckdeckgoHighlightCodeCarbonTheme["SETI"] = "seti";
  DeckdeckgoHighlightCodeCarbonTheme["SHADES-OF-PURPLE"] = "shades-of-purple";
  DeckdeckgoHighlightCodeCarbonTheme["SOLARIZED-DARK"] = "solarized-dark";
  DeckdeckgoHighlightCodeCarbonTheme["SOLARIZED-LIGHT"] = "solarized-light";
  DeckdeckgoHighlightCodeCarbonTheme["SYNTHWAVE"] = "synthwave";
  DeckdeckgoHighlightCodeCarbonTheme["TWILIGHT"] = "twilight";
  DeckdeckgoHighlightCodeCarbonTheme["VERMINAL"] = "verminal";
  DeckdeckgoHighlightCodeCarbonTheme["VSCODE"] = "vscode";
  DeckdeckgoHighlightCodeCarbonTheme["YETI"] = "yeti";
  DeckdeckgoHighlightCodeCarbonTheme["ZENBURN"] = "zenburn";
})(DeckdeckgoHighlightCodeCarbonTheme || (DeckdeckgoHighlightCodeCarbonTheme = {}));

const deckdeckgoHighlightCodeLanguages = {
  markup: { title: 'Markup' },
  html: { title: 'HTML', main: 'markup' },
  xml: { title: 'XML', main: 'markup' },
  svg: { title: 'SVG', main: 'markup' },
  mathml: { title: 'MathML', main: 'markup' },
  ssml: { title: 'SSML', main: 'markup' },
  atom: { title: 'Atom', main: 'markup' },
  rss: { title: 'RSS', main: 'markup' },
  css: { title: 'CSS' },
  clike: { title: 'C-like' },
  javascript: { title: 'JavaScript' },
  js: { title: 'js', main: 'javascript' },
  abap: { title: 'ABAP' },
  abnf: { title: 'ABNF' },
  actionscript: { title: 'ActionScript' },
  ada: { title: 'Ada' },
  agda: { title: 'Agda' },
  al: { title: 'AL' },
  antlr4: { title: 'ANTLR4' },
  g4: { title: 'g4', main: 'antlr4' },
  apacheconf: { title: 'Apache Configuration' },
  apex: { title: 'Apex', require: ['sql'] },
  apl: { title: 'APL' },
  applescript: { title: 'AppleScript' },
  aql: { title: 'AQL' },
  arduino: { title: 'Arduino', require: ['cpp'] },
  arff: { title: 'ARFF' },
  asciidoc: { title: 'AsciiDoc' },
  adoc: { title: 'adoc', main: 'asciidoc' },
  aspnet: { title: 'ASP.NET (C#)', require: ['markup', 'csharp'] },
  asm6502: { title: '6502 Assembly' },
  autohotkey: { title: 'AutoHotkey' },
  autoit: { title: 'AutoIt' },
  bash: { title: 'Bash' },
  shell: { title: 'Shell', main: 'bash' },
  basic: { title: 'BASIC' },
  batch: { title: 'Batch' },
  bbcode: { title: 'BBcode' },
  shortcode: { title: 'Shortcode', main: 'bbcode' },
  birb: { title: 'Birb' },
  bison: { title: 'Bison', require: ['c'] },
  bnf: { title: 'BNF' },
  rbnf: { title: 'RBNF', main: 'bnf' },
  brainfuck: { title: 'Brainfuck' },
  brightscript: { title: 'BrightScript' },
  bro: { title: 'Bro' },
  bsl: { title: 'BSL (1C:Enterprise)' },
  oscript: { title: 'OneScript', main: 'bsl' },
  c: { title: 'C' },
  csharp: { title: 'C#' },
  cs: { title: 'cs', main: 'csharp' },
  dotnet: { title: 'dotnet', main: 'csharp' },
  cpp: { title: 'C++', require: ['c'] },
  cfscript: { title: 'CFScript' },
  cfc: { title: 'cfc', main: 'cfscript' },
  chaiscript: { title: 'ChaiScript', require: ['cpp'] },
  cil: { title: 'CIL' },
  clojure: { title: 'Clojure' },
  cmake: { title: 'CMake' },
  cobol: { title: 'COBOL' },
  coffeescript: { title: 'CoffeeScript' },
  coffee: { title: 'coffee', main: 'coffeescript' },
  concurnas: { title: 'Concurnas' },
  conc: { title: 'conc', main: 'concurnas' },
  csp: { title: 'Content-Security-Policy' },
  coq: { title: 'Coq' },
  crystal: { title: 'Crystal', require: ['ruby'] },
  'css-extras': { title: 'CSS Extras', require: ['css'] },
  csv: { title: 'CSV' },
  cypher: { title: 'Cypher' },
  d: { title: 'D' },
  dart: { title: 'Dart' },
  dataweave: { title: 'DataWeave' },
  dax: { title: 'DAX' },
  dhall: { title: 'Dhall' },
  diff: { title: 'Diff' },
  django: { title: 'Django/Jinja2', require: ['markup-templating'] },
  jinja2: { title: 'jinja2', main: 'django', require: ['markup-templating'] },
  'dns-zone-file': { title: 'DNS zone file' },
  'dns-zone': { title: 'dns-zone', main: 'dns-zone-file' },
  docker: { title: 'Docker' },
  dockerfile: { title: 'dockerfile', main: 'docker' },
  dot: { title: 'DOT (Graphviz)' },
  gv: { title: 'gv', main: 'dot' },
  ebnf: { title: 'EBNF' },
  editorconfig: { title: 'EditorConfig' },
  eiffel: { title: 'Eiffel' },
  ejs: { title: 'EJS', require: ['markup-templating'] },
  eta: { title: 'Eta', main: 'ejs', require: ['markup-templating'] },
  elixir: { title: 'Elixir' },
  elm: { title: 'Elm' },
  etlua: { title: 'Embedded Lua templating', require: ['lua', 'markup-templating'] },
  erb: { title: 'ERB', require: ['ruby', 'markup-templating'] },
  erlang: { title: 'Erlang' },
  'excel-formula': { title: 'Excel Formula' },
  xlsx: { title: 'xlsx', main: 'excel-formula' },
  xls: { title: 'xls', main: 'excel-formula' },
  fsharp: { title: 'F#' },
  factor: { title: 'Factor' },
  false: { title: 'False' },
  'firestore-security-rules': { title: 'Firestore security rules' },
  flow: { title: 'Flow' },
  fortran: { title: 'Fortran' },
  ftl: { title: 'FreeMarker Template Language', require: ['markup-templating'] },
  gml: { title: 'GameMaker Language' },
  gamemakerlanguage: { title: 'gamemakerlanguage', main: 'gml' },
  gcode: { title: 'G-code' },
  gdscript: { title: 'GDScript' },
  gedcom: { title: 'GEDCOM' },
  gherkin: { title: 'Gherkin' },
  git: { title: 'Git' },
  glsl: { title: 'GLSL', require: ['c'] },
  go: { title: 'Go' },
  graphql: { title: 'GraphQL' },
  groovy: { title: 'Groovy' },
  haml: { title: 'Haml', require: ['ruby'] },
  handlebars: { title: 'Handlebars', require: ['markup-templating'] },
  haskell: { title: 'Haskell' },
  hs: { title: 'hs', main: 'haskell' },
  haxe: { title: 'Haxe' },
  hcl: { title: 'HCL' },
  hlsl: { title: 'HLSL', require: ['c'] },
  http: { title: 'HTTP' },
  hpkp: { title: 'HTTP Public-Key-Pins' },
  hsts: { title: 'HTTP Strict-Transport-Security' },
  ichigojam: { title: 'IchigoJam' },
  icon: { title: 'Icon' },
  'icu-message-format': { title: 'ICU Message Format' },
  idris: { title: 'Idris', require: ['haskell'] },
  idr: { title: 'idr', main: 'idris', require: ['haskell'] },
  ignore: { title: '.ignore' },
  gitignore: { title: '.gitignore', main: 'ignore' },
  hgignore: { title: '.hgignore', main: 'ignore' },
  npmignore: { title: '.npmignore', main: 'ignore' },
  inform7: { title: 'Inform 7' },
  ini: { title: 'Ini' },
  io: { title: 'Io' },
  j: { title: 'J' },
  java: { title: 'Java' },
  javadoc: { title: 'JavaDoc', require: ['markup', 'java', 'javadoclike'] },
  javadoclike: { title: 'JavaDoc-like' },
  javastacktrace: { title: 'Java stack trace' },
  jexl: { title: 'Jexl' },
  jolie: { title: 'Jolie' },
  jq: { title: 'JQ' },
  jsdoc: { title: 'JSDoc', require: ['javadoclike', 'typescript'] },
  'js-extras': { title: 'JS Extras' },
  json: { title: 'JSON' },
  webmanifest: { title: 'Web App Manifest', main: 'json' },
  json5: { title: 'JSON5', require: ['json'] },
  jsonp: { title: 'JSONP', require: ['json'] },
  jsstacktrace: { title: 'JS stack trace' },
  'js-templates': { title: 'JS Templates' },
  julia: { title: 'Julia' },
  keyman: { title: 'Keyman' },
  kotlin: { title: 'Kotlin' },
  kt: { title: 'kt', main: 'kotlin' },
  kts: { title: 'Kotlin Script', main: 'kotlin' },
  kumir: { title: 'KuMir (КуМир)' },
  kum: { title: 'kum', main: 'kumir' },
  latex: { title: 'LaTeX' },
  tex: { title: 'TeX', main: 'latex' },
  context: { title: 'ConTeXt', main: 'latex' },
  latte: { title: 'Latte', require: ['markup-templating', 'php'] },
  less: { title: 'Less', require: ['css'] },
  lilypond: { title: 'LilyPond', require: ['scheme'] },
  ly: { title: 'ly', main: 'lilypond', require: ['scheme'] },
  liquid: { title: 'Liquid' },
  lisp: { title: 'Lisp' },
  emacs: { title: 'emacs', main: 'lisp' },
  elisp: { title: 'elisp', main: 'lisp' },
  'emacs-lisp': { title: 'emacs-lisp', main: 'lisp' },
  livescript: { title: 'LiveScript' },
  llvm: { title: 'LLVM IR' },
  log: { title: 'Log file' },
  lolcode: { title: 'LOLCODE' },
  lua: { title: 'Lua' },
  makefile: { title: 'Makefile' },
  markdown: { title: 'Markdown', require: ['markup'] },
  md: { title: 'md', main: 'markdown', require: ['markup'] },
  'markup-templating': { title: 'Markup templating', require: ['markup'] },
  matlab: { title: 'MATLAB' },
  mel: { title: 'MEL' },
  mizar: { title: 'Mizar' },
  mongodb: { title: 'MongoDB' },
  monkey: { title: 'Monkey' },
  moonscript: { title: 'MoonScript' },
  moon: { title: 'moon', main: 'moonscript' },
  n1ql: { title: 'N1QL' },
  n4js: { title: 'N4JS' },
  n4jsd: { title: 'n4jsd', main: 'n4js' },
  'nand2tetris-hdl': { title: 'Nand To Tetris HDL' },
  naniscript: { title: 'Naninovel Script' },
  nani: { title: 'nani', main: 'naniscript' },
  nasm: { title: 'NASM' },
  neon: { title: 'NEON' },
  nevod: { title: 'Nevod' },
  nginx: { title: 'nginx' },
  nim: { title: 'Nim' },
  nix: { title: 'Nix' },
  nsis: { title: 'NSIS' },
  objectivec: { title: 'Objective-C', require: ['c'] },
  objc: { title: 'objc', main: 'objectivec', require: ['c'] },
  ocaml: { title: 'OCaml' },
  opencl: { title: 'OpenCL', require: ['c'] },
  openqasm: { title: 'OpenQasm' },
  qasm: { title: 'qasm', main: 'openqasm' },
  oz: { title: 'Oz' },
  parigp: { title: 'PARI/GP' },
  parser: { title: 'Parser', require: ['markup'] },
  pascal: { title: 'Pascal' },
  objectpascal: { title: 'Object Pascal', main: 'pascal' },
  pascaligo: { title: 'Pascaligo' },
  psl: { title: 'PATROL Scripting Language' },
  pcaxis: { title: 'PC-Axis' },
  px: { title: 'px', main: 'pcaxis' },
  peoplecode: { title: 'PeopleCode' },
  pcode: { title: 'pcode', main: 'peoplecode' },
  perl: { title: 'Perl' },
  php: { title: 'PHP', require: ['markup-templating'] },
  phpdoc: { title: 'PHPDoc', require: ['php', 'javadoclike'] },
  'php-extras': { title: 'PHP Extras', require: ['php'] },
  plsql: { title: 'PL/SQL', require: ['sql'] },
  powerquery: { title: 'PowerQuery' },
  pq: { title: 'pq', main: 'powerquery' },
  mscript: { title: 'mscript', main: 'powerquery' },
  powershell: { title: 'PowerShell' },
  processing: { title: 'Processing' },
  prolog: { title: 'Prolog' },
  promql: { title: 'PromQL' },
  properties: { title: '.properties' },
  protobuf: { title: 'Protocol Buffers' },
  pug: { title: 'Pug', require: ['markup'] },
  puppet: { title: 'Puppet' },
  pure: { title: 'Pure' },
  purebasic: { title: 'PureBasic' },
  pbfasm: { title: 'pbfasm', main: 'purebasic' },
  purescript: { title: 'PureScript', require: ['haskell'] },
  purs: { title: 'purs', main: 'purescript', require: ['haskell'] },
  python: { title: 'Python' },
  py: { title: 'py', main: 'python' },
  qsharp: { title: 'Q#' },
  qs: { title: 'qs', main: 'qsharp' },
  q: { title: 'Q (kdb+ database)' },
  qml: { title: 'QML' },
  qore: { title: 'Qore' },
  r: { title: 'R' },
  racket: { title: 'Racket', require: ['scheme'] },
  rkt: { title: 'rkt', main: 'racket', require: ['scheme'] },
  jsx: { title: 'React JSX', require: ['markup'] },
  tsx: { title: 'React TSX', require: ['jsx', 'typescript'] },
  reason: { title: 'Reason' },
  regex: { title: 'Regex' },
  rego: { title: 'Rego' },
  renpy: { title: "Ren'py" },
  rpy: { title: 'rpy', main: 'renpy' },
  rest: { title: 'reST (reStructuredText)' },
  rip: { title: 'Rip' },
  roboconf: { title: 'Roboconf' },
  robotframework: { title: 'Robot Framework' },
  robot: { title: 'robot', main: 'robotframework' },
  ruby: { title: 'Ruby' },
  rb: { title: 'rb', main: 'ruby' },
  rust: { title: 'Rust' },
  sas: { title: 'SAS' },
  sass: { title: 'Sass (Sass)', require: ['css'] },
  scss: { title: 'Sass (Scss)', require: ['css'] },
  scala: { title: 'Scala', require: ['java'] },
  scheme: { title: 'Scheme' },
  'shell-session': { title: 'Shell session', require: ['bash'] },
  'sh-session': { title: 'sh-session', main: 'shell-session', require: ['bash'] },
  shellsession: { title: 'shellsession', main: 'shell-session', require: ['bash'] },
  smali: { title: 'Smali' },
  smalltalk: { title: 'Smalltalk' },
  smarty: { title: 'Smarty', require: ['markup-templating'] },
  sml: { title: 'SML' },
  smlnj: { title: 'SML/NJ', main: 'sml' },
  solidity: { title: 'Solidity (Ethereum)' },
  sol: { title: 'sol', main: 'solidity' },
  'solution-file': { title: 'Solution file' },
  sln: { title: 'sln', main: 'solution-file' },
  soy: { title: 'Soy (Closure Template)', require: ['markup-templating'] },
  sparql: { title: 'SPARQL', require: ['turtle'] },
  rq: { title: 'rq', main: 'sparql', require: ['turtle'] },
  'splunk-spl': { title: 'Splunk SPL' },
  sqf: { title: 'SQF: Status Quo Function (Arma 3)' },
  sql: { title: 'SQL' },
  squirrel: { title: 'Squirrel' },
  stan: { title: 'Stan' },
  iecst: { title: 'Structured Text (IEC 61131-3)' },
  stylus: { title: 'Stylus' },
  swift: { title: 'Swift' },
  't4-templating': { title: 'T4 templating' },
  't4-cs': { title: 'T4 Text Templates (C#)', require: ['t4-templating', 'csharp'] },
  t4: { title: 't4', main: 't4-cs', require: ['t4-templating', 'csharp'] },
  't4-vb': { title: 'T4 Text Templates (VB)', require: ['t4-templating', 'vbnet'] },
  tap: { title: 'TAP', require: ['yaml'] },
  tcl: { title: 'Tcl' },
  tt2: { title: 'Template Toolkit 2', require: ['markup-templating'] },
  textile: { title: 'Textile', require: ['markup'] },
  toml: { title: 'TOML' },
  turtle: { title: 'Turtle' },
  trig: { title: 'TriG', main: 'turtle' },
  twig: { title: 'Twig', require: ['markup'] },
  typescript: { title: 'TypeScript' },
  ts: { title: 'ts', main: 'typescript' },
  typoscript: { title: 'TypoScript' },
  tsconfig: { title: 'TSConfig', main: 'typoscript' },
  unrealscript: { title: 'UnrealScript' },
  uscript: { title: 'uscript', main: 'unrealscript' },
  uc: { title: 'uc', main: 'unrealscript' },
  uri: { title: 'URI' },
  url: { title: 'URL', main: 'uri' },
  v: { title: 'V' },
  vala: { title: 'Vala' },
  vbnet: { title: 'VB.Net', require: ['basic'] },
  velocity: { title: 'Velocity', require: ['markup'] },
  verilog: { title: 'Verilog' },
  vhdl: { title: 'VHDL' },
  vim: { title: 'vim' },
  'visual-basic': { title: 'Visual Basic' },
  vb: { title: 'vb', main: 'visual-basic' },
  vba: { title: 'VBA', main: 'visual-basic' },
  warpscript: { title: 'WarpScript' },
  wasm: { title: 'WebAssembly' },
  wiki: { title: 'Wiki markup', require: ['markup'] },
  xeora: { title: 'Xeora', require: ['markup'] },
  xeoracube: { title: 'XeoraCube', main: 'xeora', require: ['markup'] },
  'xml-doc': { title: 'XML doc (.net)', require: ['markup'] },
  xojo: { title: 'Xojo (REALbasic)' },
  xquery: { title: 'XQuery', require: ['markup'] },
  yaml: { title: 'YAML' },
  yml: { title: 'yml', main: 'yaml' },
  yang: { title: 'YANG' },
  zig: { title: 'Zig' },
};




/***/ }),

/***/ "./node_modules/@deckdeckgo/highlight-code/dist/esm/deckgo-highlight-code.entry.js":
/*!*****************************************************************************************!*\
  !*** ./node_modules/@deckdeckgo/highlight-code/dist/esm/deckgo-highlight-code.entry.js ***!
  \*****************************************************************************************/
/*! exports provided: deckgo_highlight_code */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "deckgo_highlight_code", function() { return DeckdeckgoHighlightCode; });
/* harmony import */ var _index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./index-5142f31b.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/index-5142f31b.js");
/* harmony import */ var _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./deckdeckgo-highlight-code-languages-93c3ae76.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/deckdeckgo-highlight-code-languages-93c3ae76.js");



var commonjsGlobal = typeof globalThis !== 'undefined' ? globalThis : typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : typeof self !== 'undefined' ? self : {};

function createCommonjsModule(fn, basedir, module) {
	return module = {
	  path: basedir,
	  exports: {},
	  require: function (path, base) {
    return commonjsRequire();
  }
	}, fn(module, module.exports), module.exports;
}

function commonjsRequire () {
	throw new Error('Dynamic requires are not currently supported by @rollup/plugin-commonjs');
}

var prism = createCommonjsModule(function (module) {
/* **********************************************
     Begin prism-core.js
********************************************** */

/// <reference lib="WebWorker"/>

var _self = (typeof window !== 'undefined')
	? window   // if in browser
	: (
		(typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope)
		? self // if in worker
		: {}   // if in node js
	);

/**
 * Prism: Lightweight, robust, elegant syntax highlighting
 *
 * @license MIT <https://opensource.org/licenses/MIT>
 * @author Lea Verou <https://lea.verou.me>
 * @namespace
 * @public
 */
var Prism = (function (_self){

// Private helper vars
var lang = /\blang(?:uage)?-([\w-]+)\b/i;
var uniqueId = 0;


var _ = {
	/**
	 * By default, Prism will attempt to highlight all code elements (by calling {@link Prism.highlightAll}) on the
	 * current page after the page finished loading. This might be a problem if e.g. you wanted to asynchronously load
	 * additional languages or plugins yourself.
	 *
	 * By setting this value to `true`, Prism will not automatically highlight all code elements on the page.
	 *
	 * You obviously have to change this value before the automatic highlighting started. To do this, you can add an
	 * empty Prism object into the global scope before loading the Prism script like this:
	 *
	 * ```js
	 * window.Prism = window.Prism || {};
	 * Prism.manual = true;
	 * // add a new <script> to load Prism's script
	 * ```
	 *
	 * @default false
	 * @type {boolean}
	 * @memberof Prism
	 * @public
	 */
	manual: _self.Prism && _self.Prism.manual,
	disableWorkerMessageHandler: _self.Prism && _self.Prism.disableWorkerMessageHandler,

	/**
	 * A namespace for utility methods.
	 *
	 * All function in this namespace that are not explicitly marked as _public_ are for __internal use only__ and may
	 * change or disappear at any time.
	 *
	 * @namespace
	 * @memberof Prism
	 */
	util: {
		encode: function encode(tokens) {
			if (tokens instanceof Token) {
				return new Token(tokens.type, encode(tokens.content), tokens.alias);
			} else if (Array.isArray(tokens)) {
				return tokens.map(encode);
			} else {
				return tokens.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/\u00a0/g, ' ');
			}
		},

		/**
		 * Returns the name of the type of the given value.
		 *
		 * @param {any} o
		 * @returns {string}
		 * @example
		 * type(null)      === 'Null'
		 * type(undefined) === 'Undefined'
		 * type(123)       === 'Number'
		 * type('foo')     === 'String'
		 * type(true)      === 'Boolean'
		 * type([1, 2])    === 'Array'
		 * type({})        === 'Object'
		 * type(String)    === 'Function'
		 * type(/abc+/)    === 'RegExp'
		 */
		type: function (o) {
			return Object.prototype.toString.call(o).slice(8, -1);
		},

		/**
		 * Returns a unique number for the given object. Later calls will still return the same number.
		 *
		 * @param {Object} obj
		 * @returns {number}
		 */
		objId: function (obj) {
			if (!obj['__id']) {
				Object.defineProperty(obj, '__id', { value: ++uniqueId });
			}
			return obj['__id'];
		},

		/**
		 * Creates a deep clone of the given object.
		 *
		 * The main intended use of this function is to clone language definitions.
		 *
		 * @param {T} o
		 * @param {Record<number, any>} [visited]
		 * @returns {T}
		 * @template T
		 */
		clone: function deepClone(o, visited) {
			visited = visited || {};

			var clone, id;
			switch (_.util.type(o)) {
				case 'Object':
					id = _.util.objId(o);
					if (visited[id]) {
						return visited[id];
					}
					clone = /** @type {Record<string, any>} */ ({});
					visited[id] = clone;

					for (var key in o) {
						if (o.hasOwnProperty(key)) {
							clone[key] = deepClone(o[key], visited);
						}
					}

					return /** @type {any} */ (clone);

				case 'Array':
					id = _.util.objId(o);
					if (visited[id]) {
						return visited[id];
					}
					clone = [];
					visited[id] = clone;

					(/** @type {Array} */(/** @type {any} */(o))).forEach(function (v, i) {
						clone[i] = deepClone(v, visited);
					});

					return /** @type {any} */ (clone);

				default:
					return o;
			}
		},

		/**
		 * Returns the Prism language of the given element set by a `language-xxxx` or `lang-xxxx` class.
		 *
		 * If no language is set for the element or the element is `null` or `undefined`, `none` will be returned.
		 *
		 * @param {Element} element
		 * @returns {string}
		 */
		getLanguage: function (element) {
			while (element && !lang.test(element.className)) {
				element = element.parentElement;
			}
			if (element) {
				return (element.className.match(lang) || [, 'none'])[1].toLowerCase();
			}
			return 'none';
		},

		/**
		 * Returns the script element that is currently executing.
		 *
		 * This does __not__ work for line script element.
		 *
		 * @returns {HTMLScriptElement | null}
		 */
		currentScript: function () {
			if (typeof document === 'undefined') {
				return null;
			}
			if ('currentScript' in document && 1 < 2 /* hack to trip TS' flow analysis */) {
				return /** @type {any} */ (document.currentScript);
			}

			// IE11 workaround
			// we'll get the src of the current script by parsing IE11's error stack trace
			// this will not work for inline scripts

			try {
				throw new Error();
			} catch (err) {
				// Get file src url from stack. Specifically works with the format of stack traces in IE.
				// A stack will look like this:
				//
				// Error
				//    at _.util.currentScript (http://localhost/components/prism-core.js:119:5)
				//    at Global code (http://localhost/components/prism-core.js:606:1)

				var src = (/at [^(\r\n]*\((.*):.+:.+\)$/i.exec(err.stack) || [])[1];
				if (src) {
					var scripts = document.getElementsByTagName('script');
					for (var i in scripts) {
						if (scripts[i].src == src) {
							return scripts[i];
						}
					}
				}
				return null;
			}
		},

		/**
		 * Returns whether a given class is active for `element`.
		 *
		 * The class can be activated if `element` or one of its ancestors has the given class and it can be deactivated
		 * if `element` or one of its ancestors has the negated version of the given class. The _negated version_ of the
		 * given class is just the given class with a `no-` prefix.
		 *
		 * Whether the class is active is determined by the closest ancestor of `element` (where `element` itself is
		 * closest ancestor) that has the given class or the negated version of it. If neither `element` nor any of its
		 * ancestors have the given class or the negated version of it, then the default activation will be returned.
		 *
		 * In the paradoxical situation where the closest ancestor contains __both__ the given class and the negated
		 * version of it, the class is considered active.
		 *
		 * @param {Element} element
		 * @param {string} className
		 * @param {boolean} [defaultActivation=false]
		 * @returns {boolean}
		 */
		isActive: function (element, className, defaultActivation) {
			var no = 'no-' + className;

			while (element) {
				var classList = element.classList;
				if (classList.contains(className)) {
					return true;
				}
				if (classList.contains(no)) {
					return false;
				}
				element = element.parentElement;
			}
			return !!defaultActivation;
		}
	},

	/**
	 * This namespace contains all currently loaded languages and the some helper functions to create and modify languages.
	 *
	 * @namespace
	 * @memberof Prism
	 * @public
	 */
	languages: {
		/**
		 * Creates a deep copy of the language with the given id and appends the given tokens.
		 *
		 * If a token in `redef` also appears in the copied language, then the existing token in the copied language
		 * will be overwritten at its original position.
		 *
		 * ## Best practices
		 *
		 * Since the position of overwriting tokens (token in `redef` that overwrite tokens in the copied language)
		 * doesn't matter, they can technically be in any order. However, this can be confusing to others that trying to
		 * understand the language definition because, normally, the order of tokens matters in Prism grammars.
		 *
		 * Therefore, it is encouraged to order overwriting tokens according to the positions of the overwritten tokens.
		 * Furthermore, all non-overwriting tokens should be placed after the overwriting ones.
		 *
		 * @param {string} id The id of the language to extend. This has to be a key in `Prism.languages`.
		 * @param {Grammar} redef The new tokens to append.
		 * @returns {Grammar} The new language created.
		 * @public
		 * @example
		 * Prism.languages['css-with-colors'] = Prism.languages.extend('css', {
		 *     // Prism.languages.css already has a 'comment' token, so this token will overwrite CSS' 'comment' token
		 *     // at its original position
		 *     'comment': { ... },
		 *     // CSS doesn't have a 'color' token, so this token will be appended
		 *     'color': /\b(?:red|green|blue)\b/
		 * });
		 */
		extend: function (id, redef) {
			var lang = _.util.clone(_.languages[id]);

			for (var key in redef) {
				lang[key] = redef[key];
			}

			return lang;
		},

		/**
		 * Inserts tokens _before_ another token in a language definition or any other grammar.
		 *
		 * ## Usage
		 *
		 * This helper method makes it easy to modify existing languages. For example, the CSS language definition
		 * not only defines CSS highlighting for CSS documents, but also needs to define highlighting for CSS embedded
		 * in HTML through `<style>` elements. To do this, it needs to modify `Prism.languages.markup` and add the
		 * appropriate tokens. However, `Prism.languages.markup` is a regular JavaScript object literal, so if you do
		 * this:
		 *
		 * ```js
		 * Prism.languages.markup.style = {
		 *     // token
		 * };
		 * ```
		 *
		 * then the `style` token will be added (and processed) at the end. `insertBefore` allows you to insert tokens
		 * before existing tokens. For the CSS example above, you would use it like this:
		 *
		 * ```js
		 * Prism.languages.insertBefore('markup', 'cdata', {
		 *     'style': {
		 *         // token
		 *     }
		 * });
		 * ```
		 *
		 * ## Special cases
		 *
		 * If the grammars of `inside` and `insert` have tokens with the same name, the tokens in `inside`'s grammar
		 * will be ignored.
		 *
		 * This behavior can be used to insert tokens after `before`:
		 *
		 * ```js
		 * Prism.languages.insertBefore('markup', 'comment', {
		 *     'comment': Prism.languages.markup.comment,
		 *     // tokens after 'comment'
		 * });
		 * ```
		 *
		 * ## Limitations
		 *
		 * The main problem `insertBefore` has to solve is iteration order. Since ES2015, the iteration order for object
		 * properties is guaranteed to be the insertion order (except for integer keys) but some browsers behave
		 * differently when keys are deleted and re-inserted. So `insertBefore` can't be implemented by temporarily
		 * deleting properties which is necessary to insert at arbitrary positions.
		 *
		 * To solve this problem, `insertBefore` doesn't actually insert the given tokens into the target object.
		 * Instead, it will create a new object and replace all references to the target object with the new one. This
		 * can be done without temporarily deleting properties, so the iteration order is well-defined.
		 *
		 * However, only references that can be reached from `Prism.languages` or `insert` will be replaced. I.e. if
		 * you hold the target object in a variable, then the value of the variable will not change.
		 *
		 * ```js
		 * var oldMarkup = Prism.languages.markup;
		 * var newMarkup = Prism.languages.insertBefore('markup', 'comment', { ... });
		 *
		 * assert(oldMarkup !== Prism.languages.markup);
		 * assert(newMarkup === Prism.languages.markup);
		 * ```
		 *
		 * @param {string} inside The property of `root` (e.g. a language id in `Prism.languages`) that contains the
		 * object to be modified.
		 * @param {string} before The key to insert before.
		 * @param {Grammar} insert An object containing the key-value pairs to be inserted.
		 * @param {Object<string, any>} [root] The object containing `inside`, i.e. the object that contains the
		 * object to be modified.
		 *
		 * Defaults to `Prism.languages`.
		 * @returns {Grammar} The new grammar object.
		 * @public
		 */
		insertBefore: function (inside, before, insert, root) {
			root = root || /** @type {any} */ (_.languages);
			var grammar = root[inside];
			/** @type {Grammar} */
			var ret = {};

			for (var token in grammar) {
				if (grammar.hasOwnProperty(token)) {

					if (token == before) {
						for (var newToken in insert) {
							if (insert.hasOwnProperty(newToken)) {
								ret[newToken] = insert[newToken];
							}
						}
					}

					// Do not insert token which also occur in insert. See #1525
					if (!insert.hasOwnProperty(token)) {
						ret[token] = grammar[token];
					}
				}
			}

			var old = root[inside];
			root[inside] = ret;

			// Update references in other language definitions
			_.languages.DFS(_.languages, function(key, value) {
				if (value === old && key != inside) {
					this[key] = ret;
				}
			});

			return ret;
		},

		// Traverse a language definition with Depth First Search
		DFS: function DFS(o, callback, type, visited) {
			visited = visited || {};

			var objId = _.util.objId;

			for (var i in o) {
				if (o.hasOwnProperty(i)) {
					callback.call(o, i, o[i], type || i);

					var property = o[i],
					    propertyType = _.util.type(property);

					if (propertyType === 'Object' && !visited[objId(property)]) {
						visited[objId(property)] = true;
						DFS(property, callback, null, visited);
					}
					else if (propertyType === 'Array' && !visited[objId(property)]) {
						visited[objId(property)] = true;
						DFS(property, callback, i, visited);
					}
				}
			}
		}
	},

	plugins: {},

	/**
	 * This is the most high-level function in Prism’s API.
	 * It fetches all the elements that have a `.language-xxxx` class and then calls {@link Prism.highlightElement} on
	 * each one of them.
	 *
	 * This is equivalent to `Prism.highlightAllUnder(document, async, callback)`.
	 *
	 * @param {boolean} [async=false] Same as in {@link Prism.highlightAllUnder}.
	 * @param {HighlightCallback} [callback] Same as in {@link Prism.highlightAllUnder}.
	 * @memberof Prism
	 * @public
	 */
	highlightAll: function(async, callback) {
		_.highlightAllUnder(document, async, callback);
	},

	/**
	 * Fetches all the descendants of `container` that have a `.language-xxxx` class and then calls
	 * {@link Prism.highlightElement} on each one of them.
	 *
	 * The following hooks will be run:
	 * 1. `before-highlightall`
	 * 2. All hooks of {@link Prism.highlightElement} for each element.
	 *
	 * @param {ParentNode} container The root element, whose descendants that have a `.language-xxxx` class will be highlighted.
	 * @param {boolean} [async=false] Whether each element is to be highlighted asynchronously using Web Workers.
	 * @param {HighlightCallback} [callback] An optional callback to be invoked on each element after its highlighting is done.
	 * @memberof Prism
	 * @public
	 */
	highlightAllUnder: function(container, async, callback) {
		var env = {
			callback: callback,
			container: container,
			selector: 'code[class*="language-"], [class*="language-"] code, code[class*="lang-"], [class*="lang-"] code'
		};

		_.hooks.run('before-highlightall', env);

		env.elements = Array.prototype.slice.apply(env.container.querySelectorAll(env.selector));

		_.hooks.run('before-all-elements-highlight', env);

		for (var i = 0, element; element = env.elements[i++];) {
			_.highlightElement(element, async === true, env.callback);
		}
	},

	/**
	 * Highlights the code inside a single element.
	 *
	 * The following hooks will be run:
	 * 1. `before-sanity-check`
	 * 2. `before-highlight`
	 * 3. All hooks of {@link Prism.highlight}. These hooks will only be run by the current worker if `async` is `true`.
	 * 4. `before-insert`
	 * 5. `after-highlight`
	 * 6. `complete`
	 *
	 * @param {Element} element The element containing the code.
	 * It must have a class of `language-xxxx` to be processed, where `xxxx` is a valid language identifier.
	 * @param {boolean} [async=false] Whether the element is to be highlighted asynchronously using Web Workers
	 * to improve performance and avoid blocking the UI when highlighting very large chunks of code. This option is
	 * [disabled by default](https://prismjs.com/faq.html#why-is-asynchronous-highlighting-disabled-by-default).
	 *
	 * Note: All language definitions required to highlight the code must be included in the main `prism.js` file for
	 * asynchronous highlighting to work. You can build your own bundle on the
	 * [Download page](https://prismjs.com/download.html).
	 * @param {HighlightCallback} [callback] An optional callback to be invoked after the highlighting is done.
	 * Mostly useful when `async` is `true`, since in that case, the highlighting is done asynchronously.
	 * @memberof Prism
	 * @public
	 */
	highlightElement: function(element, async, callback) {
		// Find language
		var language = _.util.getLanguage(element);
		var grammar = _.languages[language];

		// Set language on the element, if not present
		element.className = element.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;

		// Set language on the parent, for styling
		var parent = element.parentElement;
		if (parent && parent.nodeName.toLowerCase() === 'pre') {
			parent.className = parent.className.replace(lang, '').replace(/\s+/g, ' ') + ' language-' + language;
		}

		var code = element.textContent;

		var env = {
			element: element,
			language: language,
			grammar: grammar,
			code: code
		};

		function insertHighlightedCode(highlightedCode) {
			env.highlightedCode = highlightedCode;

			_.hooks.run('before-insert', env);

			env.element.innerHTML = env.highlightedCode;

			_.hooks.run('after-highlight', env);
			_.hooks.run('complete', env);
			callback && callback.call(env.element);
		}

		_.hooks.run('before-sanity-check', env);

		if (!env.code) {
			_.hooks.run('complete', env);
			callback && callback.call(env.element);
			return;
		}

		_.hooks.run('before-highlight', env);

		if (!env.grammar) {
			insertHighlightedCode(_.util.encode(env.code));
			return;
		}

		if (async && _self.Worker) {
			var worker = new Worker(_.filename);

			worker.onmessage = function(evt) {
				insertHighlightedCode(evt.data);
			};

			worker.postMessage(JSON.stringify({
				language: env.language,
				code: env.code,
				immediateClose: true
			}));
		}
		else {
			insertHighlightedCode(_.highlight(env.code, env.grammar, env.language));
		}
	},

	/**
	 * Low-level function, only use if you know what you’re doing. It accepts a string of text as input
	 * and the language definitions to use, and returns a string with the HTML produced.
	 *
	 * The following hooks will be run:
	 * 1. `before-tokenize`
	 * 2. `after-tokenize`
	 * 3. `wrap`: On each {@link Token}.
	 *
	 * @param {string} text A string with the code to be highlighted.
	 * @param {Grammar} grammar An object containing the tokens to use.
	 *
	 * Usually a language definition like `Prism.languages.markup`.
	 * @param {string} language The name of the language definition passed to `grammar`.
	 * @returns {string} The highlighted HTML.
	 * @memberof Prism
	 * @public
	 * @example
	 * Prism.highlight('var foo = true;', Prism.languages.javascript, 'javascript');
	 */
	highlight: function (text, grammar, language) {
		var env = {
			code: text,
			grammar: grammar,
			language: language
		};
		_.hooks.run('before-tokenize', env);
		env.tokens = _.tokenize(env.code, env.grammar);
		_.hooks.run('after-tokenize', env);
		return Token.stringify(_.util.encode(env.tokens), env.language);
	},

	/**
	 * This is the heart of Prism, and the most low-level function you can use. It accepts a string of text as input
	 * and the language definitions to use, and returns an array with the tokenized code.
	 *
	 * When the language definition includes nested tokens, the function is called recursively on each of these tokens.
	 *
	 * This method could be useful in other contexts as well, as a very crude parser.
	 *
	 * @param {string} text A string with the code to be highlighted.
	 * @param {Grammar} grammar An object containing the tokens to use.
	 *
	 * Usually a language definition like `Prism.languages.markup`.
	 * @returns {TokenStream} An array of strings and tokens, a token stream.
	 * @memberof Prism
	 * @public
	 * @example
	 * let code = `var foo = 0;`;
	 * let tokens = Prism.tokenize(code, Prism.languages.javascript);
	 * tokens.forEach(token => {
	 *     if (token instanceof Prism.Token && token.type === 'number') {
	 *         console.log(`Found numeric literal: ${token.content}`);
	 *     }
	 * });
	 */
	tokenize: function(text, grammar) {
		var rest = grammar.rest;
		if (rest) {
			for (var token in rest) {
				grammar[token] = rest[token];
			}

			delete grammar.rest;
		}

		var tokenList = new LinkedList();
		addAfter(tokenList, tokenList.head, text);

		matchGrammar(text, tokenList, grammar, tokenList.head, 0);

		return toArray(tokenList);
	},

	/**
	 * @namespace
	 * @memberof Prism
	 * @public
	 */
	hooks: {
		all: {},

		/**
		 * Adds the given callback to the list of callbacks for the given hook.
		 *
		 * The callback will be invoked when the hook it is registered for is run.
		 * Hooks are usually directly run by a highlight function but you can also run hooks yourself.
		 *
		 * One callback function can be registered to multiple hooks and the same hook multiple times.
		 *
		 * @param {string} name The name of the hook.
		 * @param {HookCallback} callback The callback function which is given environment variables.
		 * @public
		 */
		add: function (name, callback) {
			var hooks = _.hooks.all;

			hooks[name] = hooks[name] || [];

			hooks[name].push(callback);
		},

		/**
		 * Runs a hook invoking all registered callbacks with the given environment variables.
		 *
		 * Callbacks will be invoked synchronously and in the order in which they were registered.
		 *
		 * @param {string} name The name of the hook.
		 * @param {Object<string, any>} env The environment variables of the hook passed to all callbacks registered.
		 * @public
		 */
		run: function (name, env) {
			var callbacks = _.hooks.all[name];

			if (!callbacks || !callbacks.length) {
				return;
			}

			for (var i=0, callback; callback = callbacks[i++];) {
				callback(env);
			}
		}
	},

	Token: Token
};
_self.Prism = _;


// Typescript note:
// The following can be used to import the Token type in JSDoc:
//
//   @typedef {InstanceType<import("./prism-core")["Token"]>} Token

/**
 * Creates a new token.
 *
 * @param {string} type See {@link Token#type type}
 * @param {string | TokenStream} content See {@link Token#content content}
 * @param {string|string[]} [alias] The alias(es) of the token.
 * @param {string} [matchedStr=""] A copy of the full string this token was created from.
 * @class
 * @global
 * @public
 */
function Token(type, content, alias, matchedStr) {
	/**
	 * The type of the token.
	 *
	 * This is usually the key of a pattern in a {@link Grammar}.
	 *
	 * @type {string}
	 * @see GrammarToken
	 * @public
	 */
	this.type = type;
	/**
	 * The strings or tokens contained by this token.
	 *
	 * This will be a token stream if the pattern matched also defined an `inside` grammar.
	 *
	 * @type {string | TokenStream}
	 * @public
	 */
	this.content = content;
	/**
	 * The alias(es) of the token.
	 *
	 * @type {string|string[]}
	 * @see GrammarToken
	 * @public
	 */
	this.alias = alias;
	// Copy of the full string this token was created from
	this.length = (matchedStr || '').length | 0;
}

/**
 * A token stream is an array of strings and {@link Token Token} objects.
 *
 * Token streams have to fulfill a few properties that are assumed by most functions (mostly internal ones) that process
 * them.
 *
 * 1. No adjacent strings.
 * 2. No empty strings.
 *
 *    The only exception here is the token stream that only contains the empty string and nothing else.
 *
 * @typedef {Array<string | Token>} TokenStream
 * @global
 * @public
 */

/**
 * Converts the given token or token stream to an HTML representation.
 *
 * The following hooks will be run:
 * 1. `wrap`: On each {@link Token}.
 *
 * @param {string | Token | TokenStream} o The token or token stream to be converted.
 * @param {string} language The name of current language.
 * @returns {string} The HTML representation of the token or token stream.
 * @memberof Token
 * @static
 */
Token.stringify = function stringify(o, language) {
	if (typeof o == 'string') {
		return o;
	}
	if (Array.isArray(o)) {
		var s = '';
		o.forEach(function (e) {
			s += stringify(e, language);
		});
		return s;
	}

	var env = {
		type: o.type,
		content: stringify(o.content, language),
		tag: 'span',
		classes: ['token', o.type],
		attributes: {},
		language: language
	};

	var aliases = o.alias;
	if (aliases) {
		if (Array.isArray(aliases)) {
			Array.prototype.push.apply(env.classes, aliases);
		} else {
			env.classes.push(aliases);
		}
	}

	_.hooks.run('wrap', env);

	var attributes = '';
	for (var name in env.attributes) {
		attributes += ' ' + name + '="' + (env.attributes[name] || '').replace(/"/g, '&quot;') + '"';
	}

	return '<' + env.tag + ' class="' + env.classes.join(' ') + '"' + attributes + '>' + env.content + '</' + env.tag + '>';
};

/**
 * @param {string} text
 * @param {LinkedList<string | Token>} tokenList
 * @param {any} grammar
 * @param {LinkedListNode<string | Token>} startNode
 * @param {number} startPos
 * @param {RematchOptions} [rematch]
 * @returns {void}
 * @private
 *
 * @typedef RematchOptions
 * @property {string} cause
 * @property {number} reach
 */
function matchGrammar(text, tokenList, grammar, startNode, startPos, rematch) {
	for (var token in grammar) {
		if (!grammar.hasOwnProperty(token) || !grammar[token]) {
			continue;
		}

		var patterns = grammar[token];
		patterns = Array.isArray(patterns) ? patterns : [patterns];

		for (var j = 0; j < patterns.length; ++j) {
			if (rematch && rematch.cause == token + ',' + j) {
				return;
			}

			var patternObj = patterns[j],
				inside = patternObj.inside,
				lookbehind = !!patternObj.lookbehind,
				greedy = !!patternObj.greedy,
				lookbehindLength = 0,
				alias = patternObj.alias;

			if (greedy && !patternObj.pattern.global) {
				// Without the global flag, lastIndex won't work
				var flags = patternObj.pattern.toString().match(/[imsuy]*$/)[0];
				patternObj.pattern = RegExp(patternObj.pattern.source, flags + 'g');
			}

			/** @type {RegExp} */
			var pattern = patternObj.pattern || patternObj;

			for ( // iterate the token list and keep track of the current token/string position
				var currentNode = startNode.next, pos = startPos;
				currentNode !== tokenList.tail;
				pos += currentNode.value.length, currentNode = currentNode.next
			) {

				if (rematch && pos >= rematch.reach) {
					break;
				}

				var str = currentNode.value;

				if (tokenList.length > text.length) {
					// Something went terribly wrong, ABORT, ABORT!
					return;
				}

				if (str instanceof Token) {
					continue;
				}

				var removeCount = 1; // this is the to parameter of removeBetween

				if (greedy && currentNode != tokenList.tail.prev) {
					pattern.lastIndex = pos;
					var match = pattern.exec(text);
					if (!match) {
						break;
					}

					var from = match.index + (lookbehind && match[1] ? match[1].length : 0);
					var to = match.index + match[0].length;
					var p = pos;

					// find the node that contains the match
					p += currentNode.value.length;
					while (from >= p) {
						currentNode = currentNode.next;
						p += currentNode.value.length;
					}
					// adjust pos (and p)
					p -= currentNode.value.length;
					pos = p;

					// the current node is a Token, then the match starts inside another Token, which is invalid
					if (currentNode.value instanceof Token) {
						continue;
					}

					// find the last node which is affected by this match
					for (
						var k = currentNode;
						k !== tokenList.tail && (p < to || typeof k.value === 'string');
						k = k.next
					) {
						removeCount++;
						p += k.value.length;
					}
					removeCount--;

					// replace with the new match
					str = text.slice(pos, p);
					match.index -= pos;
				} else {
					pattern.lastIndex = 0;

					var match = pattern.exec(str);
				}

				if (!match) {
					continue;
				}

				if (lookbehind) {
					lookbehindLength = match[1] ? match[1].length : 0;
				}

				var from = match.index + lookbehindLength,
					matchStr = match[0].slice(lookbehindLength),
					to = from + matchStr.length,
					before = str.slice(0, from),
					after = str.slice(to);

				var reach = pos + str.length;
				if (rematch && reach > rematch.reach) {
					rematch.reach = reach;
				}

				var removeFrom = currentNode.prev;

				if (before) {
					removeFrom = addAfter(tokenList, removeFrom, before);
					pos += before.length;
				}

				removeRange(tokenList, removeFrom, removeCount);

				var wrapped = new Token(token, inside ? _.tokenize(matchStr, inside) : matchStr, alias, matchStr);
				currentNode = addAfter(tokenList, removeFrom, wrapped);

				if (after) {
					addAfter(tokenList, currentNode, after);
				}

				if (removeCount > 1) {
					// at least one Token object was removed, so we have to do some rematching
					// this can only happen if the current pattern is greedy
					matchGrammar(text, tokenList, grammar, currentNode.prev, pos, {
						cause: token + ',' + j,
						reach: reach
					});
				}
			}
		}
	}
}

/**
 * @typedef LinkedListNode
 * @property {T} value
 * @property {LinkedListNode<T> | null} prev The previous node.
 * @property {LinkedListNode<T> | null} next The next node.
 * @template T
 * @private
 */

/**
 * @template T
 * @private
 */
function LinkedList() {
	/** @type {LinkedListNode<T>} */
	var head = { value: null, prev: null, next: null };
	/** @type {LinkedListNode<T>} */
	var tail = { value: null, prev: head, next: null };
	head.next = tail;

	/** @type {LinkedListNode<T>} */
	this.head = head;
	/** @type {LinkedListNode<T>} */
	this.tail = tail;
	this.length = 0;
}

/**
 * Adds a new node with the given value to the list.
 * @param {LinkedList<T>} list
 * @param {LinkedListNode<T>} node
 * @param {T} value
 * @returns {LinkedListNode<T>} The added node.
 * @template T
 */
function addAfter(list, node, value) {
	// assumes that node != list.tail && values.length >= 0
	var next = node.next;

	var newNode = { value: value, prev: node, next: next };
	node.next = newNode;
	next.prev = newNode;
	list.length++;

	return newNode;
}
/**
 * Removes `count` nodes after the given node. The given node will not be removed.
 * @param {LinkedList<T>} list
 * @param {LinkedListNode<T>} node
 * @param {number} count
 * @template T
 */
function removeRange(list, node, count) {
	var next = node.next;
	for (var i = 0; i < count && next !== list.tail; i++) {
		next = next.next;
	}
	node.next = next;
	next.prev = node;
	list.length -= i;
}
/**
 * @param {LinkedList<T>} list
 * @returns {T[]}
 * @template T
 */
function toArray(list) {
	var array = [];
	var node = list.head.next;
	while (node !== list.tail) {
		array.push(node.value);
		node = node.next;
	}
	return array;
}


if (!_self.document) {
	if (!_self.addEventListener) {
		// in Node.js
		return _;
	}

	if (!_.disableWorkerMessageHandler) {
		// In worker
		_self.addEventListener('message', function (evt) {
			var message = JSON.parse(evt.data),
				lang = message.language,
				code = message.code,
				immediateClose = message.immediateClose;

			_self.postMessage(_.highlight(code, _.languages[lang], lang));
			if (immediateClose) {
				_self.close();
			}
		}, false);
	}

	return _;
}

// Get current script and highlight
var script = _.util.currentScript();

if (script) {
	_.filename = script.src;

	if (script.hasAttribute('data-manual')) {
		_.manual = true;
	}
}

function highlightAutomaticallyCallback() {
	if (!_.manual) {
		_.highlightAll();
	}
}

if (!_.manual) {
	// If the document state is "loading", then we'll use DOMContentLoaded.
	// If the document state is "interactive" and the prism.js script is deferred, then we'll also use the
	// DOMContentLoaded event because there might be some plugins or languages which have also been deferred and they
	// might take longer one animation frame to execute which can create a race condition where only some plugins have
	// been loaded when Prism.highlightAll() is executed, depending on how fast resources are loaded.
	// See https://github.com/PrismJS/prism/issues/2102
	var readyState = document.readyState;
	if (readyState === 'loading' || readyState === 'interactive' && script && script.defer) {
		document.addEventListener('DOMContentLoaded', highlightAutomaticallyCallback);
	} else {
		if (window.requestAnimationFrame) {
			window.requestAnimationFrame(highlightAutomaticallyCallback);
		} else {
			window.setTimeout(highlightAutomaticallyCallback, 16);
		}
	}
}

return _;

})(_self);

if ( module.exports) {
	module.exports = Prism;
}

// hack for components to work correctly in node.js
if (typeof commonjsGlobal !== 'undefined') {
	commonjsGlobal.Prism = Prism;
}

// some additional documentation/types

/**
 * The expansion of a simple `RegExp` literal to support additional properties.
 *
 * @typedef GrammarToken
 * @property {RegExp} pattern The regular expression of the token.
 * @property {boolean} [lookbehind=false] If `true`, then the first capturing group of `pattern` will (effectively)
 * behave as a lookbehind group meaning that the captured text will not be part of the matched text of the new token.
 * @property {boolean} [greedy=false] Whether the token is greedy.
 * @property {string|string[]} [alias] An optional alias or list of aliases.
 * @property {Grammar} [inside] The nested grammar of this token.
 *
 * The `inside` grammar will be used to tokenize the text value of each token of this kind.
 *
 * This can be used to make nested and even recursive language definitions.
 *
 * Note: This can cause infinite recursion. Be careful when you embed different languages or even the same language into
 * each another.
 * @global
 * @public
*/

/**
 * @typedef Grammar
 * @type {Object<string, RegExp | GrammarToken | Array<RegExp | GrammarToken>>}
 * @property {Grammar} [rest] An optional grammar object that will be appended to this grammar.
 * @global
 * @public
 */

/**
 * A function which will invoked after an element was successfully highlighted.
 *
 * @callback HighlightCallback
 * @param {Element} element The element successfully highlighted.
 * @returns {void}
 * @global
 * @public
*/

/**
 * @callback HookCallback
 * @param {Object<string, any>} env The environment variables of the hook.
 * @returns {void}
 * @global
 * @public
 */


/* **********************************************
     Begin prism-markup.js
********************************************** */

Prism.languages.markup = {
	'comment': /<!--[\s\S]*?-->/,
	'prolog': /<\?[\s\S]+?\?>/,
	'doctype': {
		// https://www.w3.org/TR/xml/#NT-doctypedecl
		pattern: /<!DOCTYPE(?:[^>"'[\]]|"[^"]*"|'[^']*')+(?:\[(?:[^<"'\]]|"[^"]*"|'[^']*'|<(?!!--)|<!--(?:[^-]|-(?!->))*-->)*\]\s*)?>/i,
		greedy: true,
		inside: {
			'internal-subset': {
				pattern: /(\[)[\s\S]+(?=\]>$)/,
				lookbehind: true,
				greedy: true,
				inside: null // see below
			},
			'string': {
				pattern: /"[^"]*"|'[^']*'/,
				greedy: true
			},
			'punctuation': /^<!|>$|[[\]]/,
			'doctype-tag': /^DOCTYPE/,
			'name': /[^\s<>'"]+/
		}
	},
	'cdata': /<!\[CDATA\[[\s\S]*?]]>/i,
	'tag': {
		pattern: /<\/?(?!\d)[^\s>\/=$<%]+(?:\s(?:\s*[^\s>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+(?=[\s>]))|(?=[\s/>])))+)?\s*\/?>/,
		greedy: true,
		inside: {
			'tag': {
				pattern: /^<\/?[^\s>\/]+/,
				inside: {
					'punctuation': /^<\/?/,
					'namespace': /^[^\s>\/:]+:/
				}
			},
			'attr-value': {
				pattern: /=\s*(?:"[^"]*"|'[^']*'|[^\s'">=]+)/,
				inside: {
					'punctuation': [
						{
							pattern: /^=/,
							alias: 'attr-equals'
						},
						/"|'/
					]
				}
			},
			'punctuation': /\/?>/,
			'attr-name': {
				pattern: /[^\s>\/]+/,
				inside: {
					'namespace': /^[^\s>\/:]+:/
				}
			}

		}
	},
	'entity': [
		{
			pattern: /&[\da-z]{1,8};/i,
			alias: 'named-entity'
		},
		/&#x?[\da-f]{1,8};/i
	]
};

Prism.languages.markup['tag'].inside['attr-value'].inside['entity'] =
	Prism.languages.markup['entity'];
Prism.languages.markup['doctype'].inside['internal-subset'].inside = Prism.languages.markup;

// Plugin to make entity title show the real entity, idea by Roman Komarov
Prism.hooks.add('wrap', function (env) {

	if (env.type === 'entity') {
		env.attributes['title'] = env.content.replace(/&amp;/, '&');
	}
});

Object.defineProperty(Prism.languages.markup.tag, 'addInlined', {
	/**
	 * Adds an inlined language to markup.
	 *
	 * An example of an inlined language is CSS with `<style>` tags.
	 *
	 * @param {string} tagName The name of the tag that contains the inlined language. This name will be treated as
	 * case insensitive.
	 * @param {string} lang The language key.
	 * @example
	 * addInlined('style', 'css');
	 */
	value: function addInlined(tagName, lang) {
		var includedCdataInside = {};
		includedCdataInside['language-' + lang] = {
			pattern: /(^<!\[CDATA\[)[\s\S]+?(?=\]\]>$)/i,
			lookbehind: true,
			inside: Prism.languages[lang]
		};
		includedCdataInside['cdata'] = /^<!\[CDATA\[|\]\]>$/i;

		var inside = {
			'included-cdata': {
				pattern: /<!\[CDATA\[[\s\S]*?\]\]>/i,
				inside: includedCdataInside
			}
		};
		inside['language-' + lang] = {
			pattern: /[\s\S]+/,
			inside: Prism.languages[lang]
		};

		var def = {};
		def[tagName] = {
			pattern: RegExp(/(<__[\s\S]*?>)(?:<!\[CDATA\[(?:[^\]]|\](?!\]>))*\]\]>|(?!<!\[CDATA\[)[\s\S])*?(?=<\/__>)/.source.replace(/__/g, function () { return tagName; }), 'i'),
			lookbehind: true,
			greedy: true,
			inside: inside
		};

		Prism.languages.insertBefore('markup', 'cdata', def);
	}
});

Prism.languages.html = Prism.languages.markup;
Prism.languages.mathml = Prism.languages.markup;
Prism.languages.svg = Prism.languages.markup;

Prism.languages.xml = Prism.languages.extend('markup', {});
Prism.languages.ssml = Prism.languages.xml;
Prism.languages.atom = Prism.languages.xml;
Prism.languages.rss = Prism.languages.xml;


/* **********************************************
     Begin prism-css.js
********************************************** */

(function (Prism) {

	var string = /("|')(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/;

	Prism.languages.css = {
		'comment': /\/\*[\s\S]*?\*\//,
		'atrule': {
			pattern: /@[\w-]+[\s\S]*?(?:;|(?=\s*\{))/,
			inside: {
				'rule': /^@[\w-]+/,
				'selector-function-argument': {
					pattern: /(\bselector\s*\((?!\s*\))\s*)(?:[^()]|\((?:[^()]|\([^()]*\))*\))+?(?=\s*\))/,
					lookbehind: true,
					alias: 'selector'
				},
				'keyword': {
					pattern: /(^|[^\w-])(?:and|not|only|or)(?![\w-])/,
					lookbehind: true
				}
				// See rest below
			}
		},
		'url': {
			// https://drafts.csswg.org/css-values-3/#urls
			pattern: RegExp('\\burl\\((?:' + string.source + '|' + /(?:[^\\\r\n()"']|\\[\s\S])*/.source + ')\\)', 'i'),
			greedy: true,
			inside: {
				'function': /^url/i,
				'punctuation': /^\(|\)$/,
				'string': {
					pattern: RegExp('^' + string.source + '$'),
					alias: 'url'
				}
			}
		},
		'selector': RegExp('[^{}\\s](?:[^{};"\']|' + string.source + ')*?(?=\\s*\\{)'),
		'string': {
			pattern: string,
			greedy: true
		},
		'property': /[-_a-z\xA0-\uFFFF][-\w\xA0-\uFFFF]*(?=\s*:)/i,
		'important': /!important\b/i,
		'function': /[-a-z0-9]+(?=\()/i,
		'punctuation': /[(){};:,]/
	};

	Prism.languages.css['atrule'].inside.rest = Prism.languages.css;

	var markup = Prism.languages.markup;
	if (markup) {
		markup.tag.addInlined('style', 'css');

		Prism.languages.insertBefore('inside', 'attr-value', {
			'style-attr': {
				pattern: /\s*style=("|')(?:\\[\s\S]|(?!\1)[^\\])*\1/i,
				inside: {
					'attr-name': {
						pattern: /^\s*style/i,
						inside: markup.tag.inside
					},
					'punctuation': /^\s*=\s*['"]|['"]\s*$/,
					'attr-value': {
						pattern: /.+/i,
						inside: Prism.languages.css
					}
				},
				alias: 'language-css'
			}
		}, markup.tag);
	}

}(Prism));


/* **********************************************
     Begin prism-clike.js
********************************************** */

Prism.languages.clike = {
	'comment': [
		{
			pattern: /(^|[^\\])\/\*[\s\S]*?(?:\*\/|$)/,
			lookbehind: true
		},
		{
			pattern: /(^|[^\\:])\/\/.*/,
			lookbehind: true,
			greedy: true
		}
	],
	'string': {
		pattern: /(["'])(?:\\(?:\r\n|[\s\S])|(?!\1)[^\\\r\n])*\1/,
		greedy: true
	},
	'class-name': {
		pattern: /(\b(?:class|interface|extends|implements|trait|instanceof|new)\s+|\bcatch\s+\()[\w.\\]+/i,
		lookbehind: true,
		inside: {
			'punctuation': /[.\\]/
		}
	},
	'keyword': /\b(?:if|else|while|do|for|return|in|instanceof|function|new|try|throw|catch|finally|null|break|continue)\b/,
	'boolean': /\b(?:true|false)\b/,
	'function': /\w+(?=\()/,
	'number': /\b0x[\da-f]+\b|(?:\b\d+\.?\d*|\B\.\d+)(?:e[+-]?\d+)?/i,
	'operator': /[<>]=?|[!=]=?=?|--?|\+\+?|&&?|\|\|?|[?*/~^%]/,
	'punctuation': /[{}[\];(),.:]/
};


/* **********************************************
     Begin prism-javascript.js
********************************************** */

Prism.languages.javascript = Prism.languages.extend('clike', {
	'class-name': [
		Prism.languages.clike['class-name'],
		{
			pattern: /(^|[^$\w\xA0-\uFFFF])[_$A-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\.(?:prototype|constructor))/,
			lookbehind: true
		}
	],
	'keyword': [
		{
			pattern: /((?:^|})\s*)(?:catch|finally)\b/,
			lookbehind: true
		},
		{
			pattern: /(^|[^.]|\.\.\.\s*)\b(?:as|async(?=\s*(?:function\b|\(|[$\w\xA0-\uFFFF]|$))|await|break|case|class|const|continue|debugger|default|delete|do|else|enum|export|extends|for|from|function|(?:get|set)(?=\s*[\[$\w\xA0-\uFFFF])|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)\b/,
			lookbehind: true
		},
	],
	'number': /\b(?:(?:0[xX](?:[\dA-Fa-f](?:_[\dA-Fa-f])?)+|0[bB](?:[01](?:_[01])?)+|0[oO](?:[0-7](?:_[0-7])?)+)n?|(?:\d(?:_\d)?)+n|NaN|Infinity)\b|(?:\b(?:\d(?:_\d)?)+\.?(?:\d(?:_\d)?)*|\B\.(?:\d(?:_\d)?)+)(?:[Ee][+-]?(?:\d(?:_\d)?)+)?/,
	// Allow for all non-ASCII characters (See http://stackoverflow.com/a/2008444)
	'function': /#?[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*(?:\.\s*(?:apply|bind|call)\s*)?\()/,
	'operator': /--|\+\+|\*\*=?|=>|&&=?|\|\|=?|[!=]==|<<=?|>>>?=?|[-+*/%&|^!=<>]=?|\.{3}|\?\?=?|\?\.?|[~:]/
});

Prism.languages.javascript['class-name'][0].pattern = /(\b(?:class|interface|extends|implements|instanceof|new)\s+)[\w.\\]+/;

Prism.languages.insertBefore('javascript', 'keyword', {
	'regex': {
		pattern: /((?:^|[^$\w\xA0-\uFFFF."'\])\s]|\b(?:return|yield))\s*)\/(?:\[(?:[^\]\\\r\n]|\\.)*]|\\.|[^/\\\[\r\n])+\/[gimyus]{0,6}(?=(?:\s|\/\*(?:[^*]|\*(?!\/))*\*\/)*(?:$|[\r\n,.;:})\]]|\/\/))/,
		lookbehind: true,
		greedy: true
	},
	// This must be declared before keyword because we use "function" inside the look-forward
	'function-variable': {
		pattern: /#?[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*[=:]\s*(?:async\s*)?(?:\bfunction\b|(?:\((?:[^()]|\([^()]*\))*\)|[_$a-zA-Z\xA0-\uFFFF][$\w\xA0-\uFFFF]*)\s*=>))/,
		alias: 'function'
	},
	'parameter': [
		{
			pattern: /(function(?:\s+[_$A-Za-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*)?\s*\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\))/,
			lookbehind: true,
			inside: Prism.languages.javascript
		},
		{
			pattern: /[_$a-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*(?=\s*=>)/i,
			inside: Prism.languages.javascript
		},
		{
			pattern: /(\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\)\s*=>)/,
			lookbehind: true,
			inside: Prism.languages.javascript
		},
		{
			pattern: /((?:\b|\s|^)(?!(?:as|async|await|break|case|catch|class|const|continue|debugger|default|delete|do|else|enum|export|extends|finally|for|from|function|get|if|implements|import|in|instanceof|interface|let|new|null|of|package|private|protected|public|return|set|static|super|switch|this|throw|try|typeof|undefined|var|void|while|with|yield)(?![$\w\xA0-\uFFFF]))(?:[_$A-Za-z\xA0-\uFFFF][$\w\xA0-\uFFFF]*\s*)\(\s*|\]\s*\(\s*)(?!\s)(?:[^()]|\([^()]*\))+?(?=\s*\)\s*\{)/,
			lookbehind: true,
			inside: Prism.languages.javascript
		}
	],
	'constant': /\b[A-Z](?:[A-Z_]|\dx?)*\b/
});

Prism.languages.insertBefore('javascript', 'string', {
	'template-string': {
		pattern: /`(?:\\[\s\S]|\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}|(?!\${)[^\\`])*`/,
		greedy: true,
		inside: {
			'template-punctuation': {
				pattern: /^`|`$/,
				alias: 'string'
			},
			'interpolation': {
				pattern: /((?:^|[^\\])(?:\\{2})*)\${(?:[^{}]|{(?:[^{}]|{[^}]*})*})+}/,
				lookbehind: true,
				inside: {
					'interpolation-punctuation': {
						pattern: /^\${|}$/,
						alias: 'punctuation'
					},
					rest: Prism.languages.javascript
				}
			},
			'string': /[\s\S]+/
		}
	}
});

if (Prism.languages.markup) {
	Prism.languages.markup.tag.addInlined('script', 'javascript');
}

Prism.languages.js = Prism.languages.javascript;


/* **********************************************
     Begin prism-file-highlight.js
********************************************** */

(function () {
	if (typeof self === 'undefined' || !self.Prism || !self.document) {
		return;
	}

	var Prism = window.Prism;

	var LOADING_MESSAGE = 'Loading…';
	var FAILURE_MESSAGE = function (status, message) {
		return '✖ Error ' + status + ' while fetching file: ' + message;
	};
	var FAILURE_EMPTY_MESSAGE = '✖ Error: File does not exist or is empty';

	var EXTENSIONS = {
		'js': 'javascript',
		'py': 'python',
		'rb': 'ruby',
		'ps1': 'powershell',
		'psm1': 'powershell',
		'sh': 'bash',
		'bat': 'batch',
		'h': 'c',
		'tex': 'latex'
	};

	var STATUS_ATTR = 'data-src-status';
	var STATUS_LOADING = 'loading';
	var STATUS_LOADED = 'loaded';
	var STATUS_FAILED = 'failed';

	var SELECTOR = 'pre[data-src]:not([' + STATUS_ATTR + '="' + STATUS_LOADED + '"])'
		+ ':not([' + STATUS_ATTR + '="' + STATUS_LOADING + '"])';

	var lang = /\blang(?:uage)?-([\w-]+)\b/i;

	/**
	 * Sets the Prism `language-xxxx` or `lang-xxxx` class to the given language.
	 *
	 * @param {HTMLElement} element
	 * @param {string} language
	 * @returns {void}
	 */
	function setLanguageClass(element, language) {
		var className = element.className;
		className = className.replace(lang, ' ') + ' language-' + language;
		element.className = className.replace(/\s+/g, ' ').trim();
	}


	Prism.hooks.add('before-highlightall', function (env) {
		env.selector += ', ' + SELECTOR;
	});

	Prism.hooks.add('before-sanity-check', function (env) {
		var pre = /** @type {HTMLPreElement} */ (env.element);
		if (pre.matches(SELECTOR)) {
			env.code = ''; // fast-path the whole thing and go to complete

			pre.setAttribute(STATUS_ATTR, STATUS_LOADING); // mark as loading

			// add code element with loading message
			var code = pre.appendChild(document.createElement('CODE'));
			code.textContent = LOADING_MESSAGE;

			var src = pre.getAttribute('data-src');

			var language = env.language;
			if (language === 'none') {
				// the language might be 'none' because there is no language set;
				// in this case, we want to use the extension as the language
				var extension = (/\.(\w+)$/.exec(src) || [, 'none'])[1];
				language = EXTENSIONS[extension] || extension;
			}

			// set language classes
			setLanguageClass(code, language);
			setLanguageClass(pre, language);

			// preload the language
			var autoloader = Prism.plugins.autoloader;
			if (autoloader) {
				autoloader.loadLanguages(language);
			}

			// load file
			var xhr = new XMLHttpRequest();
			xhr.open('GET', src, true);
			xhr.onreadystatechange = function () {
				if (xhr.readyState == 4) {
					if (xhr.status < 400 && xhr.responseText) {
						// mark as loaded
						pre.setAttribute(STATUS_ATTR, STATUS_LOADED);

						// highlight code
						code.textContent = xhr.responseText;
						Prism.highlightElement(code);

					} else {
						// mark as failed
						pre.setAttribute(STATUS_ATTR, STATUS_FAILED);

						if (xhr.status >= 400) {
							code.textContent = FAILURE_MESSAGE(xhr.status, xhr.statusText);
						} else {
							code.textContent = FAILURE_EMPTY_MESSAGE;
						}
					}
				}
			};
			xhr.send(null);
		}
	});

	Prism.plugins.fileHighlight = {
		/**
		 * Executes the File Highlight plugin for all matching `pre` elements under the given container.
		 *
		 * Note: Elements which are already loaded or currently loading will not be touched by this method.
		 *
		 * @param {ParentNode} [container=document]
		 */
		highlight: function highlight(container) {
			var elements = (container || document).querySelectorAll(SELECTOR);

			for (var i = 0, element; element = elements[i++];) {
				Prism.highlightElement(element);
			}
		}
	};

	var logged = false;
	/** @deprecated Use `Prism.plugins.fileHighlight.highlight` instead. */
	Prism.fileHighlight = function () {
		if (!logged) {
			console.warn('Prism.fileHighlight is deprecated. Use `Prism.plugins.fileHighlight.highlight` instead.');
			logged = true;
		}
		Prism.plugins.fileHighlight.highlight.apply(this, arguments);
	};

})();
});

function debounce(func, timeout) {
    let timer;
    return (...args) => {
        const next = () => func(...args);
        if (timer) {
            clearTimeout(timer);
        }
        timer = setTimeout(next, timeout && timeout > 0 ? timeout : 300);
    };
}

function injectCSS(id, src) {
    return new Promise((resolve, reject) => {
        if (!document) {
            resolve();
            return;
        }
        if (document.getElementById(id)) {
            resolve('CSS already loaded.');
            return;
        }
        const link = document.createElement('link');
        link.id = id;
        link.setAttribute('rel', 'stylesheet');
        link.setAttribute('href', src);
        link.addEventListener('load', () => resolve('CSS loaded.'));
        link.addEventListener('error', () => reject('Error loading css.'));
        link.addEventListener('abort', () => reject('CSS loading aborted.'));
        document.head.appendChild(link);
    });
}

function loadTheme(theme) {
  if (!theme || theme === undefined) {
    return undefined;
  }
  switch (theme) {
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['3024-NIGHT']:
      return __webpack_require__.e(/*! import() */ 1).then(__webpack_require__.bind(null, /*! ./3024-night-8138cc8f.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/3024-night-8138cc8f.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['A11Y-DARK']:
      return __webpack_require__.e(/*! import() */ 2).then(__webpack_require__.bind(null, /*! ./a11y-dark-533291b3.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/a11y-dark-533291b3.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['BASE16-DARK']:
      return __webpack_require__.e(/*! import() */ 3).then(__webpack_require__.bind(null, /*! ./base16-dark-12f375e0.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/base16-dark-12f375e0.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['BASE16-LIGHT']:
      return __webpack_require__.e(/*! import() */ 4).then(__webpack_require__.bind(null, /*! ./base16-light-52469b64.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/base16-light-52469b64.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].BLACKBOARD:
      return __webpack_require__.e(/*! import() */ 5).then(__webpack_require__.bind(null, /*! ./blackboard-db1ca116.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/blackboard-db1ca116.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].COBALT:
      return __webpack_require__.e(/*! import() */ 6).then(__webpack_require__.bind(null, /*! ./cobalt-5d681c9a.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/cobalt-5d681c9a.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].DUOTONE:
      return __webpack_require__.e(/*! import() */ 8).then(__webpack_require__.bind(null, /*! ./duotone-acd564a2.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/duotone-acd564a2.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].HOPSCOTCH:
      return __webpack_require__.e(/*! import() */ 9).then(__webpack_require__.bind(null, /*! ./hopscotch-4820dd1a.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/hopscotch-4820dd1a.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].LUCARIO:
      return __webpack_require__.e(/*! import() */ 10).then(__webpack_require__.bind(null, /*! ./lucario-59b2ad39.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/lucario-59b2ad39.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].MATERIAL:
      return __webpack_require__.e(/*! import() */ 11).then(__webpack_require__.bind(null, /*! ./material-8e59c55f.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/material-8e59c55f.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].MONOKAI:
      return __webpack_require__.e(/*! import() */ 12).then(__webpack_require__.bind(null, /*! ./monokai-c3bd40c3.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/monokai-c3bd40c3.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['NIGHT-OWL']:
      return __webpack_require__.e(/*! import() */ 13).then(__webpack_require__.bind(null, /*! ./night-owl-36dc5565.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/night-owl-36dc5565.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].NORD:
      return __webpack_require__.e(/*! import() */ 14).then(__webpack_require__.bind(null, /*! ./nord-ce9f3484.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/nord-ce9f3484.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['OCEANIC-NEXT']:
      return __webpack_require__.e(/*! import() */ 15).then(__webpack_require__.bind(null, /*! ./oceanic-next-e3f20f17.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/oceanic-next-e3f20f17.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['ONE-DARK']:
      return __webpack_require__.e(/*! import() */ 16).then(__webpack_require__.bind(null, /*! ./one-dark-3da1a5b7.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/one-dark-3da1a5b7.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['ONE-LIGHT']:
      return __webpack_require__.e(/*! import() */ 17).then(__webpack_require__.bind(null, /*! ./one-light-81b6cd67.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/one-light-81b6cd67.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].PANDA:
      return __webpack_require__.e(/*! import() */ 18).then(__webpack_require__.bind(null, /*! ./panda-2e75c11f.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/panda-2e75c11f.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].PARAISO:
      return __webpack_require__.e(/*! import() */ 19).then(__webpack_require__.bind(null, /*! ./paraiso-cc30311f.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/paraiso-cc30311f.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].SETI:
      return __webpack_require__.e(/*! import() */ 20).then(__webpack_require__.bind(null, /*! ./seti-977c4512.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/seti-977c4512.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['SHADES-OF-PURPLE']:
      return __webpack_require__.e(/*! import() */ 21).then(__webpack_require__.bind(null, /*! ./shades-of-purple-649a9b24.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/shades-of-purple-649a9b24.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['SOLARIZED-DARK']:
      return __webpack_require__.e(/*! import() */ 22).then(__webpack_require__.bind(null, /*! ./solarized-dark-d1ab5e8b.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/solarized-dark-d1ab5e8b.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"]['SOLARIZED-LIGHT']:
      return __webpack_require__.e(/*! import() */ 23).then(__webpack_require__.bind(null, /*! ./solarized-light-429511f1.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/solarized-light-429511f1.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].SYNTHWAVE:
      return __webpack_require__.e(/*! import() */ 24).then(__webpack_require__.bind(null, /*! ./synthwave-af3bb869.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/synthwave-af3bb869.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].TWILIGHT:
      return __webpack_require__.e(/*! import() */ 25).then(__webpack_require__.bind(null, /*! ./twilight-68d946d8.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/twilight-68d946d8.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].VERMINAL:
      return __webpack_require__.e(/*! import() */ 26).then(__webpack_require__.bind(null, /*! ./verminal-f2bc4235.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/verminal-f2bc4235.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].VSCODE:
      return __webpack_require__.e(/*! import() */ 27).then(__webpack_require__.bind(null, /*! ./vscode-1e0057d8.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/vscode-1e0057d8.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].YETI:
      return __webpack_require__.e(/*! import() */ 28).then(__webpack_require__.bind(null, /*! ./yeti-67a4664e.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/yeti-67a4664e.js"));
    case _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].ZENBURN:
      return __webpack_require__.e(/*! import() */ 29).then(__webpack_require__.bind(null, /*! ./zenburn-a2e7f052.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/zenburn-a2e7f052.js"));
    default:
      return __webpack_require__.e(/*! import() */ 7).then(__webpack_require__.bind(null, /*! ./dracula-0eb4b4d6.js */ "./node_modules/@deckdeckgo/highlight-code/dist/esm/dracula-0eb4b4d6.js"));
  }
}

const CarbonThemeStyle = ({ style }) => {
  return (Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("style", null, `
      :host ${style};
    `));
};

const deckdeckgoHighlightCodeCss = "code[class*=\"language-\"],pre[class*=\"language-\"]{color:black;background:none;text-shadow:0 1px white;font-family:Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;font-size:1em;text-align:left;white-space:pre;word-spacing:normal;word-break:normal;word-wrap:normal;line-height:1.5;-moz-tab-size:4;-o-tab-size:4;tab-size:4;-webkit-hyphens:none;-moz-hyphens:none;-ms-hyphens:none;hyphens:none}pre[class*=\"language-\"]::-moz-selection,pre[class*=\"language-\"] ::-moz-selection,code[class*=\"language-\"]::-moz-selection,code[class*=\"language-\"] ::-moz-selection{text-shadow:none;background:#b3d4fc}pre[class*=\"language-\"]::selection,pre[class*=\"language-\"] ::selection,code[class*=\"language-\"]::selection,code[class*=\"language-\"] ::selection{text-shadow:none;background:#b3d4fc}@media print{code[class*=\"language-\"],pre[class*=\"language-\"]{text-shadow:none}}pre[class*=\"language-\"]{padding:1em;margin:.5em 0;overflow:auto}:not(pre)>code[class*=\"language-\"],pre[class*=\"language-\"]{background:#f5f2f0}:not(pre)>code[class*=\"language-\"]{padding:.1em;border-radius:.3em;white-space:normal}.token.comment,.token.prolog,.token.doctype,.token.cdata{color:slategray}.token.punctuation{color:#999}.token.namespace{opacity:.7}.token.property,.token.tag,.token.boolean,.token.number,.token.constant,.token.symbol,.token.deleted{color:#905}.token.selector,.token.attr-name,.token.string,.token.char,.token.builtin,.token.inserted{color:#690}.token.operator,.token.entity,.token.url,.language-css .token.string,.style .token.string{color:#9a6e3a;background:hsla(0, 0%, 100%, .5)}.token.atrule,.token.attr-value,.token.keyword{color:#07a}.token.function,.token.class-name{color:#DD4A68}.token.regex,.token.important,.token.variable{color:#e90}.token.important,.token.bold{font-weight:bold}.token.italic{font-style:italic}.token.entity{cursor:help}:host ::slotted([slot=code]){display:none}:host([editable]){cursor:var(--deckgo-editable-cursor, text)}:host([editable]) code:empty:not(:focus):after{content:var(--deckgo-highlight-code-empty-text, \"Click to add your code\")}:host(.deckgo-highlight-code-carbon){display:var(--deckgo-highlight-code-carbon-display, block);overflow:var(--deckgo-highlight-code-carbon-overflow, auto);border:var(--deckgo-highlight-code-carbon-border);border-radius:var(--deckgo-highlight-code-carbon-border-radius, 4px);background:var(--deckgo-highlight-code-carbon-background, #282a36);color:var(--deckgo-highlight-code-carbon-color, white);box-shadow:var(--deckgo-highlight-code-carbon-box-shadow, rgba(0, 0, 0, 0.55) 0 8px 16px);margin:var(--deckgo-highlight-code-carbon-margin, 16px 0)}:host(.deckgo-highlight-code-carbon) div.deckgo-highlight-code-container{margin:var(--deckgo-highlight-code-margin, 0 0 16px)}:host(.deckgo-highlight-code-carbon) div.deckgo-highlight-code-container code>div.deckgo-highlight-code-line-number:before{background:var(--deckgo-highlight-code-carbon-background, transparent);border-color:var(--deckgo-highlight-code-line-numbers, var(--deckgo-highlight-code-token-comment, #6272a4));color:var(--deckgo-highlight-code-line-numbers, var(--deckgo-highlight-code-token-comment, #6272a4))}:host(.deckgo-highlight-code-carbon) ::slotted([slot=code]){color:var(--deckgo-highlight-code-carbon-color, white)}:host(.deckgo-highlight-code-ubuntu){display:var(--deckgo-highlight-code-ubuntu-display, block);overflow:var(--deckgo-highlight-code-ubuntu-overflow, auto);border:var(--deckgo-highlight-code-ubuntu-border);border-radius:var(--deckgo-highlight-code-ubuntu-border-radius, 6px 6px 0 0);background:var(--deckgo-highlight-code-ubuntu-background, #4c1e3d);color:var(--deckgo-highlight-code-ubuntu-color, #ddd);box-shadow:var(--deckgo-highlight-code-ubuntu-box-shadow, 2px 4px 10px rgba(0, 0, 0, 0.5));margin:var(--deckgo-highlight-code-ubuntu-margin, 16px 0)}:host(.deckgo-highlight-code-ubuntu) div.deckgo-highlight-code-container{margin:var(--deckgo-highlight-code-margin, 0 0 16px);padding:var(--deckgo-highlight-code-padding, 2px 0 0);background:transparent}:host(.deckgo-highlight-code-ubuntu) div.deckgo-highlight-code-container code{font-family:var(--deckgo-highlight-code-font-family, \"Ubuntu mono\")}:host(.deckgo-highlight-code-ubuntu) div.deckgo-highlight-code-container code>div.deckgo-highlight-code-line-number:before{background:var(--deckgo-highlight-code-ubuntu-background, #4c1e3d)}:host(.deckgo-highlight-code-ubuntu) ::slotted([slot=code]){color:var(--deckgo-highlight-code-ubuntu-color, #ddd)}div.deckgo-highlight-code-container{color:var(--deckgo-highlight-code-color, inherit);background:var(--deckgo-highlight-code-background);padding:var(--deckgo-highlight-code-padding, 0 16px);border-radius:var(--deckgo-highlight-code-border-radius);margin:var(--deckgo-highlight-code-margin, 16px 0);transform-origin:bottom left;transition:all 0.2s ease-in-out;transform:scale(var(--deckgo-highlight-code-zoom, 1));direction:var(--deckgo-highlight-code-direction, ltr);text-align:var(--deckgo-highlight-code-text-align, start);width:var(--deckgo-highlight-code-container-width);height:var(--deckgo-highlight-code-container-height);display:var(--deckgo-highlight-code-container-display, block);justify-content:var(--deckgo-highlight-code-container-justify-content);flex-direction:var(--deckgo-highlight-code-container-flex-direction);align-items:var(--deckgo-highlight-code-container-align-items)}div.deckgo-highlight-code-container code{overflow-y:var(--deckgo-highlight-code-scroll, auto);white-space:var(--deckgo-highlight-code-white-space, pre-wrap);font-size:var(--deckgo-highlight-code-font-size);font-family:var(--deckgo-highlight-code-font-family, monospace);line-height:var(--deckgo-highlight-code-line-height);display:var(--deckgo-highlight-code-display, block);counter-reset:linenumber;height:100%;}div.deckgo-highlight-code-container code>div.deckgo-highlight-code-line-number{counter-increment:linenumber;position:relative;padding-left:3.5em}div.deckgo-highlight-code-container code>div.deckgo-highlight-code-line-number:before{content:counter(linenumber);display:inline-block;position:absolute;top:0;bottom:0;left:0;width:2.5em;background:var(--deckgo-highlight-code-background);border-right:1px solid var(--deckgo-highlight-code-line-numbers, var(--deckgo-highlight-code-token-comment, #6272a4));color:var(--deckgo-highlight-code-line-numbers, var(--deckgo-highlight-code-token-comment, #6272a4))}div.deckgo-highlight-code-container code span.deckgo-highlight-code-anchor-hidden{visibility:hidden}div.deckgo-highlight-code-container code .deckgo-highlight-code-line{background:var(--deckgo-highlight-code-line-background, #3e4564);border-top:var(--deckgo-highlight-code-line-border-top);border-bottom:var(--deckgo-highlight-code-line-border-bottom)}div.deckgo-highlight-code-container code .language-css .token.string:not(.deckgo-highlight-code-line),div.deckgo-highlight-code-container code .style .token.string:not(.deckgo-highlight-code-line),div.deckgo-highlight-code-container code .token.entity:not(.deckgo-highlight-code-line),div.deckgo-highlight-code-container code .token.operator:not(.deckgo-highlight-code-line),div.deckgo-highlight-code-container code .token.url:not(.deckgo-highlight-code-line){background:inherit}div.deckgo-highlight-code-container code .token.comment,div.deckgo-highlight-code-container code .token.prolog,div.deckgo-highlight-code-container code .token.doctype,div.deckgo-highlight-code-container code .token.cdata{color:var(--deckgo-highlight-code-token-comment, #6272a4)}div.deckgo-highlight-code-container code .token.punctuation{color:var(--deckgo-highlight-code-token-punctuation, #6272a4)}div.deckgo-highlight-code-container code .token.property,div.deckgo-highlight-code-container code .token.tag,div.deckgo-highlight-code-container code .token.boolean,div.deckgo-highlight-code-container code .token.number,div.deckgo-highlight-code-container code .token.constant,div.deckgo-highlight-code-container code .token.symbol,div.deckgo-highlight-code-container code .token.deleted{color:var(--deckgo-highlight-code-token-property, #bd93f9)}div.deckgo-highlight-code-container code .token.selector,div.deckgo-highlight-code-container code .token.attr-name,div.deckgo-highlight-code-container code .token.string,div.deckgo-highlight-code-container code .token.char,div.deckgo-highlight-code-container code .token.builtin,div.deckgo-highlight-code-container code .token.inserted{color:var(--deckgo-highlight-code-token-selector, #50fa7b)}div.deckgo-highlight-code-container code .token.operator,div.deckgo-highlight-code-container code .token.entity,div.deckgo-highlight-code-container code .token.url,div.deckgo-highlight-code-container code .language-css .token.string,div.deckgo-highlight-code-container code .style .token.string{color:var(--deckgo-highlight-code-token-operator, #ff79c6)}div.deckgo-highlight-code-container code .token.atrule,div.deckgo-highlight-code-container code .token.attr-value,div.deckgo-highlight-code-container code .token.keyword{color:var(--deckgo-highlight-code-token-atrule, #ff79c6)}div.deckgo-highlight-code-container code .token.function,div.deckgo-highlight-code-container code .token.class-name{color:var(--deckgo-highlight-code-token-function, #ffb86c)}div.deckgo-highlight-code-container code .token.regex,div.deckgo-highlight-code-container code .token.important,div.deckgo-highlight-code-container code .token.variable{color:var(--deckgo-highlight-code-token-regex, #f1fa8c)}div.carbon{display:flex;justify-content:flex-start;padding:var(--deckgo-highlight-code-carbon-header-padding, 8px 16px);margin:var(--deckgo-highlight-code-carbon-header-margin, 0)}div.carbon>div{display:var(--deckgo-highlight-code-carbon-toolbar-display, \"block\");width:var(--deckgo-highlight-code-carbon-header-button-width, 12px);height:var(--deckgo-highlight-code-carbon-header-button-height, 12px);border-radius:var(--deckgo-highlight-code-carbon-header-button-border-radius, 50%);margin:var(--deckgo-highlight-code-carbon-header-button-margin, 8px 6px 8px 0)}div.carbon>div.red{background:var(--deckgo-highlight-code-carbon-header-button-red-background, #ff5f56);border:var(--deckgo-highlight-code-carbon-header-button-red-border, 0.5px solid #e0443e)}div.carbon>div.yellow{background:var(--deckgo-highlight-code-carbon-header-button-yellow-background, #ffbd2e);border:var(--deckgo-highlight-code-carbon-header-button-yellow-border, 0.5px solid #dea123)}div.carbon>div.green{background:var(--deckgo-highlight-code-carbon-header-button-green-background, #27c93f);border:var(--deckgo-highlight-code-carbon-header-button-green-border, 0.5px solid #1aab29)}div.ubuntu{display:flex;justify-content:flex-start;align-items:center;padding:var(--deckgo-highlight-code-ubuntu-header-padding, 0 8px);height:var(--deckgo-highlight-code-ubuntu-header-height, 25px);background:var(--deckgo-highlight-code-ubuntu-header-background, linear-gradient(#504b45 0%, #3c3b37 100%));font-family:var(--deckgo-highlight-code-ubuntu-header-font-family, \"Ubuntu\")}div.ubuntu>div{display:flex;align-items:center;justify-content:center;width:var(--deckgo-highlight-code-ubuntu-header-button-width, 12px);height:var(--deckgo-highlight-code-ubuntu-header-button-height, 12px);border-radius:var(--deckgo-highlight-code-ubuntu-header-button-border-radius, 50%);margin:var(--deckgo-highlight-code-ubuntu-header-button-margin, 0 4px 0 0);font-size:var(--deckgo-highlight-code-ubuntu-header-button-font-size, 7px);color:var(--deckgo-highlight-code-ubuntu-header-button-color, black);text-shadow:var(--deckgo-highlight-code-ubuntu-header-button-text-shadow, 0px 1px 0px rgba(255, 255, 255, 0.2));box-shadow:var(--deckgo-highlight-code-ubuntu-header-button-box-shadow, 0px 0px 1px 0px #41403a, 0px 1px 1px 0px #474642)}div.ubuntu>div.close{background:var(--deckgo-highlight-code-ubuntu-header-button-close-background, linear-gradient(#f37458 0%, #de4c12 100%));border:var(--deckgo-highlight-code-ubuntu-header-button-close-border)}div.ubuntu>div.minimize{background:var(--deckgo-highlight-code-ubuntu-header-button-minimize-background, linear-gradient(#7d7871 0%, #595953 100%));border:var(--deckgo-highlight-code-ubuntu-header-button-minimize-border)}div.ubuntu>div.maximize{background:var(--deckgo-highlight-code-ubuntu-header-button-maximize-background, linear-gradient(#7d7871 0%, #595953 100%));border:var(--deckgo-highlight-code-ubuntu-header-button-maximize-border)}div.ubuntu>div.close span,div.ubuntu>div.minimize span,div.ubuntu>div.maximize span{display:var(--deckgo-highlight-code-ubuntu-header-button-span-display, inherit)}div.ubuntu>p{color:var(--deckgo-highlight-code-ubuntu-header-user-color, #d5d0ce);font-size:var(--deckgo-highlight-code-ubuntu-header-user-font-size, 12px);line-height:var(--deckgo-highlight-code-ubuntu-header-user-line-height, 14px);margin:var(--deckgo-highlight-code-ubuntu-header-user-margin, 0 0 1px 4px)}";

const DeckdeckgoHighlightCode = class {
  constructor(hostRef) {
    Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["r"])(this, hostRef);
    this.prismLanguageLoaded = Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["c"])(this, "prismLanguageLoaded", 7);
    this.codeDidChange = Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["c"])(this, "codeDidChange", 7);
    this.anchor = '// DeckDeckGo';
    this.anchorZoom = '// DeckDeckGoZoom';
    this.hideAnchor = true;
    this.language = 'javascript';
    this.lineNumbers = false;
    this.terminal = _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].CARBON;
    this.editable = false;
    this.theme = _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["a"].DRACULA;
    this.anchorOffsetTop = 0;
    this.fetchOrParseAfterUpdate = false;
    this.loaded = false;
    this.catchTab = async ($event) => {
      if ($event && $event.key === 'Tab') {
        $event.preventDefault();
        document.execCommand('insertHTML', false, '&#009');
      }
    };
    this.debounceUpdateSlot = debounce(async () => {
      await this.copyCodeToSlot();
    }, 500);
  }
  async componentWillLoad() {
    await this.loadGoogleFonts();
    await this.loadTheme();
  }
  async componentDidLoad() {
    const languageWasLoaded = await this.languageDidLoad();
    await this.loadLanguages();
    if (languageWasLoaded) {
      await this.fetchOrParse();
    }
  }
  async componentDidUpdate() {
    if (this.fetchOrParseAfterUpdate) {
      await this.fetchOrParse();
      this.fetchOrParseAfterUpdate = false;
    }
  }
  async loadTheme() {
    if (this.terminal !== _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].CARBON || !this.theme) {
      this.themeStyle = undefined;
      return;
    }
    const { theme } = await loadTheme(this.theme);
    this.themeStyle = theme;
  }
  async languageLoaded($event) {
    if (!$event || !$event.detail) {
      return;
    }
    if (this.languagesToLoad) {
      this.languagesToLoad = this.languagesToLoad.filter((lang) => lang !== $event.detail);
    }
    if (this.language && !this.loaded && (this.languagesToLoad === undefined || this.languagesToLoad.length <= 0)) {
      await this.fetchOrParse();
      this.loaded = true;
    }
  }
  async fetchOrParse() {
    if (!this.language || !_deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["d"][this.language]) {
      return;
    }
    if (this.src) {
      await this.fetchCode();
    }
    else {
      await this.parseSlottedCode();
    }
  }
  languageDidLoad() {
    return new Promise((resolve) => {
      if (!document || !this.language || this.language === '') {
        resolve(false);
        return;
      }
      const scripts = document.querySelector("[deckdeckgo-prism-loaded='" + this.language + "']");
      if (scripts) {
        resolve(true);
      }
      else {
        resolve(false);
      }
    });
  }
  async onLanguage() {
    await this.loadLanguages(true);
  }
  async loadLanguages(reload = false) {
    this.loaded = false;
    if (!this.language || !_deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["d"][this.language]) {
      console.error(`Language ${this.language} is not supported`);
      return;
    }
    await this.initLanguagesToLoad();
    await this.loadLanguagesRequire();
    await this.loadScript(this.language, reload);
  }
  async initLanguagesToLoad() {
    if (!this.language) {
      return;
    }
    const definition = _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["d"][this.language];
    this.languagesToLoad = definition.require && definition.require.length > 0 ? [this.language, ...definition.require] : [this.language];
  }
  async loadLanguagesRequire() {
    const promises = [];
    const definition = _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["d"][this.language];
    if (definition.require) {
      promises.push(...definition.require.map((extraScript) => this.loadScript(extraScript, false, true)));
    }
    if (promises.length <= 0) {
      return;
    }
    await Promise.all(promises);
  }
  loadScript(lang, reload = false, requireScript = false) {
    return new Promise(async (resolve) => {
      if (!document || !lang || lang === '') {
        resolve();
        return;
      }
      // No need to load javascript, it is there
      if (lang === 'javascript') {
        this.prismLanguageLoaded.emit('javascript');
        resolve();
        return;
      }
      const scripts = document.querySelector("[deckdeckgo-prism='" + lang + "']");
      if (scripts) {
        if (reload) {
          this.prismLanguageLoaded.emit(lang);
        }
        resolve();
        return;
      }
      const script = document.createElement('script');
      script.onload = async () => {
        script.setAttribute('deckdeckgo-prism-loaded', lang);
        this.prismLanguageLoaded.emit(lang);
      };
      script.onerror = async () => {
        if (script.parentElement) {
          script.parentElement.removeChild(script);
        }
        // if the language definition doesn't exist or if unpkg is down, display code anyway
        this.prismLanguageLoaded.emit(lang);
      };
      const definition = _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["d"][this.language];
      let language = !requireScript && definition.main ? definition.main : lang;
      script.src = 'https://unpkg.com/prismjs@latest/components/prism-' + language + '.js';
      script.setAttribute('deckdeckgo-prism', language);
      script.defer = true;
      document.head.appendChild(script);
      script.addEventListener('load', () => resolve(), { once: true });
    });
  }
  async onLineNumbersChange() {
    await this.fetchOrParse();
  }
  async onCarbonChange() {
    this.fetchOrParseAfterUpdate = true;
    await this.loadGoogleFonts();
  }
  async loadGoogleFonts() {
    if (this.terminal === _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].UBUNTU) {
      await injectCSS('google-fonts-ubuntu', 'https://fonts.googleapis.com/css?family=Ubuntu|Ubuntu+Mono&display=swap');
    }
  }
  load() {
    return new Promise(async (resolve) => {
      if (!this.language || this.language === '') {
        resolve();
        return;
      }
      if (this.language === 'javascript') {
        await this.fetchOrParse();
        resolve();
        return;
      }
      if (document.querySelector("[deckdeckgo-prism-loaded='" + this.language + "']")) {
        await this.fetchOrParse();
      }
      else {
        await this.loadLanguages();
      }
      resolve();
    });
  }
  parseSlottedCode() {
    var _a;
    const code = this.el.querySelector("[slot='code']");
    if (code) {
      return this.parseCode((_a = code === null || code === void 0 ? void 0 : code.innerHTML) === null || _a === void 0 ? void 0 : _a.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'));
    }
    else {
      return new Promise((resolve) => {
        resolve();
      });
    }
  }
  async fetchCode() {
    if (!this.src) {
      return;
    }
    let fetchedCode;
    try {
      const response = await fetch(this.src);
      fetchedCode = await response.text();
      await this.parseCode(fetchedCode);
    }
    catch (e) {
      // Prism might not be able to parse the code for the selected language
      const container = this.el.shadowRoot.querySelector('div.deckgo-highlight-code-container');
      if (container && fetchedCode) {
        container.children[0].innerHTML = fetchedCode;
      }
    }
  }
  parseCode(code) {
    return new Promise(async (resolve, reject) => {
      const container = this.el.shadowRoot.querySelector('div.deckgo-highlight-code-container');
      if (!code || code === undefined || code === '') {
        resolve();
        return;
      }
      if (container) {
        try {
          // clear the container first
          container.children[0].innerHTML = '';
          // split the code on linebreaks
          const regEx = RegExp(/\n(?!$)/g); //
          const match = code.split(regEx);
          match.forEach((m, idx, array) => {
            // On last element
            if (idx === array.length - 1) {
              this.attachHighlightObserver(container);
            }
            let div = document.createElement('div');
            if (this.lineNumbers) {
              div.classList.add('deckgo-highlight-code-line-number');
            }
            const highlight = prism.highlight(m, prism.languages[this.language], this.language);
            // If empty, use \u200B as zero width text spacer
            div.innerHTML = highlight && highlight !== '' ? highlight : '\u200B';
            container.children[0].appendChild(div);
          });
          await this.addAnchors();
          resolve();
        }
        catch (err) {
          reject(err);
        }
      }
    });
  }
  attachHighlightObserver(container) {
    if (window && 'ResizeObserver' in window) {
      // @ts-ignore
      const observer = new ResizeObserver(async (_entries) => {
        await this.addHighlight();
        observer.disconnect();
      });
      observer.observe(container);
    }
    else {
      // Back in my days...
      setTimeout(async () => {
        await this.addHighlight();
      }, 100);
    }
  }
  addAnchors() {
    return new Promise((resolve) => {
      const elements = this.el.shadowRoot.querySelectorAll('span.comment');
      if (elements) {
        const elementsArray = Array.from(elements);
        const anchors = elementsArray.filter((element) => {
          return this.hasLineAnchor(element.innerHTML);
        });
        if (anchors) {
          anchors.forEach((anchor) => {
            anchor.classList.add('deckgo-highlight-code-anchor');
            if (this.hideAnchor) {
              anchor.classList.add('deckgo-highlight-code-anchor-hidden');
            }
          });
        }
      }
      resolve();
    });
  }
  hasLineAnchor(line) {
    return line && this.anchor && line.indexOf('@Prop') === -1 && line.split(' ').join('').indexOf(this.anchor.split(' ').join('')) > -1;
  }
  addHighlight() {
    return new Promise(async (resolve) => {
      if (this.highlightLines && this.highlightLines.length > 0) {
        const rows = await this.findRowsToHighlight();
        if (rows && rows.length > 0) {
          const containerCode = this.el.shadowRoot.querySelector('code');
          if (containerCode && containerCode.hasChildNodes()) {
            const elements = Array.prototype.slice.call(containerCode.childNodes);
            let rowIndex = 0;
            let lastOffsetTop = -1;
            let offsetHeight = -1;
            elements.forEach((element) => {
              let editElement;
              // We need to convert text entries to an element in order to be able to style it
              if (element.nodeName === '#text') {
                const span = document.createElement('span');
                if (element.previousElementSibling) {
                  element.previousElementSibling.insertAdjacentElement('afterend', span);
                }
                else {
                  element.parentNode.prepend(span);
                }
                span.appendChild(element);
                editElement = span;
              }
              else {
                editElement = element;
              }
              // We try to find the row index with the offset of the element
              rowIndex = editElement.offsetTop > lastOffsetTop ? rowIndex + 1 : rowIndex;
              lastOffsetTop = editElement.offsetTop;
              // For some reason, some converted text element are displayed on two lines, that's why we should consider the 2nd one as index
              offsetHeight = offsetHeight === -1 || offsetHeight > editElement.offsetHeight ? editElement.offsetHeight : offsetHeight;
              const rowsIndexToCompare = editElement.offsetHeight > offsetHeight ? rowIndex + 1 : rowIndex;
              if (rows.indexOf(rowsIndexToCompare) > -1) {
                editElement.classList.add('deckgo-highlight-code-line');
              }
            });
          }
        }
      }
      resolve();
    });
  }
  findRowsToHighlight() {
    return new Promise((resolve) => {
      let results = [];
      const rows = this.highlightLines.split(' ');
      if (rows && rows.length > 0) {
        rows.forEach((row) => {
          const index = row.split(',');
          if (index && index.length >= 1) {
            const start = parseInt(index[0], 0);
            const end = parseInt(index[1], 0);
            for (let i = start; i <= (isNaN(end) ? start : end); i++) {
              results.push(i);
            }
          }
        });
      }
      resolve(results);
    });
  }
  findNextAnchor(enter) {
    return new Promise(async (resolve) => {
      const elements = this.el.shadowRoot.querySelectorAll('span.deckgo-highlight-code-anchor');
      if (elements) {
        const elementsArray = enter ? Array.from(elements) : Array.from(elements).reverse();
        const anchor = elementsArray.find((element) => {
          return enter ? element.offsetTop > this.anchorOffsetTop : element.offsetTop < this.anchorOffsetTop;
        });
        if (anchor) {
          this.anchorOffsetTop = anchor.offsetTop;
          resolve({
            offsetTop: anchor.offsetTop,
            hasLineZoom: this.hasLineZoom(anchor.textContent),
          });
        }
        else if (!enter) {
          const elementCode = this.el.shadowRoot.querySelector('code');
          if (elementCode && elementCode.firstElementChild) {
            this.anchorOffsetTop = 0;
            resolve({
              offsetTop: 0,
              hasLineZoom: false,
            });
          }
          else {
            resolve(null);
          }
        }
        else {
          resolve(null);
        }
      }
      else {
        resolve(null);
      }
    });
  }
  zoomCode(zoom) {
    return new Promise((resolve) => {
      const container = this.el.shadowRoot.querySelector('div.deckgo-highlight-code-container');
      if (container) {
        container.style.setProperty('--deckgo-highlight-code-zoom', zoom ? '1.3' : '1');
      }
      resolve();
    });
  }
  hasLineZoom(line) {
    return line && this.anchorZoom && line.indexOf('@Prop') === -1 && line.split(' ').join('').indexOf(this.anchorZoom.split(' ').join('')) > -1;
  }
  async applyCode() {
    if (!this.editable) {
      return;
    }
    await this.copyCodeToSlot();
    await this.parseSlottedCode();
    this.codeDidChange.emit(this.el);
  }
  inputCode() {
    if (!this.editable) {
      return;
    }
    this.debounceUpdateSlot();
  }
  async copyCodeToSlot() {
    var _a, _b;
    const code = this.el.querySelector("[slot='code']");
    if (code) {
      code.innerHTML = (_b = (_a = this.refCode) === null || _a === void 0 ? void 0 : _a.innerText) === null || _b === void 0 ? void 0 : _b.replace(/(?:\n\n)/g, '\n').replace(/\u200B/g, '');
    }
  }
  edit() {
    var _a;
    if (!this.editable) {
      return;
    }
    (_a = this.refCode) === null || _a === void 0 ? void 0 : _a.focus();
  }
  render() {
    const hostClass = {
      'deckgo-highlight-code-carbon': this.terminal === _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].CARBON,
      'deckgo-highlight-code-ubuntu': this.terminal === _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].UBUNTU,
    };
    if (this.terminal === _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].CARBON) {
      hostClass[`deckgo-highlight-code-theme-${this.theme}`] = true;
    }
    return (Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["H"], { class: hostClass, onClick: () => this.edit() }, this.renderCarbon(), this.renderUbuntu(), Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("div", { class: "deckgo-highlight-code-container" }, Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("code", { contentEditable: this.editable, onBlur: async () => await this.applyCode(), onInput: () => this.inputCode(), onKeyDown: ($event) => this.catchTab($event), ref: (el) => (this.refCode = el) }), Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("slot", { name: "code" }))));
  }
  renderCarbon() {
    if (this.terminal !== _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].CARBON) {
      return undefined;
    }
    return [
      Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])(CarbonThemeStyle, { style: this.themeStyle }),
      Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("div", { class: "carbon" }, this.renderCarbonCircle('red'), this.renderCarbonCircle('yellow'), this.renderCarbonCircle('green')),
    ];
  }
  renderCarbonCircle(color) {
    return Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("div", { class: color });
  }
  renderUbuntu() {
    if (this.terminal !== _deckdeckgo_highlight_code_languages_93c3ae76_js__WEBPACK_IMPORTED_MODULE_1__["D"].UBUNTU) {
      return undefined;
    }
    return (Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("div", { class: "ubuntu" }, this.renderUbuntuCircle('close'), this.renderUbuntuCircle('minimize'), this.renderUbuntuCircle('maximize'), Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("p", null, Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("slot", { name: "user" }))));
  }
  renderUbuntuCircle(mode) {
    const symbol = mode === 'close' ? '&#10005;' : mode === 'minimize' ? '&#9472;' : '&#9723;';
    return (Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("div", { class: mode }, Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["h"])("span", { innerHTML: symbol })));
  }
  get el() { return Object(_index_5142f31b_js__WEBPACK_IMPORTED_MODULE_0__["g"])(this); }
  static get watchers() { return {
    "theme": ["loadTheme"],
    "language": ["onLanguage"],
    "lineNumbers": ["onLineNumbersChange"],
    "terminal": ["onCarbonChange"]
  }; }
};
DeckdeckgoHighlightCode.style = deckdeckgoHighlightCodeCss;




/***/ })

};;
//# sourceMappingURL=0.render-page.js.map