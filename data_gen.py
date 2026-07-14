"""Synthetic dataset generator for the Neural Grep Transpiler.

Generates query -> tool-call JSONL rows in Needle's finetune format
(fields: query, tools, answers) for a single fixed tool, "ripgrep_search".

Three query typologies -- this model always extracts its best-effort search
terms for any input, including task instructions or chit-chat (that
distinction is a router's job, not this specialist's -- see gen_no_match,
kept below unused for that future router model):
  - literal (40%): exact config keys / identifiers / markers, is_regex=false
  - fuzzy (30%): abstract intent -> concrete literal keywords, is_regex=false
  - regex (30%): structural pattern descriptions -> regex, is_regex=true

Uses `random.Random(seed)` per example (not Python's salted `hash()`) so the
dataset is exactly reproducible across runs and machines -- this repo has
previously been bitten by non-reproducible-hash bugs in synthetic data
generation (see xariv_paper.md), so this generator re-derives every field
from the rendered example rather than trusting generator bookkeeping.
"""

import argparse
import json
import random
import re

from schema import TOOLS_JSON, make_answer

# ---------------------------------------------------------------------------
# Literal category
# ---------------------------------------------------------------------------

LITERAL_TERMS = [
    "DB_PORT=5432", "API_KEY", "REDIS_URL", "JWT_SECRET", "MAX_RETRIES=3",
    "TIMEOUT_MS=30000", "AWS_ACCESS_KEY_ID", "STRIPE_SECRET_KEY",
    "LOG_LEVEL=DEBUG", "NODE_ENV=production", "DATABASE_URL",
    "calculate_checksum", "UserRepository", "parse_config", "validate_token",
    "HttpClient", "SessionManager", "TokenRefresher", "RequestValidator",
    "ConnectionPool", "ERR_TIMEOUT", "E_NOENT", "HTTP_404", "ERR_INVALID_ARG",
    "TODO", "FIXME", "XXX", "DEPRECATED", "HACK",
    "import requests", "import numpy as np", "from django.db import models",
    "CACHE_TTL_SECONDS", "SECRET_KEY", "ALLOWED_HOSTS", "CORS_ORIGINS",
    "handle_webhook", "process_payment", "refresh_access_token",
    "MAX_CONNECTIONS", "RETRY_BACKOFF_FACTOR", "feature_flag_enabled",
    # broader domains: mobile, ML/data, infra, gaming, embedded, security
    "GRADLE_HOME", "ANDROID_SDK_ROOT", "CFBundleIdentifier", "Info.plist",
    "onCreate", "viewDidLoad", "componentDidMount", "useEffect",
    "learning_rate=0.001", "batch_size=32", "num_epochs", "torch.cuda",
    "train_test_split", "model.fit", "checkpoint_path", "EPOCHS",
    "KUBECONFIG", "helm upgrade", "terraform apply", "docker-compose.yml",
    "kubectl get pods", "namespace: default", "replicas: 3", "livenessProbe",
    "PlayerController", "onCollisionEnter", "Rigidbody", "GameObject",
    "spawn_enemy", "health_regen", "frame_rate", "vsync_enabled",
    "I2C_ADDRESS", "GPIO_PIN", "baud_rate=9600", "interrupt_handler",
    "watchdog_timer", "firmware_version", "bootloader",
    "sql_injection", "xss_payload", "csrf_token", "rate_limit_exceeded",
    "encryption_key", "salt_rounds=10", "bcrypt.hash", "oauth2_callback",
]

# Common/exotic/foreign/numeric single- or multi-word terms with no software
# connection at all -- ripgrep will happily search for any literal string,
# and a user might genuinely want to. Without examples like these, the model
# has never seen "search for an arbitrary plain-English word" during
# training and can behave unpredictably (garbled or silently-wrong output)
# on inputs like "oranges" or "context" that don't resemble an identifier.
GENERIC_WORDS = [
    "oranges", "bananas", "widgets", "apples", "mangoes", "kiwis", "cherries",
    "elephants", "tigers", "dolphins", "penguins", "octopus", "butterflies",
    "mountains", "rivers", "oceans", "deserts", "glaciers", "volcanoes",
    "umbrellas", "bicycles", "guitars", "violins", "trumpets", "drums",
    "coffee", "chocolate", "vanilla", "cinnamon", "saffron", "paprika",
    "castles", "dragons", "wizards", "pirates", "knights", "kingdoms",
    "galaxies", "nebulae", "comets", "asteroids", "telescopes", "satellites",
    "notebooks", "backpacks", "sunglasses", "sneakers", "blankets", "pillows",
    "gardens", "orchards", "vineyards", "meadows", "harvests", "seedlings",
    "context", "widget", "gizmo", "gadget", "trinket", "artifact", "relic",
    "sunrise", "sunset", "thunder", "lightning", "rainbow", "shadow", "mirror",
    "lantern", "compass", "anchor", "harbor", "voyage", "treasure", "phoenix",
    # abstract / exotic vocabulary
    "serendipity", "ephemeral", "labyrinth", "kaleidoscope", "mellifluous",
    "ubiquitous", "enigma", "paradox", "quixotic", "ethereal", "nebulous",
    "esoteric", "arcane", "wanderlust", "petrichor", "chiaroscuro", "sonder",
    # foreign loanwords
    "sushi", "tempura", "croissant", "baguette", "paella", "samosa",
    "falafel", "hummus", "kimchi", "matcha", "sriracha", "wasabi",
    # numbers / short alnum tokens
    "42", "3.14", "v2.0.1", "0xFF", "2026", "404",
]

LITERAL_PHRASES = [
    "find {terms}",
    "search for {terms}",
    "grep for {terms}",
    "where is {terms} used",
    "locate occurrences of {terms}",
    "look for {terms} in the codebase",
    "find all instances of {terms}",
    "show me every place {terms} appears",
    "find references to {terms}",
    "search the repo for {terms}",
    "find comments about {terms}",
    "search for comments mentioning {terms}",
    "find any mention of {terms}",
    "find the word {terms}",
    "{terms}",
]

LITERAL_PHRASES_CI = [
    "find {terms}, regardless of case",
    "search for {terms} case-insensitively",
    "look for {terms} ignoring case",
    "find {terms} in any case",
]


def _join_terms_en(terms):
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} or {terms[1]}"
    return ", ".join(terms[:-1]) + f", or {terms[-1]}"


def gen_literal(rng):
    pool = LITERAL_TERMS if rng.random() < 0.55 else GENERIC_WORDS
    n = 1 if rng.random() < 0.7 else rng.choice([2, 3])
    terms = rng.sample(pool, min(n, len(pool)))
    case_insensitive = rng.random() < 0.2
    phrase = rng.choice(LITERAL_PHRASES_CI if case_insensitive else LITERAL_PHRASES)
    query = phrase.format(terms=_join_terms_en(terms))
    return query, terms, False, case_insensitive


# ---------------------------------------------------------------------------
# Fuzzy / semantic category
# ---------------------------------------------------------------------------

CONCEPTS = {
    "payment failures": ["payment_failed", "stripe_error", "PaymentException"],
    "authentication errors": ["auth_failed", "Unauthorized", "InvalidCredentials"],
    "database connection issues": ["ConnectionError", "db_timeout", "connection_refused"],
    "rate limiting": ["rate_limit", "RateLimitExceeded", "throttle"],
    "retry logic": ["retry", "backoff", "max_retries"],
    "caching": ["cache_get", "cache_set", "redis_client"],
    "email sending": ["send_email", "smtp_client", "EmailService"],
    "file uploads": ["upload_file", "multipart", "FileUploadError"],
    "user logout": ["logout", "sign_out", "session_end"],
    "session management": ["session_id", "SessionStore", "session_expired"],
    "password resets": ["reset_password", "password_reset_token", "forgot_password"],
    "webhook handling": ["handle_webhook", "webhook_signature", "WebhookEvent"],
    "background jobs": ["enqueue_job", "TaskQueue", "worker_process"],
    "logging setup": ["configure_logging", "log_level", "get_logger"],
    "feature flags": ["feature_flag_enabled", "FeatureFlag", "is_enabled"],
    "pagination": ["page_size", "next_cursor", "paginate"],
    "search indexing": ["index_document", "SearchIndex", "reindex"],
    "image resizing": ["resize_image", "thumbnail", "ImageProcessor"],
    "csv export": ["export_csv", "CsvWriter", "to_csv"],
    "websocket connections": ["ws_connect", "WebSocketHandler", "on_message"],
    "health check endpoints": ["health_check", "healthz", "liveness_probe"],
    "circuit breakers": ["CircuitBreaker", "circuit_open", "trip_breaker"],
    "dependency injection": ["inject", "Container", "provide"],
    "config loading": ["load_config", "ConfigLoader", "parse_settings"],
    "database migrations": ["migrate", "Migration", "apply_migrations"],
    "serialization errors": ["SerializationError", "to_json", "deserialize"],
    "cors handling": ["cors_middleware", "CorsConfig", "allowed_origins"],
    "input validation": ["validate_input", "ValidationError", "is_valid"],
    "memory leaks": ["memory_leak", "gc_collect", "leaked_reference"],
    "connection pooling": ["ConnectionPool", "pool_acquire", "max_pool_size"],
    "token refresh": ["refresh_access_token", "TokenRefresher", "expired_token"],
    "permission checks": ["check_permission", "has_role", "PermissionDenied"],
    "audit logging": ["audit_log", "AuditEvent", "log_action"],
    "input sanitization": ["sanitize_input", "escape_html", "strip_tags"],
    "job scheduling": ["schedule_job", "CronTrigger", "next_run_time"],
    "distributed locking": ["acquire_lock", "DistributedLock", "release_lock"],
    "graceful shutdown": ["on_shutdown", "drain_connections", "sigterm_handler"],
    "config validation": ["validate_config", "ConfigError", "required_field_missing"],
    "service discovery": ["resolve_service", "ServiceRegistry", "discover_endpoint"],
    "load shedding": ["shed_load", "reject_request", "overload_protection"],
    "backpressure handling": ["apply_backpressure", "queue_full", "BackpressureError"],
    "idempotency keys": ["idempotency_key", "is_duplicate_request", "IdempotencyStore"],
    "webhook retries": ["retry_webhook", "WebhookRetryPolicy", "exponential_backoff"],
    "api versioning": ["api_version", "VersionHeader", "deprecated_version"],
    "soft deletes": ["soft_delete", "deleted_at", "restore_record"],
    "tenant isolation": ["tenant_id", "TenantContext", "cross_tenant_access"],
    "encryption at rest": ["encrypt_field", "KeyManagementService", "decrypt_field"],
    "secret rotation": ["rotate_secret", "SecretVersion", "rotate_credentials"],
    # frontend / mobile
    "form validation": ["validate_form", "FormError", "required_field"],
    "drag and drop": ["onDragStart", "onDrop", "DragContext"],
    "infinite scroll": ["load_more", "IntersectionObserver", "has_next_page"],
    "dark mode": ["theme_toggle", "prefers-color-scheme", "ThemeProvider"],
    "push notifications": ["register_device_token", "PushNotification", "send_push"],
    "deep linking": ["handle_deep_link", "UniversalLink", "parse_route"],
    "biometric auth": ["FaceID", "TouchID", "biometric_prompt"],
    "offline sync": ["sync_queue", "OfflineStore", "conflict_resolution"],
    "app crash reporting": ["CrashReporter", "log_exception", "stack_trace"],
    "screen navigation": ["NavigationController", "push_screen", "pop_to_root"],
    # ML / data
    "model training loop": ["train_step", "compute_loss", "backward"],
    "data preprocessing": ["normalize", "tokenize", "augment_data"],
    "hyperparameter tuning": ["grid_search", "learning_rate", "objective_fn"],
    "model checkpointing": ["save_checkpoint", "load_checkpoint", "best_model"],
    "feature engineering": ["extract_features", "FeatureVector", "one_hot_encode"],
    "embedding lookup": ["embedding_layer", "vocab_size", "lookup_table"],
    "gradient clipping": ["clip_grad_norm", "max_grad_norm", "gradient_explosion"],
    "data pipeline": ["DataLoader", "batch_iterator", "prefetch"],
    # infra / devops
    "container orchestration": ["kubectl_apply", "PodSpec", "rolling_update"],
    "ci pipeline": ["build_step", "run_tests", "deploy_stage"],
    "infrastructure as code": ["terraform_plan", "cloudformation", "provision"],
    "secrets management": ["vault_read", "SecretManager", "inject_secret"],
    "autoscaling": ["scale_out", "HorizontalPodAutoscaler", "cpu_threshold"],
    "blue green deploys": ["swap_traffic", "canary_release", "rollback"],
    "log aggregation": ["ship_logs", "LogAggregator", "structured_log"],
    "service mesh": ["sidecar_proxy", "mTLS", "traffic_split"],
    # gaming
    "collision detection": ["OnCollisionEnter", "BoundingBox", "raycast"],
    "inventory system": ["add_item", "InventorySlot", "stack_size"],
    "save game state": ["save_slot", "GameState", "serialize_progress"],
    "matchmaking": ["find_match", "MatchmakingQueue", "skill_rating"],
    "particle effects": ["ParticleSystem", "emit_particles", "shader_pass"],
    # testing / quality
    "flaky test detection": ["retry_flaky", "quarantine_test", "test_flake_rate"],
    "test fixtures": ["setup_fixture", "teardown", "mock_response"],
    "code coverage": ["coverage_report", "uncovered_lines", "branch_coverage"],
    "snapshot testing": ["snapshot_match", "update_snapshot", "SnapshotMismatch"],
    # security
    "sql injection prevention": ["parameterized_query", "escape_sql", "prepared_statement"],
    "secrets scanning": ["scan_for_secrets", "SecretLeak", "redact_credentials"],
    "dependency vulnerabilities": ["audit_dependencies", "CVE", "vulnerable_package"],
    "access control lists": ["AclEntry", "check_acl", "deny_by_default"],
}

FUZZY_PHRASES = [
    "find all code handling {concept}",
    "where do we handle {concept}",
    "search for {concept} logic",
    "look for code related to {concept}",
    "show me the {concept} implementation",
    "find everything related to {concept}",
    "locate the code that deals with {concept}",
    "find the part of the codebase that handles {concept}",
    "which files deal with {concept}",
    "point me to the {concept} code",
    "dig up the code responsible for {concept}",
    "I need to look at how we do {concept}",
    "surface the code for {concept}",
    "track down the {concept} implementation",
]

FUZZY_PHRASES_CI = [
    "find all code handling {concept}, regardless of case",
    "search for {concept} logic, ignoring case",
    "look for code related to {concept}, case-insensitively",
    "show me the {concept} implementation, ignoring case",
]


def gen_fuzzy(rng):
    concept = rng.choice(list(CONCEPTS.keys()))
    # Always return the concept's full keyword set: the query text only ever
    # names the concept, never which subset of keywords was intended, so a
    # randomly-sized subset would make the target label unrecoverable from
    # the query alone (the same query would map to different "correct"
    # answers across examples). A fixed, deterministic mapping keeps this
    # learnable.
    terms = list(CONCEPTS[concept])
    case_insensitive = rng.random() < 0.15
    phrase = rng.choice(FUZZY_PHRASES_CI if case_insensitive else FUZZY_PHRASES)
    query = phrase.format(concept=concept)
    return query, terms, False, case_insensitive


# ---------------------------------------------------------------------------
# Structural regex category
# ---------------------------------------------------------------------------

VERBS = ["fetch", "pull", "push", "sync", "load", "save", "emit", "poll", "flush", "spawn"]


def _p_async_prefix(rng):
    a, b = rng.sample(VERBS, 2)
    desc = f"async python functions starting with {a} or {b}"
    regex = f"async def ({a}|{b})_"
    return desc, regex


def _p_get_set(rng):
    return "functions named get_x or set_x", r"def (get|set)_\w+"


def _p_todo_fixme(rng):
    return "TODO or FIXME comments", r"#\s*(TODO|FIXME)"


def _p_email(rng):
    return "email addresses", r"[\w.+-]+@[\w-]+\.[\w.-]+"


def _p_ipv4(rng):
    return "IPv4 addresses", r"\b\d{1,3}(\.\d{1,3}){3}\b"


def _p_hex_color(rng):
    return "hex color codes", r"#[0-9a-fA-F]{6}\b"


def _p_trailing_ws(rng):
    return "lines with trailing whitespace", r"[ \t]+$"


def _p_import(rng):
    a, b = rng.sample(["requests", "urllib", "boto3", "numpy", "pandas", "asyncio"], 2)
    desc = f"import statements for {a} or {b}"
    regex = f"^import ({a}|{b})"
    return desc, regex


def _p_class_prefix(rng):
    a, b = rng.sample(["Base", "Abstract", "Mock", "Fake", "Test"], 2)
    desc = f"class definitions starting with {a} or {b}"
    regex = f"class ({a}|{b})\\w*"
    return desc, regex


def _p_long_number(rng):
    n = rng.choice([4, 5, 6])
    desc = f"numeric literals with {n} or more digits"
    regex = f"\\b\\d{{{n},}}\\b"
    return desc, regex


def _p_phone(rng):
    return "phone numbers", r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"


def _p_uuid(rng):
    return (
        "UUID values",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    )


def _p_url(rng):
    return "URLs starting with http or https", r"https?://\S+"


def _p_print_stmt(rng):
    return "leftover print statements", r"\bprint\("


def _p_env_assignment(rng):
    return "environment variable assignments", r"^[A-Z_][A-Z0-9_]*="


def _p_deprecated(rng):
    return "deprecated decorators", r"@deprecated"


def _p_test_fn(rng):
    return "test functions named test_x", r"def test_\w+"


def _p_private_method(rng):
    return "private methods starting with an underscore", r"def _\w+"


def _p_camel_call(rng):
    return "camelCase function calls", r"\b[a-z]+[A-Z]\w*\("


def _p_semicolon_eol(rng):
    return "lines ending in a semicolon followed by whitespace", r";\s*$"


def _p_sql_select(rng):
    return "SQL SELECT statements", r"SELECT\s+.*\s+FROM"


def _p_localhost_ip(rng):
    return "hardcoded localhost IP addresses", r"127\.0\.0\.1"


def _p_semver(rng):
    return "version strings like 1.2.3", r"\b\d+\.\d+\.\d+\b"


def _p_iso_date(rng):
    return "ISO dates like 2026-07-14", r"\b\d{4}-\d{2}-\d{2}\b"


def _p_currency(rng):
    return "dollar amounts", r"\$\d+(?:,\d{3})*(?:\.\d{2})?"


def _p_json_key(rng):
    key = rng.choice(["user_id", "session_token", "error_code", "status", "payload"])
    desc = f'JSON keys named "{key}"'
    regex = f'"{key}"\\s*:'
    return desc, regex


def _p_markdown_header(rng):
    level = rng.choice([1, 2, 3])
    desc = f"level-{level} markdown headers"
    regex = f"^{'#' * level}\\s+"
    return desc, regex


def _p_hex_literal(rng):
    return "hexadecimal literals", r"\b0x[0-9a-fA-F]+\b"


def _p_binary_literal(rng):
    return "binary literals", r"\b0b[01]+\b"


def _p_html_tag(rng):
    tag = rng.choice(["div", "span", "button", "input", "table"])
    desc = f"<{tag}> HTML tags"
    regex = f"</?{tag}[ >]"
    return desc, regex


def _p_file_extension(rng):
    ext = rng.choice(["py", "ts", "yaml", "json", "log"])
    desc = f".{ext} filenames"
    regex = f"\\b\\w+\\.{ext}\\b"
    return desc, regex


def _p_base64_blob(rng):
    return "long base64-looking strings", r"\b[A-Za-z0-9+/]{40,}={0,2}\b"


def _p_snake_case_const(rng):
    return "SCREAMING_SNAKE_CASE constants", r"\b[A-Z][A-Z0-9_]{2,}\b"


def _p_multiline_comment(rng):
    return "multi-line comment blocks", r"/\*[\s\S]*?\*/"


def _p_shebang(rng):
    return "shebang lines", r"^#!/"


PATTERN_GENERATORS = [
    _p_async_prefix, _p_get_set, _p_todo_fixme, _p_email, _p_ipv4,
    _p_hex_color, _p_trailing_ws, _p_import, _p_class_prefix, _p_long_number,
    _p_phone, _p_uuid, _p_url, _p_print_stmt, _p_env_assignment,
    _p_deprecated, _p_test_fn, _p_private_method, _p_camel_call,
    _p_semicolon_eol, _p_sql_select, _p_localhost_ip, _p_semver,
    _p_iso_date, _p_currency, _p_json_key, _p_markdown_header,
    _p_hex_literal, _p_binary_literal, _p_html_tag, _p_file_extension,
    _p_base64_blob, _p_snake_case_const, _p_multiline_comment, _p_shebang,
]

REGEX_PHRASES = [
    "find {desc}",
    "search for {desc}",
    "look for {desc}",
    "grep for {desc} using regex",
    "find all {desc}",
    "search the codebase for {desc}",
    "write a regex to find {desc}",
    "locate {desc} with a pattern match",
    "scan the repo for {desc}",
    "pattern-match for {desc}",
    "find every occurrence of {desc}",
]

REGEX_PHRASES_CI = [
    "find {desc}, regardless of case",
    "search for {desc} case-insensitively",
    "look for {desc}, ignoring case",
    "grep for {desc}, case-insensitive",
]


def gen_regex(rng):
    pattern_gen = rng.choice(PATTERN_GENERATORS)
    desc, regex = pattern_gen(rng)
    re.compile(regex)  # fail fast if a generator ever produces a bad pattern
    case_insensitive = rng.random() < 0.15
    phrase = rng.choice(REGEX_PHRASES_CI if case_insensitive else REGEX_PHRASES)
    query = phrase.format(desc=desc)
    return query, [regex], True, case_insensitive


# ---------------------------------------------------------------------------
# No-match category: not a search query at all -> answers=[]
# ---------------------------------------------------------------------------

TASK_VERBS = [
    "modify", "refactor", "rewrite", "update", "fix", "remove", "delete",
    "optimize", "clean up", "add logging to", "add error handling to",
    "add type hints to", "add tests for", "split", "merge", "document",
    "simplify", "extract a helper function from", "add retries to",
    "add caching to", "profile", "benchmark", "deploy", "revert",
]

TASK_TARGETS = [
    "the payment module", "this function", "the login flow",
    "reasoning_engine_data_gen.py", "the caching layer", "the API client",
    "the auth middleware", "the retry logic", "the config loader",
    "the tokenizer", "the training loop", "the eval script",
    "the CLI wrapper", "the database schema", "the webhook handler",
    "the session manager", "the rate limiter", "the export pipeline",
]

TASK_GOALS = [
    "add these label-decorrelated negatives", "support async",
    "reduce duplication", "fix the bug", "improve readability",
    "add type hints", "handle the edge case", "match the new schema",
    "pass the new tests", "reduce memory usage", "improve throughput",
]

TASK_TEMPLATES = [
    "{verb} {target}",
    "{verb} {target} to {goal}",
    "can you {verb} {target}",
    "please {verb} {target} to {goal}",
]

EXPLAIN_TEMPLATES = [
    "explain how {target} works",
    "what does {target} do",
    "why is {target} failing",
    "how do I use {target}",
    "can you summarize {target}",
    "walk me through {target}",
]

GENERIC_PROMPTS = [
    "what's the weather today",
    "tell me a joke",
    "how are you doing",
    "what time is it",
    "translate 'hello' to french",
    "write a haiku about autumn",
    "who won the game last night",
    "what's 2+2",
    "summarize the news",
    "recommend a good book",
    "what's your favorite color",
    "set a reminder for tomorrow",
]


def gen_no_match(rng):
    style = rng.random()
    if style < 0.65:
        verb = rng.choice(TASK_VERBS)
        target = rng.choice(TASK_TARGETS)
        goal = rng.choice(TASK_GOALS)
        template = rng.choice(TASK_TEMPLATES)
        query = template.format(verb=verb, target=target, goal=goal)
    elif style < 0.85:
        target = rng.choice(TASK_TARGETS)
        template = rng.choice(EXPLAIN_TEMPLATES)
        query = template.format(target=target)
    else:
        query = rng.choice(GENERIC_PROMPTS)
    return query, None, None, None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CATEGORY_TARGETS = [
    ("literal", gen_literal, 0.40),
    ("fuzzy", gen_fuzzy, 0.30),
    ("regex", gen_regex, 0.30),
]
# gen_no_match is kept above (unused here) for a future router model, whose
# job is exactly "does any tool apply" -- this standalone specialist always
# extracts its best-effort search terms instead, per user direction.


def generate(count, seed_start, exclude=None):
    """Generate `count` examples split across categories by their target share.

    Each category is filled independently to its own target count (rather
    than one shared weighted random draw) so that a category with a smaller
    combinatorial query-text space can't starve out the others before the
    overall count is reached. Returns (rows, category_counts).
    """
    seen_queries = set(exclude or ())
    rows = []
    category_counts = {}
    seed = seed_start

    for name, gen_fn, weight in CATEGORY_TARGETS:
        target = round(count * weight)
        produced = 0
        attempts = 0
        max_attempts = target * 300 + 5000
        while produced < target and attempts < max_attempts:
            attempts += 1
            rng = random.Random(seed)
            seed += 1
            query, terms, is_regex, case_insensitive = gen_fn(rng)
            if query in seen_queries:
                continue
            seen_queries.add(query)
            if terms is None:
                answers = []  # no applicable tool
            else:
                assert terms and all(isinstance(t, str) and t for t in terms)
                if is_regex:
                    for t in terms:
                        re.compile(t)
                answers = make_answer(terms, is_regex, case_insensitive)
            rows.append({
                "query": query,
                "tools": TOOLS_JSON,
                "answers": json.dumps(answers, separators=(",", ":")),
                "category": name,
            })
            produced += 1
        category_counts[name] = produced
        if produced < target:
            print(f"warning: only produced {produced}/{target} unique '{name}' "
                  f"examples (exhausted the combinatorial pool) -- consider "
                  f"adding more templates/terms for this category")

    random.Random(seed_start).shuffle(rows)
    return rows, category_counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--exclude-file", type=str, default=None,
                    help="JSONL file whose queries must not appear in the output (e.g. the train set, to keep an eval set held out)")
    args = p.parse_args()

    exclude = set()
    if args.exclude_file:
        with open(args.exclude_file) as f:
            for line in f:
                if line.strip():
                    exclude.add(json.loads(line)["query"])

    rows, category_counts = generate(args.count, args.seed_start, exclude=exclude)
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps({k: v for k, v in row.items() if k != "category"}) + "\n")

    print(f"Wrote {len(rows)} examples to {args.output}")
    print("Category breakdown:", category_counts)


if __name__ == "__main__":
    main()
