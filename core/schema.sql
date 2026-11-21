
CREATE TABLE IF NOT EXISTS idempotency_records (
    tenant_id TEXT NOT NULL,
    source TEXT NOT NULL,
    delivery_key TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    input_json TEXT NOT NULL,
    request_id TEXT NOT NULL,
    profile_slug TEXT NOT NULL,
    profile_revision TEXT NOT NULL,
    received_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, source, delivery_key),
    UNIQUE (tenant_id, request_id)
) STRICT;

CREATE TABLE IF NOT EXISTS transaction_telemetry (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    processing_time_sec REAL CHECK (processing_time_sec >= 0),
    tokens_in INTEGER CHECK (tokens_in >= 0),
    tokens_out INTEGER CHECK (tokens_out >= 0),
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    mode TEXT CHECK (mode IN ('llm', 'deterministic', 'fallback')),
    fallback_reason TEXT,
    ab_variant TEXT,
    channel TEXT NOT NULL,
    is_after_hours INTEGER NOT NULL CHECK (is_after_hours IN (0, 1)),
    need_human INTEGER CHECK (need_human IN (0, 1)),
    saved_minutes REAL,
    run_origin TEXT NOT NULL CHECK (run_origin IN ('mock', 'benchmark', 'pilot')),
    outcome TEXT NOT NULL CHECK (outcome IN ('running', 'waiting_human', 'completed', 'failed', 'reconcile')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id),
    FOREIGN KEY (tenant_id, request_id) REFERENCES idempotency_records (tenant_id, request_id),
    CHECK (CASE WHEN mode = 'fallback'
        THEN fallback_reason IS NOT NULL AND length(fallback_reason) > 0
        ELSE fallback_reason IS NULL END)
) STRICT;

CREATE TABLE IF NOT EXISTS transaction_stages (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('S1', 'S2', 'S3')),
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    response_json TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_code TEXT,
    mode TEXT CHECK (mode IN ('llm', 'deterministic', 'fallback')),
    fallback_reason TEXT,
    prompt_versions_json TEXT,
    tokens_in INTEGER CHECK (tokens_in >= 0),
    tokens_out INTEGER CHECK (tokens_out >= 0),
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    PRIMARY KEY (tenant_id, request_id, stage, attempt),
    FOREIGN KEY (tenant_id, request_id) REFERENCES transaction_telemetry (tenant_id, request_id),
    CHECK (status <> 'completed' OR (response_json IS NOT NULL AND mode IS NOT NULL)),
    CHECK (CASE WHEN mode = 'fallback'
        THEN fallback_reason IS NOT NULL AND length(fallback_reason) > 0
        ELSE fallback_reason IS NULL END)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS one_running_attempt
    ON transaction_stages (tenant_id, request_id, stage) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS crm_publications (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    stage_attempt INTEGER NOT NULL,
    purpose TEXT NOT NULL CHECK (
        purpose IN ('stage_result', 'human_review') OR
        (stage = 'S3' AND purpose = 'human_review:' || stage_attempt)
    ),
    portal_id TEXT NOT NULL,
    external_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('approved', 'publishing', 'published', 'reconcile')),
    approved_at TEXT NOT NULL,
    approval_ref TEXT NOT NULL,
    command_json TEXT NOT NULL,
    owner_token TEXT,
    entity_type_id INTEGER NOT NULL CHECK (entity_type_id > 0),
    item_id INTEGER CHECK (item_id > 0),
    task_required INTEGER NOT NULL CHECK (task_required IN (0, 1)),
    task_id INTEGER CHECK (task_id > 0),
    pending_step TEXT CHECK (pending_step IN ('item', 'task')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, stage, purpose),
    UNIQUE (portal_id, external_key),
    FOREIGN KEY (tenant_id, request_id, stage, stage_attempt)
        REFERENCES transaction_stages (tenant_id, request_id, stage, attempt),
    CHECK (status <> 'publishing' OR owner_token IS NOT NULL),
    CHECK (status <> 'published' OR
        (pending_step IS NULL AND item_id IS NOT NULL AND (task_required = 0 OR task_id IS NOT NULL)))
) STRICT;

CREATE TABLE IF NOT EXISTS s1_extractions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    response_json TEXT,
    mode TEXT CHECK (mode IN ('llm', 'deterministic', 'fallback')),
    fallback_reason TEXT,
    tokens_in INTEGER CHECK (tokens_in >= 0),
    tokens_out INTEGER CHECK (tokens_out >= 0),
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    PRIMARY KEY (tenant_id, request_id),
    FOREIGN KEY (tenant_id, request_id) REFERENCES idempotency_records (tenant_id, request_id)
) STRICT;

CREATE TABLE IF NOT EXISTS s2_extractions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    quote_id TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    response_json TEXT,
    mode TEXT CHECK (mode IN ('llm', 'deterministic', 'fallback')),
    fallback_reason TEXT,
    tokens_in INTEGER CHECK (tokens_in >= 0),
    tokens_out INTEGER CHECK (tokens_out >= 0),
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    PRIMARY KEY (tenant_id, request_id, quote_id),
    FOREIGN KEY (tenant_id, request_id) REFERENCES idempotency_records (tenant_id, request_id)
) STRICT;

CREATE TABLE IF NOT EXISTS s2_intakes (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    delivery_key TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    PRIMARY KEY (tenant_id, request_id),
    UNIQUE (tenant_id, delivery_key),
    FOREIGN KEY (tenant_id, request_id) REFERENCES idempotency_records (tenant_id, request_id)
) STRICT;

CREATE TABLE IF NOT EXISTS publication_decisions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'cancel')),
    evidence_ref TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, stage, attempt),
    FOREIGN KEY (tenant_id, request_id, stage, attempt)
        REFERENCES transaction_stages (tenant_id, request_id, stage, attempt)
) STRICT;

CREATE TABLE IF NOT EXISTS crm_publication_comments (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    purpose TEXT NOT NULL,
    pending INTEGER NOT NULL DEFAULT 0 CHECK (pending IN (0, 1)),
    comment_id INTEGER CHECK (comment_id > 0),
    PRIMARY KEY (tenant_id, request_id, stage, purpose),
    FOREIGN KEY (tenant_id, request_id, stage, purpose)
        REFERENCES crm_publications (tenant_id, request_id, stage, purpose),
    CHECK (pending = 0 OR comment_id IS NULL)
) STRICT;

CREATE TABLE IF NOT EXISTS s3_packages (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    delivery_key TEXT NOT NULL,
    input_sha256 TEXT NOT NULL CHECK (length(input_sha256) = 64),
    input_json TEXT NOT NULL,
    s2_item_id INTEGER NOT NULL CHECK (s2_item_id > 0),
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    created_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, package_id),
    UNIQUE (tenant_id, request_id, delivery_key),
    FOREIGN KEY (tenant_id, request_id) REFERENCES idempotency_records (tenant_id, request_id)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS one_active_s3_package
    ON s3_packages (tenant_id, request_id) WHERE active = 1;

CREATE TABLE IF NOT EXISTS s3_tender_gates (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    response_json TEXT NOT NULL,
    checked_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, package_id),
    FOREIGN KEY (tenant_id, request_id, package_id)
        REFERENCES s3_packages (tenant_id, request_id, package_id)
) STRICT;

CREATE TABLE IF NOT EXISTS s3_extractions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_sha256 TEXT NOT NULL CHECK (length(source_sha256) = 64),
    profile_revision TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_variant TEXT NOT NULL,
    response_json TEXT,
    cost_rub_e8 INTEGER CHECK (cost_rub_e8 >= 0),
    mode TEXT CHECK (mode IN ('llm', 'deterministic', 'fallback')),
    fallback_reason TEXT,
    tokens_in INTEGER CHECK (tokens_in >= 0),
    tokens_out INTEGER CHECK (tokens_out >= 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    recovery_ref TEXT,
    PRIMARY KEY (tenant_id, request_id, package_id, document_id),
    FOREIGN KEY (tenant_id, request_id, package_id)
        REFERENCES s3_packages (tenant_id, request_id, package_id)
) STRICT;

CREATE TABLE IF NOT EXISTS s3_attempts (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    stage TEXT NOT NULL DEFAULT 'S3' CHECK (stage = 'S3'),
    gate_json TEXT NOT NULL,
    penalty_sha256 TEXT CHECK (length(penalty_sha256) = 64),
    PRIMARY KEY (tenant_id, request_id, attempt),
    FOREIGN KEY (tenant_id, request_id, package_id)
        REFERENCES s3_packages (tenant_id, request_id, package_id),
    FOREIGN KEY (tenant_id, request_id, stage, attempt)
        REFERENCES transaction_stages (tenant_id, request_id, stage, attempt)
) STRICT;

CREATE TABLE IF NOT EXISTS s3_decisions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    package_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('ACCEPTED', 'CLAIM_REQUIRED')),
    evidence_ref TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, attempt),
    FOREIGN KEY (tenant_id, request_id, attempt) REFERENCES s3_attempts,
    FOREIGN KEY (tenant_id, request_id, package_id) REFERENCES s3_packages
) STRICT;

CREATE TABLE IF NOT EXISTS s3_penalty_terms (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    s2_attempt INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'S2' CHECK (stage = 'S2'),
    evidence_ref TEXT NOT NULL,
    award_sha256 TEXT NOT NULL CHECK (length(award_sha256) = 64),
    base TEXT NOT NULL CHECK (base = 'AWARDED_TOTAL_RUB'),
    daily_penalty_pct TEXT NOT NULL,
    max_penalty_pct TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, s2_attempt),
    FOREIGN KEY (tenant_id, request_id, stage, s2_attempt)
        REFERENCES transaction_stages (tenant_id, request_id, stage, attempt)
) STRICT;

CREATE TABLE IF NOT EXISTS s2_award_items (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    stage TEXT NOT NULL DEFAULT 'S2' CHECK (stage = 'S2'),
    items_json TEXT NOT NULL,
    PRIMARY KEY (tenant_id, request_id, attempt),
    FOREIGN KEY (tenant_id, request_id, stage, attempt)
        REFERENCES transaction_stages (tenant_id, request_id, stage, attempt)
) STRICT;
