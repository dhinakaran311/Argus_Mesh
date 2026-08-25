// =============================================================================
// AbuseRing Sentinel — Neo4j Graph Schema
// graph/schema/schema.cypher
//
// Run this ONCE after connecting to the Neo4j Aura instance.
// Defines constraints, indexes, and documents node/relationship types.
// =============================================================================


// -----------------------------------------------------------------------------
// 1. UNIQUENESS CONSTRAINTS (also create indexes automatically)
// -----------------------------------------------------------------------------

CREATE CONSTRAINT customer_id_unique
    IF NOT EXISTS
    FOR (c:Customer) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT device_id_unique
    IF NOT EXISTS
    FOR (d:Device) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT ip_id_unique
    IF NOT EXISTS
    FOR (i:IP) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT merchant_id_unique
    IF NOT EXISTS
    FOR (m:Merchant) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT transaction_id_unique
    IF NOT EXISTS
    FOR (t:Transaction) REQUIRE t.id IS UNIQUE;


// -----------------------------------------------------------------------------
// 2. INDEXES FOR LOOKUP PERFORMANCE
// -----------------------------------------------------------------------------

CREATE INDEX customer_cluster_idx
    IF NOT EXISTS
    FOR (c:Customer) ON (c.cluster_id);

CREATE INDEX customer_abuse_idx
    IF NOT EXISTS
    FOR (c:Customer) ON (c.is_abuse);

CREATE INDEX customer_risk_idx
    IF NOT EXISTS
    FOR (c:Customer) ON (c.risk_score);

CREATE INDEX device_accounts_idx
    IF NOT EXISTS
    FOR (d:Device) ON (d.accounts_count);

CREATE INDEX ip_accounts_idx
    IF NOT EXISTS
    FOR (i:IP) ON (i.accounts_count);


// -----------------------------------------------------------------------------
// 3. NODE SCHEMAS (documented — not enforced in Aura Free)
// -----------------------------------------------------------------------------

// (:Customer)
//   id                  : String   — UUID, primary key
//   account_created_at  : DateTime — ISO 8601
//   customer_age_days   : Integer
//   location_city       : String
//   location_state      : String
//   email_domain        : String
//   is_verified         : Boolean
//   is_abuse            : Boolean  — ground truth label
//   cluster_id          : String   — abuse ring cluster id (if abuse)
//   ring_type           : String   — REFUND_RING | PROMO_ABUSE | RETURN_FRAUD | null
//   risk_score          : Float    — XGBoost model output ∈ [0, 1]
//   return_rate         : Float    — engineered feature
//   num_transactions    : Integer
//   num_returns         : Integer

// (:Device)
//   id                  : String   — UUID
//   device_type         : String   — MOBILE | DESKTOP | TABLET
//   browser             : String
//   os                  : String
//   accounts_count      : Integer  — total unique accounts seen on this device
//   first_seen          : DateTime

// (:IP)
//   id                  : String   — UUID
//   country             : String
//   region              : String
//   city                : String
//   isp                 : String
//   accounts_count      : Integer
//   first_seen          : DateTime

// (:Merchant)
//   id                  : String   — UUID
//   name                : String
//   baseline_refund_rate: Float    — merchant's normal refund rate

// (:Transaction)
//   id                  : String   — UUID
//   timestamp           : DateTime
//   amount              : Float    — INR
//   payment_method      : String
//   status              : String   — SUCCESS | FAILED | REVERSED
//   attempt_count       : Integer


// -----------------------------------------------------------------------------
// 4. RELATIONSHIP SCHEMAS (documented)
// -----------------------------------------------------------------------------

// (:Customer)-[:USES {first_seen: DateTime}]->(:Device)
//   Indicates a customer used this device for at least one transaction.

// (:Customer)-[:CONNECTS_FROM {count: Integer}]->(:IP)
//   Indicates a customer connected from this IP address.
//   count = number of transactions from this IP.

// (:Customer)-[:MADE {timestamp: DateTime, amount: Float}]->(:Transaction)
//   Customer initiated this transaction.

// (:Transaction)-[:AT {merchant_id: String}]->(:Merchant)
//   Transaction occurred at this merchant.

// (:Transaction)-[:VIA {device_id: String}]->(:Device)
//   Transaction was made from this device.


// -----------------------------------------------------------------------------
// 5. EXAMPLE VERIFICATION QUERIES (run after seeding)
// -----------------------------------------------------------------------------

// MATCH (c:Customer) RETURN count(c) AS customers;
// MATCH (d:Device)   RETURN count(d) AS devices;
// MATCH (i:IP)       RETURN count(i) AS ips;
// MATCH (m:Merchant) RETURN count(m) AS merchants;
// MATCH ()-[r:USES]->()          RETURN count(r) AS uses_rels;
// MATCH ()-[r:CONNECTS_FROM]->() RETURN count(r) AS connects_rels;
