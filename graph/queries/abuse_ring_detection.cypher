// =============================================================================
// AbuseRing Sentinel — Abuse Ring Detection Query
// graph/queries/abuse_ring_detection.cypher
//
// The primary ring detection query combining:
//   1. Shared device signal (many accounts per device)
//   2. Cluster return rate signal (abnormally high refund rate)
//   3. Account creation burst signal (many accounts in 24h window)
//   4. Transaction velocity signal
//
// This is the query run by the backend to score clusters for the dashboard.
// =============================================================================


// -----------------------------------------------------------------------------
// Query 1: PRIMARY RING DETECTION
// Find all device clusters that meet the abuse ring criteria
// Returns ranked clusters with composite risk score
// -----------------------------------------------------------------------------
MATCH (c:Customer)-[:USES]->(d:Device)
WITH d,
     collect(DISTINCT c)             AS members,
     count(DISTINCT c)               AS cluster_size,
     avg(c.return_rate)              AS cluster_return_rate,
     avg(c.risk_score)               AS avg_ml_score,
     sum(c.num_transactions)         AS total_transactions,
     sum(c.num_returns)              AS total_returns,
     min(c.account_created_at)       AS earliest_created,
     max(c.account_created_at)       AS latest_created

WHERE cluster_size >= 3

// Composite graph risk score (matches backend risk_engine.py weights)
WITH d, members, cluster_size, cluster_return_rate, avg_ml_score,
     total_transactions, total_returns, earliest_created, latest_created,

     // Signal 1: normalised cluster size (max ring = 30)
     CASE WHEN cluster_size >= 30 THEN 1.0
          ELSE toFloat(cluster_size) / 30 END                   AS size_signal,

     // Signal 2: cluster return rate (already 0-1)
     cluster_return_rate                                          AS return_signal,

     // Signal 3: creation burst (hours between first and last account)
     CASE
         WHEN duration.between(earliest_created, latest_created).hours <= 24 THEN 1.0
         WHEN duration.between(earliest_created, latest_created).hours <= 72 THEN 0.7
         WHEN duration.between(earliest_created, latest_created).hours <= 168 THEN 0.4
         ELSE 0.1
     END                                                          AS burst_signal

WITH d, members, cluster_size, cluster_return_rate, avg_ml_score,
     total_transactions, total_returns,
     size_signal, return_signal, burst_signal,

     // Graph risk score (used as the 30% graph component in final risk)
     (0.35 * size_signal + 0.40 * return_signal + 0.25 * burst_signal) AS graph_score

WHERE graph_score > 0.3

RETURN
    d.id                                    AS device_id,
    cluster_size,
    round(cluster_return_rate * 1000) / 1000 AS cluster_return_rate,
    round(avg_ml_score * 1000) / 1000       AS avg_ml_score,
    total_transactions,
    total_returns,
    round(graph_score * 1000) / 1000        AS graph_score,

    // Final combined risk (40% ML + 30% graph + 30% behaviour)
    round(
        (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate)
        * 1000
    ) / 1000                                AS combined_risk_score,

    CASE
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.80 THEN 'CRITICAL'
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.60 THEN 'HIGH'
        WHEN (0.40 * avg_ml_score + 0.30 * graph_score + 0.30 * cluster_return_rate) >= 0.30 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                     AS risk_level,

    [m IN members | m.id]                   AS member_ids,
    [m IN members | m.is_abuse]             AS member_labels

ORDER BY combined_risk_score DESC
LIMIT 25;


// -----------------------------------------------------------------------------
// Query 2: CONFIRMED ABUSE RING STATS
// Returns statistics on known abuse rings (from labels)
// Used to validate the graph detects known rings
// -----------------------------------------------------------------------------
MATCH (c:Customer {is_abuse: true})-[:USES]->(d:Device)
WITH d,
     collect(DISTINCT c) AS members,
     count(DISTINCT c) AS ring_members,
     avg(c.return_rate) AS ring_return_rate,
     collect(DISTINCT c.cluster_id) AS cluster_ids
WHERE ring_members >= 2
RETURN
    d.id                AS device_id,
    ring_members,
    round(ring_return_rate * 1000) / 1000 AS ring_return_rate,
    cluster_ids,
    [m IN members | m.id] AS member_ids
ORDER BY ring_members DESC;


// -----------------------------------------------------------------------------
// Query 3: FALSE POSITIVE ANALYSIS
// Devices shared by LEGITIMATE families (not rings)
// These are the hard negatives the model must not flag
// -----------------------------------------------------------------------------
MATCH (c:Customer)-[:USES]->(d:Device)
WHERE c.is_abuse = false
WITH d,
     collect(DISTINCT c) AS legit_members,
     count(DISTINCT c) AS legit_count,
     avg(c.return_rate) AS avg_return_rate

WHERE legit_count >= 2
  AND avg_return_rate < 0.20   // Legitimate family: low return rate

RETURN
    d.id            AS device_id,
    legit_count     AS legitimate_accounts_sharing_device,
    round(avg_return_rate * 1000) / 1000 AS avg_return_rate,
    [m IN legit_members | m.id] AS member_ids
ORDER BY legit_count DESC
LIMIT 20;


// -----------------------------------------------------------------------------
// Query 4: DASHBOARD SUMMARY
// Aggregate stats for the dashboard overview panel
// -----------------------------------------------------------------------------
MATCH (c:Customer)
WITH
    count(c)                    AS total_customers,
    sum(CASE WHEN c.is_abuse THEN 1 ELSE 0 END) AS abuse_customers,
    sum(CASE WHEN c.risk_score >= 0.80 THEN 1 ELSE 0 END) AS critical_risk,
    sum(CASE WHEN c.risk_score >= 0.60 AND c.risk_score < 0.80 THEN 1 ELSE 0 END) AS high_risk,
    avg(c.risk_score) AS avg_risk_score

MATCH (d:Device)
WITH total_customers, abuse_customers, critical_risk, high_risk, avg_risk_score,
     count(d) AS total_devices,
     sum(CASE WHEN d.accounts_count >= 10 THEN 1 ELSE 0 END) AS high_share_devices

MATCH (i:IP)
WITH total_customers, abuse_customers, critical_risk, high_risk, avg_risk_score,
     total_devices, high_share_devices,
     count(i) AS total_ips

RETURN
    total_customers,
    abuse_customers,
    round(toFloat(abuse_customers) / total_customers * 10000) / 100 AS abuse_rate_pct,
    critical_risk,
    high_risk,
    round(avg_risk_score * 1000) / 1000 AS avg_risk_score,
    total_devices,
    high_share_devices,
    total_ips;
