// =============================================================================
// AbuseRing Sentinel — Find All Suspicious Clusters
// graph/queries/find_clusters.cypher
//
// Detects connected components of customers sharing devices/IPs.
// Returns clusters sorted by combined risk signal.
//
// Note: Neo4j Aura Free does NOT include GDS (Graph Data Science) library.
//       This query uses pure Cypher traversal to find connected components
//       without requiring GDS.
// =============================================================================


// -----------------------------------------------------------------------------
// Query 1: Find all customers sharing the same device (device clusters)
// Returns: device_id, list of customer_ids, cluster_size
// -----------------------------------------------------------------------------
MATCH (c:Customer)-[:USES]->(d:Device)
WITH d, collect(c.id) AS customer_ids, count(c) AS cluster_size
WHERE cluster_size >= 2
RETURN
    d.id                AS device_id,
    d.accounts_count    AS device_accounts_count,
    customer_ids,
    cluster_size
ORDER BY cluster_size DESC
LIMIT 50;


// -----------------------------------------------------------------------------
// Query 2: Find high-risk device clusters (many accounts + high return rate)
// Returns ranked list of suspicious device clusters
// -----------------------------------------------------------------------------
MATCH (c:Customer)-[:USES]->(d:Device)
WITH d,
     collect(c) AS members,
     count(c) AS cluster_size,
     avg(c.return_rate) AS avg_return_rate,
     sum(c.num_transactions) AS total_txns
WHERE cluster_size >= 3
   AND avg_return_rate > 0.5
RETURN
    d.id                              AS device_id,
    cluster_size,
    round(avg_return_rate * 100) / 100 AS avg_return_rate,
    total_txns,
    [m IN members | m.id]             AS member_ids,
    [m IN members | m.is_abuse]       AS member_labels,
    // Composite risk signal
    (toFloat(cluster_size) / 30 * 0.4
     + avg_return_rate * 0.4
     + CASE WHEN cluster_size > 10 THEN 0.2 ELSE 0 END) AS risk_signal
ORDER BY risk_signal DESC
LIMIT 20;


// -----------------------------------------------------------------------------
// Query 3: Find customers in the same CLUSTER (sharing device OR IP)
// Two-hop traversal: Customer → Device/IP → Customer
// -----------------------------------------------------------------------------
MATCH (c1:Customer)-[:USES|CONNECTS_FROM]->(shared)<-[:USES|CONNECTS_FROM]-(c2:Customer)
WHERE c1.id <> c2.id
WITH c1, c2, collect(DISTINCT labels(shared)[0] + ':' + shared.id) AS shared_infrastructure
RETURN
    c1.id               AS customer_a,
    c2.id               AS customer_b,
    shared_infrastructure,
    size(shared_infrastructure) AS shared_count
ORDER BY shared_count DESC
LIMIT 100;


// -----------------------------------------------------------------------------
// Query 4: Full cluster summary — abuse ring statistics
// Groups all ring members by cluster_id
// -----------------------------------------------------------------------------
MATCH (c:Customer)
WHERE c.cluster_id IS NOT NULL
WITH c.cluster_id AS cluster_id,
     collect(c) AS members,
     count(c) AS member_count
RETURN
    cluster_id,
    member_count,
    round(avg([m IN members | m.return_rate]) * 1000) / 1000   AS avg_return_rate,
    round(avg([m IN members | m.risk_score]) * 1000) / 1000    AS avg_risk_score,
    sum([m IN members | m.num_transactions])                    AS total_transactions,
    sum([m IN members | coalesce(m.is_abuse, false) = true])   AS confirmed_abuse_count,
    [m IN members | m.id]                                       AS member_ids
ORDER BY avg_risk_score DESC;


// -----------------------------------------------------------------------------
// Query 5: Cluster exposure estimate
// Returns estimated financial exposure per cluster
// -----------------------------------------------------------------------------
MATCH (c:Customer)-[:MADE]->(t:Transaction)
WHERE c.cluster_id IS NOT NULL
   AND t.status = 'REVERSED'
WITH c.cluster_id AS cluster_id,
     count(DISTINCT c) AS members,
     sum(t.amount) AS total_reversed_amount,
     avg(c.risk_score) AS avg_risk_score
RETURN
    cluster_id,
    members,
    round(total_reversed_amount) AS exposure_inr,
    round(avg_risk_score * 1000) / 1000 AS avg_risk_score
ORDER BY exposure_inr DESC;
