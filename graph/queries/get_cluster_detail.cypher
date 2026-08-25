// =============================================================================
// AbuseRing Sentinel — Get Cluster Detail
// graph/queries/get_cluster_detail.cypher
//
// Given a cluster_id, returns all members, their devices, IPs,
// transactions, and risk metrics.
//
// Usage: Replace $cluster_id with the actual cluster ID string.
// =============================================================================


// -----------------------------------------------------------------------------
// Query 1: All members of a cluster with their profile
// -----------------------------------------------------------------------------
MATCH (c:Customer {cluster_id: $cluster_id})
OPTIONAL MATCH (c)-[:USES]->(d:Device)
OPTIONAL MATCH (c)-[:CONNECTS_FROM]->(i:IP)
RETURN
    c.id                    AS customer_id,
    c.account_created_at    AS created_at,
    c.is_abuse              AS is_abuse,
    c.risk_score            AS risk_score,
    c.return_rate           AS return_rate,
    c.num_transactions      AS num_transactions,
    c.num_returns           AS num_returns,
    collect(DISTINCT d.id)  AS devices,
    collect(DISTINCT d.device_type) AS device_types,
    collect(DISTINCT i.id)  AS ips,
    collect(DISTINCT i.city) AS cities
ORDER BY c.risk_score DESC;


// -----------------------------------------------------------------------------
// Query 2: Cluster summary statistics
// -----------------------------------------------------------------------------
MATCH (c:Customer {cluster_id: $cluster_id})
OPTIONAL MATCH (c)-[:USES]->(d:Device)
OPTIONAL MATCH (c)-[:CONNECTS_FROM]->(i:IP)
WITH
    count(DISTINCT c)   AS member_count,
    count(DISTINCT d)   AS device_count,
    count(DISTINCT i)   AS ip_count,
    avg(c.return_rate)  AS avg_return_rate,
    avg(c.risk_score)   AS avg_risk_score,
    min(c.account_created_at) AS earliest_account,
    max(c.account_created_at) AS latest_account
RETURN
    $cluster_id                                         AS cluster_id,
    member_count,
    device_count,
    ip_count,
    round(avg_return_rate * 1000) / 1000               AS avg_return_rate,
    round(avg_risk_score * 1000) / 1000                AS avg_risk_score,
    earliest_account,
    latest_account,
    duration.between(earliest_account, latest_account).hours AS creation_window_hours;


// -----------------------------------------------------------------------------
// Query 3: Shared infrastructure detail (for React Flow graph)
// Returns nodes and edges for visualisation
// -----------------------------------------------------------------------------
MATCH (c:Customer {cluster_id: $cluster_id})
OPTIONAL MATCH (c)-[r1:USES]->(d:Device)
OPTIONAL MATCH (c)-[r2:CONNECTS_FROM]->(i:IP)
OPTIONAL MATCH (c)-[r3:MADE]->(t:Transaction)-[r4:AT]->(m:Merchant)

WITH
    collect(DISTINCT {
        id: c.id,
        type: 'Customer',
        risk_score: c.risk_score,
        is_abuse: c.is_abuse,
        return_rate: c.return_rate
    }) AS customer_nodes,
    collect(DISTINCT {
        id: d.id,
        type: 'Device',
        device_type: d.device_type,
        accounts_count: d.accounts_count
    }) AS device_nodes,
    collect(DISTINCT {
        id: i.id,
        type: 'IP',
        city: i.city,
        accounts_count: i.accounts_count
    }) AS ip_nodes,
    collect(DISTINCT {
        id: m.id,
        type: 'Merchant',
        name: m.name
    }) AS merchant_nodes,
    collect(DISTINCT {source: c.id, target: d.id, type: 'USES'})          AS uses_edges,
    collect(DISTINCT {source: c.id, target: i.id, type: 'CONNECTS_FROM'}) AS ip_edges

RETURN
    customer_nodes,
    device_nodes,
    ip_nodes,
    merchant_nodes,
    uses_edges,
    ip_edges;


// -----------------------------------------------------------------------------
// Query 4: Cluster transaction timeline
// -----------------------------------------------------------------------------
MATCH (c:Customer {cluster_id: $cluster_id})-[:MADE]->(t:Transaction)
RETURN
    t.timestamp     AS timestamp,
    t.amount        AS amount,
    t.status        AS status,
    t.payment_method AS payment_method,
    c.id            AS customer_id
ORDER BY t.timestamp;
