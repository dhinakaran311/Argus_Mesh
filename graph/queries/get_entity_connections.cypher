// =============================================================================
// AbuseRing Sentinel — Get Entity Connections (2-hop neighbourhood)
// graph/queries/get_entity_connections.cypher
//
// Given a customer_id, returns the full 2-hop neighbourhood:
//   Customer → Device/IP → Other Customers → Their Devices/IPs
//
// Used by the backend investigation agent.
// =============================================================================


// -----------------------------------------------------------------------------
// Query 1: Direct connections of a customer (1-hop)
// -----------------------------------------------------------------------------
MATCH (c:Customer {id: $customer_id})
OPTIONAL MATCH (c)-[:USES]->(d:Device)
OPTIONAL MATCH (c)-[:CONNECTS_FROM]->(i:IP)
OPTIONAL MATCH (c)-[:MADE]->(t:Transaction)-[:AT]->(m:Merchant)
RETURN
    c.id                            AS customer_id,
    c.risk_score                    AS risk_score,
    c.return_rate                   AS return_rate,
    c.cluster_id                    AS cluster_id,
    collect(DISTINCT d.id)          AS device_ids,
    collect(DISTINCT d.device_type) AS device_types,
    collect(DISTINCT d.accounts_count) AS device_account_counts,
    collect(DISTINCT i.id)          AS ip_ids,
    collect(DISTINCT i.city)        AS ip_cities,
    collect(DISTINCT m.id)          AS merchant_ids,
    count(DISTINCT t)               AS transaction_count;


// -----------------------------------------------------------------------------
// Query 2: Full 2-hop neighbourhood (co-users of same device/IP)
// Returns all customers connected to the target customer
// -----------------------------------------------------------------------------
MATCH (c:Customer {id: $customer_id})-[:USES|CONNECTS_FROM]->(shared)<-[:USES|CONNECTS_FROM]-(co:Customer)
WHERE co.id <> c.id
WITH
    shared,
    labels(shared)[0]               AS shared_type,
    co,
    count(DISTINCT shared)          AS shared_infrastructure_count
RETURN
    co.id                           AS co_customer_id,
    co.risk_score                   AS co_risk_score,
    co.return_rate                  AS co_return_rate,
    co.is_abuse                     AS co_is_abuse,
    co.cluster_id                   AS co_cluster_id,
    collect(DISTINCT shared_type + ':' + shared.id) AS shared_via,
    shared_infrastructure_count
ORDER BY co.risk_score DESC
LIMIT 50;


// -----------------------------------------------------------------------------
// Query 3: Degree centrality — how many unique customers does each device connect?
// Identifies hub devices used by many accounts (key abuse signal)
// -----------------------------------------------------------------------------
MATCH (d:Device)<-[:USES]-(c:Customer)
WITH d, count(DISTINCT c) AS degree, collect(DISTINCT c.id) AS connected_customers
WHERE degree >= 3
RETURN
    d.id            AS device_id,
    d.device_type   AS device_type,
    degree          AS customer_count,
    connected_customers
ORDER BY degree DESC
LIMIT 20;


// -----------------------------------------------------------------------------
// Query 4: Path between two customers (are they in the same ring?)
// Returns the shortest path connecting two customers through shared infrastructure
// -----------------------------------------------------------------------------
MATCH path = shortestPath(
    (c1:Customer {id: $customer_id})-[:USES|CONNECTS_FROM*..4]-(c2:Customer {id: $target_customer_id})
)
RETURN
    path,
    length(path) AS path_length,
    [node IN nodes(path) | labels(node)[0] + ':' + node.id] AS path_nodes;


// -----------------------------------------------------------------------------
// Query 5: Cluster size distribution (aggregate stats for dashboard)
// -----------------------------------------------------------------------------
MATCH (c1:Customer)-[:USES|CONNECTS_FROM]->(shared)<-[:USES|CONNECTS_FROM]-(c2:Customer)
WHERE c1.id < c2.id
WITH c1, c2
WITH c1.id AS anchor, count(DISTINCT c2) AS connections
RETURN
    connections AS cluster_size_minus_1,
    count(*) AS number_of_anchors,
    connections + 1 AS cluster_size
ORDER BY cluster_size DESC
LIMIT 30;
