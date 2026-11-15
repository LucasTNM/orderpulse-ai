-- ==============================================================================
-- OrderPulse AI — PostgreSQL Operational Schema & Seed Dataset (Read-Only MCP)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    customer_tier VARCHAR(20) NOT NULL,
    restaurant_name VARCHAR(120) NOT NULL,
    order_status VARCHAR(30) NOT NULL,
    total_amount_brl NUMERIC(10, 2) NOT NULL,
    promised_eta_min INTEGER NOT NULL,
    actual_delivery_min INTEGER,
    delay_minutes INTEGER NOT NULL DEFAULT 0,
    missing_items_reported BOOLEAN NOT NULL DEFAULT FALSE,
    missing_items_value_brl NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    prior_refunds_30d INTEGER NOT NULL DEFAULT 0,
    created_at_iso VARCHAR(40) NOT NULL,
    courier_gps_delayed BOOLEAN NOT NULL DEFAULT FALSE
);

INSERT INTO orders (
    order_id, customer_id, customer_tier, restaurant_name, order_status,
    total_amount_brl, promised_eta_min, actual_delivery_min, delay_minutes,
    missing_items_reported, missing_items_value_brl, prior_refunds_30d,
    created_at_iso, courier_gps_delayed
) VALUES
    ('ORD-9001', 'CUST-101', 'GOLD', 'Burger Lab Paulista', 'DELIVERED', 89.90, 35, 90, 55, FALSE, 0.00, 0, '2026-11-15T11:00:00-03:00', TRUE),
    ('ORD-9002', 'CUST-102', 'DIAMOND', 'Japa Prime Jardins', 'DELIVERED', 245.50, 40, 95, 55, FALSE, 0.00, 0, '2026-11-15T10:30:00-03:00', TRUE),
    ('ORD-9003', 'CUST-103', 'STANDARD', 'Pizzaria Bella Napoli', 'DELIVERED', 74.00, 45, 47, 2, FALSE, 0.00, 5, '2026-11-15T11:10:00-03:00', FALSE),
    ('ORD-9004', 'CUST-104', 'GOLD', 'Smash & Co', 'DELIVERED', 68.50, 30, 33, 3, TRUE, 14.90, 1, '2026-11-15T11:20:00-03:00', FALSE),
    ('ORD-9005', 'CUST-105', 'STANDARD', 'Cantina Roma', 'DELIVERED', 112.00, 40, 62, 22, FALSE, 0.00, 0, '2026-11-15T10:45:00-03:00', TRUE),
    ('ORD-9006', 'CUST-106', 'STANDARD', 'Taqueria El Sol', 'DELIVERED', 100.00, 35, 71, 36, FALSE, 0.00, 1, '2026-11-15T10:15:00-03:00', TRUE),
    ('ORD-9007', 'CUST-107', 'DIAMOND', 'Bistro Vila Madalena', 'DELIVERED', 120.00, 40, 75, 35, FALSE, 0.00, 0, '2026-11-15T10:20:00-03:00', TRUE),
    ('ORD-9008', 'CUST-108', 'GOLD', 'Churrascaria Sul', 'CREATED', 135.00, 45, NULL, 0, FALSE, 0.00, 0, '2026-11-15T12:10:00-03:00', FALSE),
    ('ORD-9009', 'CUST-109', 'STANDARD', 'Padaria Central', 'IN_TRANSIT', 58.00, 35, NULL, 5, FALSE, 0.00, 0, '2026-11-15T11:55:00-03:00', FALSE),
    ('ORD-9010', 'CUST-110', 'DIAMOND', 'Osteria Milano', 'ACCEPTED', 198.00, 50, NULL, 0, FALSE, 0.00, 0, '2026-11-15T12:15:00-03:00', FALSE),
    ('ORD-9011', 'CUST-111', 'STANDARD', 'Poke House', 'DELIVERED', 210.00, 35, 38, 3, TRUE, 165.00, 0, '2026-11-15T10:05:00-03:00', FALSE),
    ('ORD-9012', 'CUST-112', 'STANDARD', 'Acai Express', 'DELIVERED', 42.00, 25, 28, 3, TRUE, 18.00, 4, '2026-11-15T11:40:00-03:00', FALSE),
    ('ORD-9013', 'CUST-113', 'GOLD', 'Sushi Mori', 'PREPARING', 94.00, 45, NULL, 0, FALSE, 0.00, 1, '2026-11-15T12:05:00-03:00', FALSE),
    ('ORD-9014', 'CUST-114', 'STANDARD', 'Hamburgueria 24h', 'DELIVERED', 149.90, 40, 92, 52, FALSE, 0.00, 2, '2026-11-15T09:40:00-03:00', TRUE),
    ('ORD-9015', 'CUST-115', 'GOLD', 'Temakeria Barra', 'DELIVERED', 150.01, 35, 88, 53, FALSE, 0.00, 0, '2026-11-15T09:50:00-03:00', TRUE)
ON CONFLICT (order_id) DO NOTHING;
