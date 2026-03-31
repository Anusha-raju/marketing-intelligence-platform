-- Redshift-style customer feature mart sketch for Olist marketing intelligence

with order_revenue as (
    select
        oi.order_id,
        sum(oi.price) as item_price,
        sum(oi.freight_value) as freight_value,
        count(*) as item_count
    from olist_order_items_dataset oi
    group by 1
),
payments as (
    select
        op.order_id,
        sum(op.payment_value) as payment_value,
        max(op.payment_installments) as payment_installments
    from olist_order_payments_dataset op
    group by 1
),
reviews as (
    select
        review_id,
        order_id,
        avg(review_score) as review_score
    from olist_order_reviews_dataset
    group by 1,2
),
order_fact as (
    select
        o.customer_id,
        o.order_id,
        o.order_purchase_timestamp,
        r.item_price,
        r.freight_value,
        r.item_count,
        p.payment_value,
        p.payment_installments,
        rv.review_score
    from olist_orders_dataset o
    left join order_revenue r using(order_id)
    left join payments p using(order_id)
    left join reviews rv using(order_id)
),
customer_orders as (
    select
        customer_id,
        min(order_purchase_timestamp) as first_purchase,
        max(order_purchase_timestamp) as last_purchase,
        count(distinct order_id) as order_count,
        sum(payment_value) as total_revenue,
        avg(payment_value) as avg_order_value,
        avg(review_score) as avg_review_score
    from order_fact
    group by 1
),
marketing_touch_agg as (
    select
        customer_id,
        min(touch_timestamp) as first_touch,
        max(touch_timestamp) as last_touch,
        count(*) as num_touches,
        sum(click_flag) as num_clicks,
        sum(cost) as total_marketing_cost,
        count(distinct channel) as unique_channels
    from marketing_touchpoints
    group by 1
),
session_agg as (
    select
        customer_id,
        count(distinct session_id) as sessions,
        avg(pages_viewed) as avg_pages_viewed,
        avg(session_duration_seconds) as avg_session_duration,
        sum(add_to_cart_flag) as add_to_cart_sessions,
        sum(checkout_started_flag) as checkout_sessions,
        sum(purchase_flag) as purchase_sessions
    from customer_sessions
    group by 1
)
select
    c.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    co.*,
    mt.first_touch,
    mt.last_touch,
    mt.num_touches,
    mt.num_clicks,
    mt.total_marketing_cost,
    mt.unique_channels,
    sa.sessions,
    sa.avg_pages_viewed,
    sa.avg_session_duration,
    sa.add_to_cart_sessions,
    sa.checkout_sessions,
    sa.purchase_sessions
from olist_customers_dataset c
left join customer_orders co using(customer_id)
left join marketing_touch_agg mt using(customer_id)
left join session_agg sa using(customer_id);
