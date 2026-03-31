select
    variant,
    count(distinct customer_id) as customers,
    avg(converted_30d::float) as conversion_rate,
    avg(retained_180d::float) as retention_rate,
    avg(total_revenue) as avg_revenue
from customer_feature_mart
where eligibility_flag = 1
group by 1
order by 1;
