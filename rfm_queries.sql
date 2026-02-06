-- ============================================
-- RFM ANALYSIS SQL QUERIES
-- Customer Segmentation for Targeted Marketing
-- ============================================

-- ============================================
-- 1. CALCULATE RFM METRICS
-- ============================================

-- Calculate Recency, Frequency, and Monetary for each customer
CREATE VIEW customer_rfm AS
SELECT 
    CustomerID,
    -- Recency: Days since last purchase
    DATEDIFF(day, MAX(InvoiceDate), '2011-12-31') as Recency,
    -- Frequency: Number of unique orders
    COUNT(DISTINCT InvoiceNo) as Frequency,
    -- Monetary: Total amount spent
    ROUND(SUM(Quantity * UnitPrice), 2) as Monetary
FROM sales_data
WHERE CustomerID IS NOT NULL
    AND Quantity > 0
    AND UnitPrice > 0
GROUP BY CustomerID;

-- ============================================
-- 2. CREATE RFM SCORES (1-5 SCALE)
-- ============================================

-- Assign scores using NTILE function
CREATE VIEW customer_rfm_scores AS
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    -- Recency score (lower is better, so reverse the order)
    6 - NTILE(5) OVER (ORDER BY Recency) as R_Score,
    -- Frequency score (higher is better)
    NTILE(5) OVER (ORDER BY Frequency) as F_Score,
    -- Monetary score (higher is better)
    NTILE(5) OVER (ORDER BY Monetary) as M_Score
FROM customer_rfm;

-- ============================================
-- 3. CREATE CUSTOMER SEGMENTS
-- ============================================

-- Segment customers based on RFM scores
CREATE VIEW customer_segments AS
SELECT 
    *,
    -- Combined RFM Score
    CAST(R_Score AS VARCHAR) + CAST(F_Score AS VARCHAR) + CAST(M_Score AS VARCHAR) as RFM_Score,
    -- Segment label
    CASE 
        WHEN R_Score >= 4 AND F_Score >= 4 AND M_Score >= 4 THEN 'Champions'
        WHEN R_Score >= 3 AND F_Score >= 3 AND M_Score >= 3 THEN 'Loyal Customers'
        WHEN R_Score <= 2 AND F_Score >= 3 AND M_Score >= 3 THEN 'At Risk'
        WHEN R_Score <= 2 AND F_Score <= 2 THEN 'Lost Customers'
        WHEN R_Score >= 4 AND F_Score <= 2 THEN 'New Customers'
        ELSE 'Potential Loyalists'
    END as Customer_Segment
FROM customer_rfm_scores;

-- ============================================
-- 4. SEGMENT ANALYSIS
-- ============================================

-- Count and revenue by segment
SELECT 
    Customer_Segment,
    COUNT(*) as Customer_Count,
    ROUND(AVG(Recency), 1) as Avg_Recency,
    ROUND(AVG(Frequency), 1) as Avg_Frequency,
    ROUND(AVG(Monetary), 2) as Avg_Monetary,
    ROUND(SUM(Monetary), 2) as Total_Revenue,
    ROUND(SUM(Monetary) * 100.0 / SUM(SUM(Monetary)) OVER (), 1) as Revenue_Percentage
FROM customer_segments
GROUP BY Customer_Segment
ORDER BY Total_Revenue DESC;

-- ============================================
-- 5. TOP CUSTOMERS BY SEGMENT
-- ============================================

-- Champions (Top tier customers)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    RFM_Score
FROM customer_segments
WHERE Customer_Segment = 'Champions'
ORDER BY Monetary DESC
LIMIT 20;

-- At Risk Customers (Need immediate attention)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    RFM_Score
FROM customer_segments
WHERE Customer_Segment = 'At Risk'
ORDER BY Monetary DESC
LIMIT 20;

-- ============================================
-- 6. CUSTOMER LIFETIME VALUE (CLV)
-- ============================================

-- Estimate CLV based on historical data
SELECT 
    CustomerID,
    Customer_Segment,
    Frequency as Total_Orders,
    Monetary as Total_Revenue,
    ROUND(Monetary / NULLIF(Frequency, 0), 2) as Avg_Order_Value,
    -- Estimated annual value (assuming purchase frequency continues)
    ROUND((Monetary / NULLIF(Frequency, 0)) * (365.0 / NULLIF(Recency, 0)) * 12, 2) as Estimated_Annual_Value
FROM customer_segments
WHERE Frequency > 0 AND Recency > 0
ORDER BY Estimated_Annual_Value DESC
LIMIT 50;

-- ============================================
-- 7. MONTHLY COHORT ANALYSIS
-- ============================================

-- Customer acquisition by month
SELECT 
    DATE_TRUNC('month', first_purchase_date) as Cohort_Month,
    COUNT(*) as New_Customers,
    ROUND(AVG(total_spent), 2) as Avg_First_Month_Spend
FROM (
    SELECT 
        CustomerID,
        MIN(InvoiceDate) as first_purchase_date,
        SUM(CASE WHEN InvoiceDate = MIN(InvoiceDate) 
            THEN Quantity * UnitPrice ELSE 0 END) as total_spent
    FROM sales_data
    WHERE CustomerID IS NOT NULL
    GROUP BY CustomerID
) cohorts
GROUP BY DATE_TRUNC('month', first_purchase_date)
ORDER BY Cohort_Month;

-- ============================================
-- 8. CHURN PREDICTION
-- ============================================

-- Identify customers likely to churn (no purchase in 90+ days)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    Customer_Segment,
    CASE 
        WHEN Recency > 180 THEN 'High Churn Risk'
        WHEN Recency > 90 THEN 'Medium Churn Risk'
        ELSE 'Low Churn Risk'
    END as Churn_Risk
FROM customer_segments
WHERE Recency > 90
ORDER BY Monetary DESC, Recency DESC;

-- ============================================
-- 9. REVENUE IMPACT BY SEGMENT
-- ============================================

-- Compare revenue contribution
WITH segment_totals AS (
    SELECT 
        Customer_Segment,
        COUNT(*) as customer_count,
        SUM(Monetary) as segment_revenue
    FROM customer_segments
    GROUP BY Customer_Segment
)
SELECT 
    Customer_Segment,
    customer_count,
    ROUND(customer_count * 100.0 / SUM(customer_count) OVER (), 1) as pct_of_customers,
    ROUND(segment_revenue, 2) as total_revenue,
    ROUND(segment_revenue * 100.0 / SUM(segment_revenue) OVER (), 1) as pct_of_revenue,
    ROUND(segment_revenue / customer_count, 2) as revenue_per_customer
FROM segment_totals
ORDER BY total_revenue DESC;

-- ============================================
-- 10. ACTIONABLE INSIGHTS
-- ============================================

-- Marketing campaign targets
SELECT 
    'Champions' as Campaign_Target,
    COUNT(*) as Target_Size,
    ROUND(SUM(Monetary), 2) as Total_Value,
    'VIP Rewards Program' as Recommended_Action
FROM customer_segments
WHERE Customer_Segment = 'Champions'

UNION ALL

SELECT 
    'At Risk' as Campaign_Target,
    COUNT(*) as Target_Size,
    ROUND(SUM(Monetary), 2) as Total_Value,
    'Re-engagement Campaign (15% discount)' as Recommended_Action
FROM customer_segments
WHERE Customer_Segment = 'At Risk'

UNION ALL

SELECT 
    'Lost Customers' as Campaign_Target,
    COUNT(*) as Target_Size,
    ROUND(SUM(Monetary), 2) as Total_Value,
    'Win-back Campaign (25% discount)' as Recommended_Action
FROM customer_segments
WHERE Customer_Segment = 'Lost Customers';
