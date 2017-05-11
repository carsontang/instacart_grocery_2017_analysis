/*COPY orders(order_id, user_id, order_eval_set, order_number, order_dow, order_hour_of_day, days_since_prior)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/orders.csv'
DELIMITER ','
CSV HEADER;
*/

COPY departments(department_id, department)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/departments.csv'
DELIMITER ','
CSV HEADER;

COPY aisles(aisle_id, aisle)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/aisles.csv'
DELIMITER ','
CSV HEADER;

COPY products(product_id, product_name, aisle_id, department_id)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/products.csv'
DELIMITER ','
CSV HEADER;

COPY order_products__train(order_id, product_id, add_to_cart_order, reordered)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/order_products__train.csv'
DELIMITER ','
CSV HEADER;

COPY order_products__prior(order_id, product_id, add_to_cart_order, reordered)
FROM '/home/carson/Documents/instacart_grocery_2017_analysis/data/instacart_2017_05_01/order_products__prior.csv'
DELIMITER ','
CSV HEADER;
