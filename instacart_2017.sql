CREATE TYPE eval_set AS ENUM ('prior', 'train', 'test');

CREATE TABLE orders (
	order_id serial PRIMARY KEY,
	user_id serial,
	order_eval_set eval_set,
	order_number smallserial,
	order_dow smallserial,
	order_hour_of_day smallserial,
	days_since_prior numeric DEFAULT NULL /*  CONSTRAINT positive_days_since_prior CHECK */
);

CREATE TABLE departments (
	department_id serial PRIMARY KEY,
	department text
);

CREATE TABLE aisles (
	aisle_id serial PRIMARY KEY,
	aisle text
);

CREATE TABLE products (
	product_id serial PRIMARY KEY,
	product_name text,
	aisle_id serial REFERENCES aisles(aisle_id),
	department_id serial REFERENCES departments(department_id)
);

CREATE TABLE order_products__train (
	order_id serial references orders(order_id),
	product_id serial references products(product_id),
	add_to_cart_order serial,
	reordered boolean
);

SELECT product_id, AVG(add_to_cart_order) avg_order
  FROM order_products__train
  GROUP BY product_id
  ORDER BY avg_order DESC
LIMIT 10;

SELECT * FROM order_products__train WHERE product_id = 42148;

/* alter table orders alter column days_since_prior set data type numeric; */
COPY orders(order_id, user_id, order_eval_set, order_number, order_dow, order_hour_of_day, days_since_prior) FROM '/Users/ctang/Downloads/instacart_2017_05_01/orders.csv' DELIMITER ',' CSV HEADER;
COPY departments(department_id, department) FROM '/Users/ctang/Downloads/instacart_2017_05_01/departments.csv' DELIMITER ',' CSV HEADER;
COPY aisles(aisle_id, aisle) FROM '/Users/ctang/Downloads/instacart_2017_05_01/aisles.csv' DELIMITER ',' CSV HEADER;
COPY products(product_id, product_name, aisle_id, department_id) FROM '/Users/ctang/Downloads/instacart_2017_05_01/products.csv' DELIMITER ',' CSV HEADER;
COPY order_products__train(order_id, product_id, add_to_cart_order, reordered) FROM '/Users/ctang/Downloads/instacart_2017_05_01/order_products__train.csv' DELIMITER ',' CSV HEADER;

select p.product_name, d.department from products p inner join departments d on p.department_id = d.department_id;

/* Top 10 departments with the most product listings */
SELECT d.department, COUNT(*) AS product_count
  FROM products p
  JOIN departments d ON p.department_id = d.department_id
  GROUP BY d.department
  ORDER BY product_count DESC
 LIMIT 10;

/*
   department    | product_count
-----------------+---------------
 personal care   |          6563
 snacks          |          6264
 pantry          |          5371
 beverages       |          4365
 frozen          |          4007
 dairy eggs      |          3449
 household       |          3085
 canned goods    |          2092
 dry goods pasta |          1858
 produce         |          1684
 */

/* Top 10 most ordered items */
SELECT p.product_name, o.num_ordered, a.aisle
  FROM (SELECT o.product_id as product_id, COUNT(*) AS num_ordered
          FROM order_products__train o
      GROUP BY o.product_id
      ORDER BY num_ordered DESC) AS o
  JOIN products p ON p.product_id = o.product_id
  JOIN aisles a ON p.aisle_id = a.aisle_id
 LIMIT 10;

/*
      product_name      | num_ordered |           aisle
------------------------+-------------+----------------------------
 Banana                 |       18726 | fresh fruits
 Bag of Organic Bananas |       15480 | fresh fruits
 Organic Strawberries   |       10894 | fresh fruits
 Organic Baby Spinach   |        9784 | packaged vegetables fruits
 Large Lemon            |        8135 | fresh fruits
 Organic Avocado        |        7409 | fresh fruits
 Organic Hass Avocado   |        7293 | fresh fruits
 Strawberries           |        6494 | fresh fruits
 Limes                  |        6033 | fresh fruits
 Organic Raspberries    |        5546 | packaged vegetables fruits
 */

/* Top 10 earliest ordered products (in same order) that's ordered "often"
"often" = product ordered at least 10 times
 */
SELECT p.product_name, op.num_ordered, op.avg_order, a.aisle
  FROM (
    SELECT op.product_id, AVG(op.add_to_cart_order) avg_order, COUNT(*) num_ordered
      FROM order_products__train op
  GROUP BY op.product_id
    HAVING COUNT(op.product_id) > 10
  ORDER BY avg_order ASC
  ) AS op
  JOIN products p ON p.product_id = op.product_id
  JOIN aisles a ON p.aisle_id = a.aisle_id
 LIMIT 10;

/*
                       product_name                        | num_ordered |     avg_order      |             aisle
-----------------------------------------------------------+-------------+--------------------+-------------------------------
 Irish Whiskey Ireland                                     |          16 | 2.1250000000000000 | spirits
 2% Lactose Free Milk                                      |          92 | 2.1956521739130435 | milk
 Organic Dark Roast                                        |          25 | 2.2800000000000000 | coffee
 Lemon & Lime Blossom Ocean Fresh Scent Disinfecting Wipes |          14 | 2.3571428571428571 | cleaning products
 High Efficiency Complete Dual Formula                     |          14 | 2.3571428571428571 | laundry
 Truvia Sweetener Packets                                  |          15 | 2.4666666666666667 | baking ingredients
 Sparkling Water, Bottles                                  |          69 | 2.5362318840579710 | water seltzer sparkling water
 Bourbon Kentucky Frontier Whiskey                         |          13 | 2.5384615384615385 | spirits
 Vegetable Juice                                           |          11 | 2.5454545454545455 | missing
 Spiced Rum                                                |          12 | 2.5833333333333333 | spirits
 */


/*
List out how many products were ordered N times.
For example, 7,884 products were ordered once only.
*/

SELECT ordered_n_times, COUNT(*) items_ordered_N_times
  FROM
    (SELECT op.product_id, COUNT(*) ordered_n_times
       FROM order_products__train op
   GROUP BY op.product_id
   ORDER BY ordered_n_times ASC) as op
GROUP BY ordered_n_times
ORDER BY items_ordered_N_times DESC;

/*
Order with the most number of items and its item count: 80
*/
    SELECT order_id, COUNT(*) num_items_in_cart
      FROM order_products__train
  GROUP BY order_id
  ORDER BY num_items_in_cart DESC
     LIMIT 1;

/*
Average number of items in an order:  10.5527593381551570
*/
SELECT AVG(op.num_items_in_cart)
  FROM (
    SELECT order_id, COUNT(*) num_items_in_cart
      FROM order_products__train
  GROUP BY order_id) AS op;

/*
The sample standard deviation of the number items in a cart: 7.9328467650272154
Use sample standard deviation because we don't have the entire population of orders.
*/
SELECT STDDEV_SAMP(op.num_items_in_cart)
  FROM (
    SELECT order_id, COUNT(*) num_items_in_cart
      FROM order_products__train
  GROUP BY order_id) AS op;