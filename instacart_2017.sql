CREATE TYPE eval_set AS ENUM ('prior', 'train', 'test');

CREATE TABLE orders (
	order_id serial PRIMARY KEY,
	user_id serial,
	order_eval_set eval_set,
	order_number smallserial,
	order_dow smallserial,
	order_hour_of_day smallserial,
	days_since_prior smallserial DEFAULT NULL /*  CONSTRAINT positive_days_since_prior CHECK */
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

COPY orders(order_id, user_id, order_eval_set, order_number, order_dow, order_hour_of_day, days_since_prior) FROM '/Users/ctang/Downloads/instacart_2017_05_01/orders.csv' DELIMITER ',' CSV HEADER;
COPY departments(department_id, department) FROM '/Users/ctang/Downloads/instacart_2017_05_01/departments.csv' DELIMITER ',' CSV HEADER;
COPY aisles(aisle_id, aisle) FROM '/Users/ctang/Downloads/instacart_2017_05_01/aisles.csv' DELIMITER ',' CSV HEADER;
COPY products(product_id, product_name, aisle_id, department_id) FROM '/Users/ctang/Downloads/instacart_2017_05_01/products.csv' DELIMITER ',' CSV HEADER;

select p.product_name, d.department from products p inner join departments d on p.department_id = d.department_id;
select d.department, count(*) as product_count from products p inner join departments d on p.department_id = d.department_id group by d.department order by product_count desc;